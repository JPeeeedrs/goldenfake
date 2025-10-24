import os
import re
import json
import time
import logging
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Substitui o carregamento via variáveis de ambiente por leitura direta de .env
_DEF_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

def _read_env(path: str = _DEF_ENV_PATH) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        if not os.path.isfile(path):
            return env
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, val = s.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    env[key] = val
    except Exception:
        logger.debug("Falha ao ler .env", exc_info=True)
    return env

ENV = _read_env()

# Domínios de checagem de fatos PT/ES comuns
FACT_CHECK_DOMAINS = {
    "aosfatos.org": "Aos Fatos",
    "lupa.uol.com.br": "Agência Lupa",
    "boatos.org": "Boatos.org",
    "e-farsas.com": "E-Farsas",
    "g1.globo.com/fato-ou-fake": "G1 Fato ou Fake",
    "checamos.afp.com": "AFP Checamos",
    "checagem.afp.com": "AFP Checagem",
    "chequeado.com": "Chequeado",
}

# Confiabilidade por domínio (ajustável). 0.0-1.0
TRUSTED_DOMAINS = {
    # Fact-checkers (muito alta)
    "aosfatos.org": 1.0,
    "lupa.uol.com.br": 1.0,
    "boatos.org": 0.9,
    "e-farsas.com": 0.9,
    "g1.globo.com": 0.9,
    "checamos.afp.com": 1.0,
    "checagem.afp.com": 1.0,
    "chequeado.com": 0.95,
    # Veículos tradicionais (alta)
    "g1.globo.com": 0.9,
    "bbc.com": 0.9,
    "bbc.co.uk": 0.9,
    "folha.uol.com.br": 0.85,
    "estadao.com.br": 0.85,
    "uol.com.br": 0.8,
    "cnnbrasil.com.br": 0.85,
    "veja.abril.com.br": 0.8,
}

# Penalização para redes sociais: reduzir confiança de fontes como Instagram, X/Twitter, Facebook, Pinterest, Bluesky, Threads
SOCIAL_DOMAINS = {
    "instagram.com", "x.com", "twitter.com", "facebook.com", "fb.com", "pinterest.com", "bsky.app", "bsky.social", "threads.net",
}
SOCIAL_PUBLISHER_NAMES = {"instagram", "x", "twitter", "facebook", "pinterest", "bluesky", "threads"}


def _get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        # remover 'www.'
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _domain_weight(url: str) -> float:
    d = _get_domain(url)
    if not d:
        return 0.5
    # Match por subdomínio também
    for known, w in TRUSTED_DOMAINS.items():
        if d == known or d.endswith("." + known):
            return float(w)
    # Heurísticas para domínios acadêmicos e governamentais
    try:
        if d.endswith(".gov") or d.endswith(".gov.br"):
            return 0.95
        if d.endswith(".edu") or d.endswith(".edu.br") or ".ac." in d:
            return 0.9
        # Universidades BR comuns (usp, unicamp, unesp, e UF*)
        if re.search(r"(^|\.)((usp|unicamp|unesp|uf[a-z]{1,3}|ufrj|ufmg|ufba|ufms|ufpb|ufpr|ufsc|ufrn))(\.|$)", d):
            return 0.9
    except Exception:
        pass
    return 0.5  # neutro quando desconhecido


def _is_social_source(url: str, publisher: str | None) -> bool:
    d = _get_domain(url)
    if d:
        for sd in SOCIAL_DOMAINS:
            if d == sd or d.endswith("." + sd):
                return True
    if publisher:
        p = publisher.strip().lower()
        # normalizar caracteres comuns
        p = re.sub(r"[^a-z0-9]+", " ", p)
        tokens = set(p.split())
        if tokens & SOCIAL_PUBLISHER_NAMES:
            return True
    return False


def extract_claims(text: str, max_claims: int = 3) -> List[str]:
    """Extrai afirmações simples por sentença com heurísticas leves."""
    # Split por pontuação forte
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    # Limpeza básica e filtro por tamanho
    claims = []
    for p in parts:
        c = p.strip()
        if len(c.split()) >= 8 and not c.endswith(":"):
            claims.append(c)
        if len(claims) >= max_claims:
            break
    # Fallback: se nada, pega o texto inteiro
    if not claims and text:
        claims = [text.strip()]
    return claims


def _cosine_sim(model: SentenceTransformer, a: str, b: str) -> float:
    embs = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def query_google_factcheck(claim: str, api_key: str, language: str = "pt-BR") -> List[Dict[str, Any]]:
    if not api_key:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "languageCode": language,
        "pageSize": 10,
        "key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            logger.warning(f"FactCheck API status {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        items = []
        for cl in data.get("claims", []) or []:
            base_text = cl.get("text") or ""
            for rev in cl.get("claimReview", []) or []:
                items.append({
                    "provider": "google-fact-check",
                    "title": rev.get("title") or base_text[:120],
                    "url": rev.get("url"),
                    "publisher": (rev.get("publisher") or {}).get("name"),
                    "rating": rev.get("textualRating"),
                    "claim_text": base_text,
                })
        return items
    except Exception as e:
        logger.exception("Erro na consulta ao Google Fact Check Tools")
        return []


def query_newsapi(claim: str, api_key: str, language: str = "pt") -> List[Dict[str, Any]]:
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": claim[:128],  # limitar consulta
        "language": language,
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            logger.warning(f"NewsAPI status {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        items = []
        for art in data.get("articles", []) or []:
            url_art = art.get("url") or ""
            domain = url_art.split("//")[-1].split("/")[0].lower()
            # Marcar se é de um verificador conhecido
            fc_name = None
            for fc_domain, name in FACT_CHECK_DOMAINS.items():
                if fc_domain in url_art.lower():
                    fc_name = name
                    break
            items.append({
                "provider": "newsapi",
                "title": art.get("title"),
                "url": url_art,
                "publisher": (art.get("source") or {}).get("name"),
                "rating": "fact-check-article" if fc_name else None,
                "fact_checker": fc_name,
                "claim_text": art.get("description") or art.get("title") or "",
            })
        return items
    except Exception:
        logger.exception("Erro na consulta ao NewsAPI")
        return []


# Novo: busca web genérica via SerpAPI (Google) – opcional
# Requer variável de ambiente SERPAPI_KEY

def query_serpapi(claim: str, api_key: str, hl: str = "pt-BR", gl: str = "br") -> List[Dict[str, Any]]:
    if not api_key:
        return []
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": claim,
        "hl": hl,
        "gl": gl,
        "num": 10,
        "api_key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            logger.warning(f"SerpAPI status {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        items = []
        for it in (data.get("organic_results") or []):
            link = it.get("link") or ""
            domain = _get_domain(link)
            # marcar se é verificador conhecido
            fc_name = None
            for fc_domain, name in FACT_CHECK_DOMAINS.items():
                if fc_domain in link.lower():
                    fc_name = name
                    break
            items.append({
                "provider": "serpapi",
                "title": it.get("title"),
                "url": link,
                "publisher": domain,
                "rating": "fact-check-article" if fc_name else None,
                "fact_checker": fc_name,
                "claim_text": it.get("snippet") or "",
            })
        return items
    except Exception:
        logger.exception("Erro na consulta ao SerpAPI")
        return []


# Novo: busca web via Bing Search API – opcional
# Requer variável de ambiente BING_SEARCH_KEY

def query_bing(claim: str, api_key: str, mkt: str = "pt-BR") -> List[Dict[str, Any]]:
    if not api_key:
        return []
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": claim,
        "mkt": mkt,
        "textDecorations": False,
        "textFormat": "Raw",
        "count": 15,
    }
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            logger.warning(f"Bing status {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        web = (data.get("webPages") or {}).get("value") or []
        items = []
        for it in web:
            link = it.get("url") or ""
            domain = _get_domain(link)
            fc_name = None
            for fc_domain, name in FACT_CHECK_DOMAINS.items():
                if fc_domain in link.lower():
                    fc_name = name
                    break
            items.append({
                "provider": "bing",
                "title": it.get("name"),
                "url": link,
                "publisher": domain,
                "rating": "fact-check-article" if fc_name else None,
                "fact_checker": fc_name,
                "claim_text": it.get("snippet") or "",
            })
        return items
    except Exception:
        logger.exception("Erro na consulta ao Bing Search API")
        return []


RATING_MAP = {
    # Normalização simples para números em [0,1]
    "true": 1.0,
    "verdadeiro": 1.0,
    "verídico": 1.0,
    "mostly true": 0.8,
    "mostly-true": 0.8,
    "mostly verdadeiro": 0.8,
    "half true": 0.5,
    "meia-verdade": 0.5,
    "partly true": 0.5,
    "partly falso": 0.5,
    "mixed": 0.5,
    "uncertain": 0.4,
    "inconclusivo": 0.4,
    "misleading": 0.2,
    "exaggerated": 0.2,
    "falso": 0.0,
    "false": 0.0,
}

# Stopwords básicas PT/EN para evitar contar palavras funcionais
STOPWORDS = {
    # pt
    "a","o","as","os","um","uma","uns","umas","de","do","da","dos","das","e","em","no","na","nos","nas","por","para","com","sem","sobre","entre","até","após","antes","como","que","se","sua","seu","suas","seus","é","foi","ser","são","era","ao","à","às","aos","mais","menos","muito","muita","muitas","muitos","já","não","sim","também","ou","onde","quando","porque","porquê","qual","quais","qualquer","toda","todo","todas","todos","há","teve","ter","tem","têm","desde","contra","meu","minha","meus","minhas",
    # en
    "the","a","an","and","or","of","in","on","for","to","from","by","with","without","as","at","that","this","these","those","is","are","was","were","be","been","being","it","its","into","their","there","here","not","yes","no","also","any","all","more","less",
}

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Captura palavras com letras latinas e números; normaliza para minúsculas
    tokens = re.findall(r"[\wÀ-ÿ]+", text.lower())
    out: List[str] = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        if len(t) < 3:
            continue
        out.append(t)
    return out


def _overlap_ratio(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    if not ta:
        return 0.0
    tb = set(_tokenize(b))
    inter = ta & tb
    return len(inter) / max(1, len(ta))


def _rating_to_score(text: str | None) -> float:
    if not text:
        return 0.5  # neutro quando não há rating
    t = text.strip().lower()
    # tentar chave direta
    if t in RATING_MAP:
        return RATING_MAP[t]
    # reduzir variações
    for key, val in RATING_MAP.items():
        if key in t:
            return val
    return 0.5


# Amortecedor global para reduzir a influência do score de fontes externas
EXTERNAL_SCORE_SCALE = 1.0

# Lista de domínios de notícias comuns para classificar rapidamente
NEWS_DOMAINS = {
    "g1.globo.com", "bbc.com", "bbc.co.uk", "uol.com.br", "folha.uol.com.br", "estadao.com.br",
    "cnn.com", "cnnbrasil.com.br", "nytimes.com", "washingtonpost.com", "reuters.com", "apnews.com",
    "elpais.com", "dw.com", "r7.com", "terra.com.br", "oglobo.globo.com", "gazetadopovo.com.br",
}

BLOG_DOMAINS = {"medium.com", "blogspot.com", "wordpress.com", "substack.com"}
FORUM_DOMAINS = {"reddit.com", "quora.com", "stackexchange.com", "stackoverflow.com"}
VIDEO_DOMAINS = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "tiktok.com"}
WIKI_DOMAINS = {"wikipedia.org", "wikinews.org", "wikiversity.org", "wikibooks.org"}
PR_DOMAINS = {"prnewswire.com", "businesswire.com", "globenewswire.com"}


def _has_path_blog(url: str) -> bool:
    try:
        p = urlparse(url)
        return "/blog" in (p.path or "").lower()
    except Exception:
        return False


def _source_tags(url: str, publisher: str | None, rating: str | None) -> List[str]:
    tags = set()
    d = _get_domain(url)
    pub = (publisher or "").lower().strip()

    # fact-check
    if rating and isinstance(rating, str):
        if "fact-check" in rating.lower() or rating.lower() in RATING_MAP or any(fc in (url or "").lower() for fc in FACT_CHECK_DOMAINS.keys()):
            tags.add("fact-check")
    else:
        for fc in FACT_CHECK_DOMAINS.keys():
            if fc in (url or "").lower():
                tags.add("fact-check")
                break

    # social
    if _is_social_source(url, publisher):
        tags.add("social")

    # gov/academic/wiki/video/forum/blog/news/pr
    if d:
        if d.endswith(".gov") or d.endswith(".gov.br"):
            tags.add("gov")
        if d.endswith(".edu") or d.endswith(".edu.br") or ".ac." in d:
            tags.add("academic")
        if any(d == w or d.endswith("." + w) for w in WIKI_DOMAINS):
            tags.add("wiki")
        if any(d == v or d.endswith("." + v) for v in VIDEO_DOMAINS):
            tags.add("video")
        if any(d == f or d.endswith("." + f) for f in FORUM_DOMAINS):
            tags.add("forum")
        if any(d == b or d.endswith("." + b) for b in BLOG_DOMAINS) or _has_path_blog(url) or "blog" in pub:
            tags.add("blog")
        if any(d == n or d.endswith("." + n) for n in NEWS_DOMAINS) or "noticias" in d or "news" in d or "globo.com" in d:
            tags.add("news")
        if any(d == pr or d.endswith("." + pr) for pr in PR_DOMAINS):
            tags.add("press-release")

    # publisher heuristics
    if pub:
        if "universidade" in pub or "universidad" in pub or "university" in pub:
            tags.add("academic")
        if "minist" in pub or "prefeitura" in pub or "governo" in pub or "gov" == pub:
            tags.add("gov")
        if "noticias" in pub or "news" in pub or "jornal" in pub:
            tags.add("news")

    return sorted(tags)


def evaluate_claim_against_results(claim: str, results: List[Dict[str, Any]], sbert: SentenceTransformer,
                                   sim_threshold: float = 0.25, topk: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """Retorna (score_claim, detalhes_topk).
    Regras de overlap:
      - ov >= 0.50: peso total (factor=1.0)
      - 0.40 <= ov < 0.50: penaliza (factor=0.95)
      - ov < 0.40: peso bem baixo (factor=0.85)

    Cálculo de score final por fontes externas: média das 10 primeiras evidências (padding implícito).
    Mesmo que haja menos de 10 evidências, a soma é dividida por 10.
    Limite rígido: o número de evidências usadas na base do cálculo não pode exceder 10.
    """
    scored = []
    for r in results:
        ref_text = r.get("claim_text") or r.get("title") or ""
        sim = _cosine_sim(sbert, claim, ref_text) if ref_text else 0.0
        base_score = _rating_to_score(r.get("rating"))
        # calcular confiança do domínio e aplicar penalização se for rede social
        url = r.get("url") or ""
        publisher = r.get("publisher") or ""
        trust = _domain_weight(url)
        social = _is_social_source(url, publisher)
        if social:
            trust = min(trust, 0.2)  # reduzir confiança para fontes de redes sociais
        # Overlap de palavras entre a afirmação e o texto da evidência
        ov = _overlap_ratio(claim, ref_text)
        if ov >= 0.50:
            ov_factor = 1.0
            ov_bucket = ">=50%"
            pass50 = True
            pass40 = True
        elif ov >= 0.40:
            ov_factor = 0.95  # mais brando
            ov_bucket = "40-49%"
            pass50 = False
            pass40 = True
        else:
            ov_factor = 0.85  # mais brando
            ov_bucket = "<40%"
            pass50 = False
            pass40 = False
        # Boost baseado em confiança do domínio para fontes fortes
        if trust >= 0.85:
            if ov_bucket == "40-49%":
                ov_factor = max(ov_factor, 0.95)
            elif ov_bucket == "<40%":
                ov_factor = max(ov_factor, 0.90)
        # Combinação base: se há rating (fact-check), dar mais peso ao rating e à similaridade
        if r.get("rating"):
            comb = 0.6 * base_score + 0.25 * max(sim, 0.0) + 0.15 * trust
        else:
            comb = 0.45 * max(sim, 0.0) + 0.40 * trust + 0.05 * ov
        comb_eff = comb * ov_factor
        final_eff = comb_eff * EXTERNAL_SCORE_SCALE
        r2 = dict(r)
        r2.update({
            "similaridade": round(sim, 3),
            "confianca_fonte": round(trust, 2),
            "overlap_ratio": round(ov, 3),
            "overlap_bucket": ov_bucket,
            "overlap_factor": round(ov_factor, 2),
            "passes_50pct": bool(pass50),
            "passes_40pct": bool(pass40),
            "score": round(final_eff, 3),
            "is_social": bool(social),
            "source_tags": _source_tags(url, publisher, r.get("rating")),
        })
        scored.append(r2)
    # filtrar apenas por similaridade mínima (overlap agora ajusta peso, não exclui)
    scored = [x for x in scored if x["similaridade"] >= sim_threshold]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Impõe limite rígido de 10 evidências no cálculo, independentemente do valor recebido em topk
    max_base = 10
    try:
        topk = int(topk)
    except Exception:
        topk = max_base
    topk = min(max_base, max(0, topk))

    top = scored[:topk]
    # Média das 10 primeiras evidências; se houver menos de 10, dividir por 10 mesmo assim (padding zero)
    if not top:
        return 0.0, []
    sum_top_scores = float(sum(it.get("score", 0.0) for it in top))  # score está em [0,1]
    avg_top10_pct = (sum_top_scores / 10.0) * 100.0
    return avg_top10_pct, top


def verify_with_external_sources(text: str, sbert: SentenceTransformer) -> Tuple[float, List[Dict[str, Any]]]:
    """Calcula fonte_score (0-100) e retorna detalhes por afirmação.
    Integra Google Fact Check, NewsAPI e, opcionalmente, SerpAPI/Bing para busca geral.

    Regra de pontuação global:
    - O score final (fonte_score) é a média dos TOP 10 resultados globais considerando TODAS as afirmações (padding zero),
      isto é, soma dos scores dos 10 melhores resultados dividida por 10, multiplicada por 100.
      Dessa forma, o denominador é sempre 10, independente de quantos sites forem encontrados.
    """
    claims = extract_claims(text)

    # Ler chaves somente do .env local
    google_key = ENV.get("FACT_CHECK_API_KEY") or ENV.get("GOOGLE_FACTCHECK_API_KEY")
    newsapi_key = ENV.get("NEWSAPI_KEY") or ENV.get("NEWS_API_KEY")
    serpapi_key = ENV.get("SERPAPI_KEY")
    bing_key = ENV.get("BING_SEARCH_KEY")

    details_all: List[Dict[str, Any]] = []
    per_claim_scores: List[float] = []
    # Acumular evidências globalmente para cálculo do TOP 10 geral
    global_evidences: List[Dict[str, Any]] = []

    for c in claims:
        results: List[Dict[str, Any]] = []
        # Google Fact-Check Tools
        g_res = query_google_factcheck(c, google_key)
        if g_res:
            results.extend(g_res)
        # NewsAPI (com filtro de domínios de fact-check)
        n_res = query_newsapi(c, newsapi_key)
        if n_res:
            results.extend(n_res)
        # SerpAPI (Google) – emula busca humana
        s_res = query_serpapi(c, serpapi_key)
        if s_res:
            results.extend(s_res)
        # Bing Web Search – alternativa
        b_res = query_bing(c, bing_key)
        if b_res:
            results.extend(b_res)

        score_c, top = evaluate_claim_against_results(c, results, sbert)
        per_claim_scores.append(score_c)
        details_all.append({
            "afirmacao": c,
            "score_afirmacao": round(score_c, 1),
            "evidencias": top,
        })
        # Acumular para ranking global
        if top:
            global_evidences.extend(top)
        # Evitar rate limits agressivos
        time.sleep(0.3)

    # Cálculo do score final com divisor fixo 10 (TOP 10 global)
    if global_evidences:
        global_evidences.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top10_global = global_evidences[:10]
        sum_scores = float(sum(it.get("score", 0.0) for it in top10_global))  # scores em [0,1]
        fonte_score = (sum_scores / 10.0) * 100.0
    else:
        fonte_score = 0.0

    return fonte_score, details_all
