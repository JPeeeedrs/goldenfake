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
    return 0.5  # neutro quando desconhecido


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


def evaluate_claim_against_results(claim: str, results: List[Dict[str, Any]], sbert: SentenceTransformer,
                                   sim_threshold: float = 0.3, topk: int = 3) -> Tuple[float, List[Dict[str, Any]]]:
    """Retorna (score_claim, detalhes_topk)."""
    scored = []
    for r in results:
        ref_text = r.get("claim_text") or r.get("title") or ""
        sim = _cosine_sim(sbert, claim, ref_text) if ref_text else 0.0
        base_score = _rating_to_score(r.get("rating"))
        trust = _domain_weight(r.get("url") or "")
        # Se há rating (artigo de checagem), dar mais peso ao rating; caso contrário, dar mais peso à similaridade+confiança
        if r.get("rating"):
            comb = 0.6 * base_score + 0.3 * max(sim, 0.0) + 0.1 * trust
        else:
            comb = 0.6 * max(sim, 0.0) + 0.4 * trust
        r2 = dict(r)
        r2.update({
            "similaridade": round(sim, 3),
            "confianca_fonte": round(trust, 2),
            "score": round(comb, 3),
        })
        scored.append(r2)
    # filtrar por similaridade mínima
    scored = [x for x in scored if x["similaridade"] >= sim_threshold]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:topk]
    if not top:
        return 0.0, []
    return float(top[0]["score"]) * 100.0, top


def verify_with_external_sources(text: str, sbert: SentenceTransformer) -> Tuple[float, List[Dict[str, Any]]]:
    """Calcula fonte_score (0-100) e retorna detalhes por afirmação.
    Integra Google Fact Check, NewsAPI e, opcionalmente, SerpAPI/Bing para busca geral.
    """
    claims = extract_claims(text)

    # Ler chaves somente do .env local
    google_key = ENV.get("FACT_CHECK_API_KEY") or ENV.get("GOOGLE_FACTCHECK_API_KEY")
    newsapi_key = ENV.get("NEWSAPI_KEY") or ENV.get("NEWS_API_KEY")
    serpapi_key = ENV.get("SERPAPI_KEY")
    bing_key = ENV.get("BING_SEARCH_KEY")

    details_all: List[Dict[str, Any]] = []
    per_claim_scores: List[float] = []

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
        # Evitar rate limits agressivos
        time.sleep(0.3)

    fonte_score = float(np.mean(per_claim_scores)) if per_claim_scores else 0.0
    return fonte_score, details_all
