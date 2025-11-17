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

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - spaCy pode não estar disponível em runtime
    spacy = None


_SPACY_NLP = None
ENTITY_TYPES = {
    "PERSON",
    "PER",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "EVENT",
    "NORP",
    "PRODUCT",
    "WORK_OF_ART",
    "LAW",
    "MISC",
}
MAX_ENTITY_CHECKS = 18
ENTITY_SEARCH_SLEEP = 0.25
ENTITY_STRONG_SCORE = 1.0
ENTITY_WEAK_SCORE = 0.35
ENTITY_MISSING_SCORE = 0.0
ENTITY_STATUS_LABELS = {
    "strong": "Corroboração forte",
    
    "weak": "Corroboração fraca",
    "missing": "Bolha de vácuo",
}
TITLE_PREFIXES = {
    "dr",
    "dr.",
    "dra",
    "dra.",
    "prof",
    "prof.",
    "profª",
    "profª.",
    "sr",
    "sr.",
    "sra",
    "sra.",
}

GENERIC_ENTITY_PREFIXES = {
    "cientista",
    "cientistas",
    "pesquisador",
    "pesquisadores",
    "organismo",
    "campo",
    "teoricamente",
    "investigação",
    "investigadores",
    "relatório",
    "descoberta",
    "metal",
    "banda",
}

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

def _get_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        _SPACY_NLP = None
        return None
    try:
        _SPACY_NLP = spacy.load("pt_core_news_sm")
    except Exception:
        logger.warning("Falha ao carregar modelo spaCy pt_core_news_sm", exc_info=True)
        _SPACY_NLP = None
    return _SPACY_NLP


def _fallback_entities(text: str, limit: int) -> List[str]:
    pattern = re.compile(
        r"([A-ZÁÉÍÓÚÂÊÔÃÕ][\wÁÉÍÓÚÂÊÔÃÕçãõà-ü'-]+(?:\s+(?:de|da|do|dos|das|e|para|del|di|la|el|van|von)?\s*[A-ZÁÉÍÓÚÂÊÔÃÕ][\wÁÉÍÓÚÂÊÔÃÕçãõà-ü'-]+){0,5})"
    )
    seen = set()
    entities: List[str] = []
    for match in pattern.finditer(text):
        candidate = match.group(0).strip()
        words = candidate.split()
        if len(words) < 2:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(candidate)
        if len(entities) >= limit:
            break
    return entities


def _clean_entity_value(value: str) -> str:
    value = value or ""
    # remover citações do tipo [n] ou [n]
    value = re.sub(r"\[\d+\]?", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,.;:\"'[]")


def _normalize_entity_key(value: str) -> str:
    cleaned = _clean_entity_value(value).lower()
    cleaned = re.sub(r"[\.,;:\"'`´()\[\]]+", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    while tokens:
        first = tokens[0].rstrip(".")
        if first in TITLE_PREFIXES:
            tokens = tokens[1:]
            continue
        break
    if not tokens:
        return cleaned.strip()
    return " ".join(tokens)


def _should_skip_entity(value: str, label: str | None) -> bool:
    raw = _clean_entity_value(value)
    key = _normalize_entity_key(raw)
    if not key:
        return True
    tokens = key.split()
    if not tokens:
        return True
    first = tokens[0]
    if first in GENERIC_ENTITY_PREFIXES:
        return True
    if label in {"MISC"} and len(tokens) < 2:
        token_raw = raw.strip()
        if token_raw.isalpha() and token_raw.upper() == token_raw and len(token_raw) >= 4:
            return False
        return True
    return False


def _collect_following_tokens(
    doc,
    start: int,
    max_tokens: int = 4,
    skip_initial_punct: bool = False,
) -> List[str]:
    tokens: List[str] = []
    idx = start
    captured = 0
    while idx < len(doc) and captured < max_tokens:
        tok = doc[idx]
        if tok.is_space:
            idx += 1
            continue
        text = tok.text.strip()
        if not text:
            idx += 1
            continue
        if tok.is_punct:
            if not tokens and skip_initial_punct:
                idx += 1
                continue
            break
        if re.match(r"^\[\d+\]?$", text):
            idx += 1
            continue
        if text in {"-", "·"}:
            tokens.append(text)
            idx += 1
            continue
        if len(text) == 1 and text.lower() in {"o", "a", "e"}:
            idx += 1
            continue
        if text[0].isupper() or text.isupper() or tok.ent_type_ in ENTITY_TYPES:
            tokens.append(text)
            captured += 1
            idx += 1
            continue
        break
    return tokens


def _collect_parenthetical(doc, start: int, max_len: int = 6) -> List[str]:
    if start >= len(doc) or doc[start].text != "(":
        return []
    tokens = []
    depth = 0
    idx = start
    while idx < len(doc) and len(tokens) < max_len:
        tok = doc[idx]
        tokens.append(tok.text)
        if tok.text == "(":
            depth += 1
        elif tok.text == ")":
            depth -= 1
            if depth == 0:
                break
        idx += 1
    if depth == 0 and tokens:
        return tokens
    return []


def _normalize_entity_span(ent) -> str:
    value = _clean_entity_value(ent.text)
    doc = ent.doc
    lower_value = value.lower().rstrip(".")
    is_person_label = ent.label_ in {"PERSON", "PER"}
    needs_title_extension = lower_value in TITLE_PREFIXES
    if is_person_label and (needs_title_extension or len(value.split()) < 2):
        extra = _collect_following_tokens(
            doc,
            ent.end,
            max_tokens=4,
            skip_initial_punct=True,
        )
        if extra:
            value = _clean_entity_value(f"{value} {' '.join(extra)}")
    elif needs_title_extension:
        extra = _collect_following_tokens(
            doc,
            ent.end,
            max_tokens=4,
            skip_initial_punct=True,
        )
        if extra:
            value = _clean_entity_value(f"{value} {' '.join(extra)}")
    if ent.label_ in {"ORG", "GPE", "FAC", "LOC"}:
        paren = _collect_parenthetical(doc, ent.end)
        if paren:
            value = _clean_entity_value(f"{value} {' '.join(paren)}")
    return value


def extract_entities_for_verification(text: str, limit: int = MAX_ENTITY_CHECKS) -> List[str]:

    text = (text or "").strip()
    if not text:
        return []
    nlp = _get_spacy_model()
    entities: List[str] = []
    seen = set()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in ENTITY_TYPES:
                continue
            value = _normalize_entity_span(ent)
            if len(value) < 3:
                continue
            if _should_skip_entity(value, ent.label_):
                continue
            key = _normalize_entity_key(value)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            entities.append(value)
            if len(entities) >= limit:
                break
    if entities:
        return entities
    fallback = _fallback_entities(text, limit)
    if fallback:
        deduped = []
        seen.clear()
        for value in fallback:
            if _should_skip_entity(value, None):
                continue
            key = _normalize_entity_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(value)
            if len(deduped) >= limit:
                break
        return deduped
    return []


_GEMINI_MODEL_DEFAULT = "gemini-1.5-flash"
_GEMINI_MODEL = ENV.get("GEMINI_MODEL") or ENV.get("GEMINI_MODEL_NAME")
if isinstance(_GEMINI_MODEL, str):
    _GEMINI_MODEL = _GEMINI_MODEL.strip()
if not _GEMINI_MODEL:
    _GEMINI_MODEL = _GEMINI_MODEL_DEFAULT
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"
)

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


def _classify_entity_results(results: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not results:
        return "missing", ENTITY_MISSING_SCORE
    trusted_hits = 0
    weak_hits = 0
    for r in results:
        url = r.get("url") or ""
        publisher = r.get("publisher") or ""
        tags = _source_tags(url, publisher, r.get("rating"))
        trust = _domain_weight(url)
        if "wiki" in tags or "news" in tags or trust >= 0.85:
            trusted_hits += 1
        elif "blog" in tags or "forum" in tags or "social" in tags or trust <= 0.5:
            weak_hits += 1
    if trusted_hits:
        return "strong", ENTITY_STRONG_SCORE
    if weak_hits:
        return "weak", ENTITY_WEAK_SCORE
    return "weak", 0.5


def verify_entities_with_serpapi(text: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        return {}
    entities = extract_entities_for_verification(text)
    if not entities:
        return {}
    results_block: List[Dict[str, Any]] = []
    strong = weak = missing = 0
    for ent in entities:
        try:
            res = query_serpapi(ent, api_key)
        except Exception:
            res = []
        top = res[:5]
        for item in top:
            url = item.get("url") or ""
            tags = _source_tags(url, item.get("publisher"), item.get("rating"))
            item["source_tags"] = tags
            item["confianca_fonte"] = round(_domain_weight(url), 2)
        status, score = _classify_entity_results(top)
        if status == "strong":
            strong += 1
        elif status == "missing":
            missing += 1
        else:
            weak += 1
        results_block.append({
            "entidade": ent,
            "status": status,
            "rotulo": ENTITY_STATUS_LABELS.get(status, status),
            "score": round(score, 2),
            "resultados": top,
            "total_resultados": len(res),
        })
        time.sleep(ENTITY_SEARCH_SLEEP)
    if not results_block:
        return {}
    avg_score = sum(item["score"] for item in results_block) / max(1, len(results_block))
    return {
        "entidades": results_block,
        "media_score": round(avg_score, 3),
        "media_percent": round(avg_score * 100.0, 1),
        "total": len(results_block),
        "fortes": strong,
        "fracas": weak,
        "ausentes": missing,
    }


def _extract_json_dict(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    snippet = text.strip()
    if snippet.startswith("```"):
        parts = snippet.split("```")
        if len(parts) >= 2:
            snippet = parts[1]
    snippet = snippet.strip()
    if snippet.lower().startswith("json"):
        snippet = snippet[4:].strip()
    try:
        return json.loads(snippet)
    except Exception:
        pass
    start = snippet.find("{")
    end = snippet.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(snippet[start:end + 1])
        except Exception:
            return None
    return None


def query_gemini_factcheck(claim: str, api_key: str, context: List[Dict[str, Any]] | None = None) -> Dict[str, Any] | None:
    if not api_key or not claim:
        return None
    evidence_snippets: List[str] = []
    for item in (context or [])[:3]:
        title = item.get("title") or item.get("url") or "(sem título)"
        publisher = item.get("publisher") or item.get("provider") or ""
        evidence_snippets.append(f"- {title} ({publisher})")
    instructions = [
        "Você é um checador de fatos especializado em português.",
        "Analise a afirmação abaixo considerando as evidências disponíveis (se houver) e determine se ela é verdadeira, falsa ou inconclusiva.",
        "Responda estritamente em JSON com as chaves: verdict (true/false/inconclusivo), confidence (0-1), justification (texto curto), evidence_used (array de strings).",
        f"Afirmação: {claim}",
    ]
    if evidence_snippets:
        instructions.append("Referências detectadas:")
        instructions.extend(evidence_snippets)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "\n".join(instructions)
                    }
                ]
            }
        ]
    }
    try:
        resp = requests.post(
            GEMINI_API_URL,
            params={"key": api_key},
            json=payload,
            timeout=20,
        )
        if resp.status_code != 200:
            logger.warning("Gemini API status %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        candidates = data.get("candidates") or []
        for cand in candidates:
            parts = (cand.get("content") or {}).get("parts") or []
            for part in parts:
                txt = part.get("text")
                parsed = _extract_json_dict(txt)
                if not isinstance(parsed, dict):
                    continue
                verdict = (parsed.get("verdict") or "").strip().lower()
                if not verdict and isinstance(parsed.get("verdict"), bool):
                    verdict = "true" if parsed["verdict"] else "false"
                    if verdict in {"verdadeiro", "true", "falso", "false", "inconclusivo", "unknown", "incerto", "uncertain"}:
                        return parsed
        return None
    except Exception:
        logger.exception("Erro na consulta ao Gemini API")
        return None


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

FACT_CHECK_TRUE_KEYWORDS = {
    "true",
    "verdadeiro",
    "verídico",
    "correct",
    "correto",
    "accurate",
    "real",
    "confirmed",
    "confirmado",
}

FACT_CHECK_FALSE_KEYWORDS = {
    "false",
    "falso",
    "fake",
    "enganoso",
    "mentira",
    "boato",
    "hoax",
    "incorrect",
    "incorreto",
    "impreciso",
    "misleading",
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


def _fact_check_verdict(rating: str | None) -> str | None:
    if not rating:
        return None
    t = rating.strip().lower()
    for token in FACT_CHECK_FALSE_KEYWORDS:
        if token in t:
            return "false"
    for token in FACT_CHECK_TRUE_KEYWORDS:
        if token in t:
            return "true"
    return None


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
        if "ai.google" in d or "gemini" in d:
            tags.add("ai")
            tags.add("gemini")

    # publisher heuristics
    if pub:
        if "universidade" in pub or "universidad" in pub or "university" in pub:
            tags.add("academic")
        if "minist" in pub or "prefeitura" in pub or "governo" in pub or "gov" == pub:
            tags.add("gov")
        if "noticias" in pub or "news" in pub or "jornal" in pub:
            tags.add("news")
        if "gemini" in pub:
            tags.add("ai")
            tags.add("gemini")

    if rating and isinstance(rating, str) and "gemini" in rating.lower():
        tags.add("ai")
        tags.add("gemini")

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
        source_tags = _source_tags(url, publisher, r.get("rating"))
        # Always set confianca_fonte to 0.7 for wiki sources, otherwise use calculated trust
        confianca_fonte = 0.7 if "wiki" in source_tags else round(trust, 2)
        r2.update({
            "similaridade": round(sim, 3),
            "confianca_fonte": confianca_fonte,
            "overlap_ratio": round(ov, 3),
            "overlap_bucket": ov_bucket,
            "overlap_factor": round(ov_factor, 2),
            "passes_50pct": bool(pass50),
            "passes_40pct": bool(pass40),
            "score": round(final_eff, 3),
            "is_social": bool(social),
            "source_tags": source_tags,
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


def verify_with_external_sources(text: str, sbert: SentenceTransformer) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    """Calcula fonte_score (0-100), retorna detalhes por afirmação e resumo de entidades.
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
    gemini_key = ENV.get("GEMINI_API_KEY") or ENV.get("GOOGLE_GEMINI_API_KEY")
    entity_block: Dict[str, Any] = {}

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

        has_fact_rating = any((r.get("rating") or "").strip() for r in results)
        if not has_fact_rating and gemini_key:
            gemini_res = query_gemini_factcheck(c, gemini_key, context=results)
            if gemini_res:
                verdict_raw = (gemini_res.get("verdict") or "").strip().lower()
                verdict_map = {
                    "true": "verdadeiro (Gemini)",
                    "verdadeiro": "verdadeiro (Gemini)",
                    "false": "falso (Gemini)",
                    "falso": "falso (Gemini)",
                }
                rating_text = verdict_map.get(verdict_raw, "inconclusivo (Gemini)")
                conf = gemini_res.get("confidence")
                try:
                    conf_val = float(conf)
                    conf_val = max(0.0, min(1.0, conf_val))
                except Exception:
                    conf_val = None
                results.append({
                    "provider": "gemini",
                    "title": "Análise automática Gemini",
                    "url": "https://ai.google/gemini",
                    "publisher": "Gemini AI",
                    "rating": rating_text,
                    "fact_checker": "Gemini AI",
                    "claim_text": gemini_res.get("justification") or c,
                    "gemini_verdict": verdict_raw or "inconclusivo",
                    "gemini_confidence": conf_val,
                    "gemini_evidence_used": gemini_res.get("evidence_used"),
                })

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
    fonte_score = 0.0
    has_true_fact_check = False
    has_false_fact_check = False
    if global_evidences:
        global_evidences.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top10_global = global_evidences[:10]
        sum_scores = float(sum(it.get("score", 0.0) for it in top10_global))  # scores em [0,1]
        fonte_score = (sum_scores / 10.0) * 100.0
        for ev in global_evidences:
            verdict = _fact_check_verdict(ev.get("rating"))
            if verdict == "true":
                has_true_fact_check = True
            elif verdict == "false":
                has_false_fact_check = True
        if has_false_fact_check and not has_true_fact_check:
            fonte_score = 0.0
        elif has_false_fact_check and has_true_fact_check:
            fonte_score = 50.0

    try:
        entity_block = verify_entities_with_serpapi(text, serpapi_key)
    except Exception:
        logger.exception("Erro ao verificar entidades via SerpAPI")
        entity_block = {}
    if entity_block:
        avg_pct = entity_block.get("media_percent")
        if isinstance(avg_pct, (int, float)):
            fonte_score = min(fonte_score, float(avg_pct))

    return fonte_score, details_all, entity_block
