import os
import re
import json
import time
import logging
import copy
from collections import OrderedDict
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
ENTITY_SCORE_RULES = {
    "default": {
        "strong": ENTITY_STRONG_SCORE,
        "weak": ENTITY_WEAK_SCORE,
        "missing": ENTITY_MISSING_SCORE,
    },
    "kg": {
        "strong": 1.0,
        "weak": 0.5,
        "missing": 0.0,
    },
    "serp": {
        "strong": 0.75,
        "weak": 0.25,
        "missing": 0.0,
    },
}
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

FACT_CHECK_SIM_THRESHOLD = 0.3

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
        logger.warning(
            "Falha ao carregar modelo spaCy pt_core_news_sm", exc_info=True)
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

GEMINI_CACHE_MAX = 128
GEMINI_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
try:
    GEMINI_TIMEOUT_SECONDS = float(ENV.get("GEMINI_TIMEOUT", 15))
except Exception:
    GEMINI_TIMEOUT_SECONDS = 15.0
try:
    GEMINI_MAX_RETRIES = int(ENV.get("GEMINI_MAX_RETRIES", 3))
except Exception:
    GEMINI_MAX_RETRIES = 3


def _normalize_claim_for_cache(claim: str) -> str:
    return re.sub(r"\s+", " ", (claim or "").strip().lower())


def _get_cached_gemini_result(claim: str) -> Dict[str, Any] | None:
    key = _normalize_claim_for_cache(claim)
    if not key:
        return None
    cached = GEMINI_CACHE.get(key)
    if cached is None:
        return None
    GEMINI_CACHE.move_to_end(key)
    return copy.deepcopy(cached)


def _store_gemini_result(claim: str, result: Dict[str, Any]) -> None:
    key = _normalize_claim_for_cache(claim)
    if not key:
        return
    GEMINI_CACHE[key] = copy.deepcopy(result)
    GEMINI_CACHE.move_to_end(key)
    while len(GEMINI_CACHE) > GEMINI_CACHE_MAX:
        GEMINI_CACHE.popitem(last=False)


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
SOCIAL_PUBLISHER_NAMES = {"instagram", "x", "twitter",
                          "facebook", "pinterest", "bluesky", "threads"}


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
    embs = model.encode([a, b], convert_to_numpy=True,
                        normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def _filter_fact_checks_by_similarity(
    claim: str,
    evidences: List[Dict[str, Any]] | None,
    sbert: SentenceTransformer | None,
    threshold: float = FACT_CHECK_SIM_THRESHOLD,
) -> List[Dict[str, Any]]:
    if not evidences or not sbert or threshold <= 0:
        return [] if evidences is None else list(evidences)

    filtered: List[Dict[str, Any]] = []
    for ev in evidences:
        ref_text = ev.get("claim_text") or ev.get(
            "claim") or ev.get("title") or ""
        similarity = _cosine_sim(sbert, claim, ref_text) if ref_text else 0.0
        if similarity < threshold:
            continue
        ev_copy = dict(ev)
        ev_copy["_precomputed_similarity"] = similarity
        filtered.append(ev_copy)

    return filtered


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
            logger.warning(
                f"FactCheck API status {r.status_code}: {r.text[:200]}")
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


# Helpers específicos para o fluxo em dois níveis (fact-check oficial + investigação)
def _partition_newsapi_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fact_checks: List[Dict[str, Any]] = []
    general: List[Dict[str, Any]] = []
    for item in results or []:
        rating = (item.get("rating") or "").lower()
        fact_checker = item.get("fact_checker")
        if fact_checker or rating.startswith("fact-check"):
            fact_checks.append(item)
        else:
            general.append(item)
    return fact_checks, general


NIVEL_1 = "nivel1"
NIVEL_2 = "nivel2"


def _query_fact_checks_for_text(text: str, google_key: str | None, newsapi_key: str | None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evidences: List[Dict[str, Any]] = []
    general_news: List[Dict[str, Any]] = []
    snippet = (text or "").strip()
    if not snippet:
        return evidences, general_news
    query_text = snippet[:512]

    g_res = query_google_factcheck(query_text, google_key)
    if g_res:
        evidences.extend(g_res)

    if newsapi_key:
        news_results = query_newsapi(query_text, newsapi_key)
        if news_results:
            fact_checks, general = _partition_newsapi_results(news_results)
            if fact_checks:
                evidences.extend(fact_checks)
            if general:
                general_news.extend(general)

    return evidences, general_news


def _summarize_context_hits(results: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in (results or [])[:limit]:
        out.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "publisher": item.get("publisher"),
            "provider": item.get("provider"),
        })
    return out


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


def _classify_entity_results(results: List[Dict[str, Any]], provider: str | None = None) -> Tuple[str, float]:
    if not results:
        score_map = ENTITY_SCORE_RULES.get(
            provider) or ENTITY_SCORE_RULES["default"]
        return "missing", score_map["missing"]
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
        status = "strong"
    elif weak_hits:
        status = "weak"
    else:
        status = "weak"
    score_map = ENTITY_SCORE_RULES.get(
        provider) or ENTITY_SCORE_RULES["default"]
    return status, score_map[status]


def verify_entities_with_serpapi(text: str, api_key: str) -> Dict[str, Any]:
    # Prefer Knowledge Graph API when available; fall back to SerpAPI for entities not found.
    kg_key = ENV.get("KG_API_KEY") or ENV.get("GOOGLE_KG_KEY")
    if not api_key and not kg_key:
        return {}
    entities = extract_entities_for_verification(text)
    if not entities:
        return {}
    results_block: List[Dict[str, Any]] = []
    strong = weak = missing = 0
    for ent in entities:
        try:
            res: List[Dict[str, Any]] = []
            provider_used: str | None = None
            # try Knowledge Graph first
            if kg_key:
                try:
                    res = query_kg(ent, kg_key)
                    if res:
                        provider_used = "kg"
                except Exception:
                    logger.exception(
                        "Erro na consulta ao Knowledge Graph API para %s", ent)
                    res = []
            # fallback to SerpAPI only if KG returned nothing
            if not res and api_key:
                try:
                    res = query_serpapi(ent, api_key)
                    if res:
                        provider_used = "serp"
                except Exception:
                    logger.exception(
                        "Erro na consulta ao SerpAPI para %s", ent)
                    res = []
        except Exception:
            res = []
        top = res[:5]
        for item in top:
            url = item.get("url") or ""
            tags = _source_tags(url, item.get("publisher"), item.get("rating"))
            item["source_tags"] = tags
            item["confianca_fonte"] = round(_domain_weight(url), 2)
        status, score = _classify_entity_results(top, provider_used)
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
            "percent": round(score * 100.0, 1),
            "resultados": top,
            "total_resultados": len(res),
            "provider": provider_used,
        })
        time.sleep(ENTITY_SEARCH_SLEEP)
    if not results_block:
        return {}
    avg_score = sum(item["score"]
                    for item in results_block) / max(1, len(results_block))
    return {
        "entidades": results_block,
        "media_score": round(avg_score, 3),
        "media_percent": round(avg_score * 100.0, 1),
        "total": len(results_block),
        "fortes": strong,
        "fracas": weak,
        "ausentes": missing,
    }


def query_kg(query: str, api_key: str, limit: int = 5, languages: str = "pt") -> List[Dict[str, Any]]:
    """Consulta a Knowledge Graph Search API do Google e mapeia resultados para o formato usado internamente.

    Retorna lista de dicts com chaves parecidas com as de `query_serpapi`: provider, title, url, publisher, rating, claim_text
    """
    if not api_key or not query:
        return []
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key,
        "limit": int(limit),
        "languages": languages,
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            logger.warning("Knowledge Graph status %s: %s",
                           r.status_code, r.text[:200])
            return []
        data = r.json()
        items = []
        for el in (data.get("itemListElement") or [])[:limit]:
            res = el.get("result") or {}
            name = res.get("name")
            descr = res.get("description")
            dd = res.get("detailedDescription") or {}
            dd_url = dd.get("url")
            url_out = dd_url or res.get("@id") or ""
            title = name or dd.get("articleBody") or descr or url_out
            publisher = "Knowledge Graph"
            items.append({
                "provider": "kg",
                "title": title,
                "url": url_out,
                "publisher": publisher,
                "rating": None,
                "claim_text": descr or name or query,
            })
        return items
    except Exception:
        logger.exception("Erro na consulta ao Knowledge Graph API")
        return []


def query_gemini_factcheck(claim: str, api_key: str, context: List[Dict[str, Any]] | None = None) -> Dict[str, Any] | None:
    if not api_key or not claim:
        return None
    cached = _get_cached_gemini_result(claim)
    if cached:
        return cached
    evidences = (context or [])[:3]
    evidence_snippets: List[str] = []
    for item in evidences:
        title = item.get("title") or item.get("url") or "(sem título)"
        publisher = item.get("publisher") or item.get("provider") or ""
        snippet = item.get("snippet") or item.get(
            "description") or item.get("claim_text") or ""
        snippet_clean = snippet.strip().replace("\n", " ")[:260]
        evidence_snippets.append(
            f"- {title} ({publisher}): {snippet_clean}".strip())
    instructions = [
        "Você é um algoritmo de verificação cético.",
        "Analise a afirmação baseada EXCLUSIVAMENTE no contexto fornecido. Ignore qualquer conhecimento prévio.",
        "Pontuação obrigatória:",
        "0-20 se o contexto contradiz a afirmação ou se não existir contexto relevante (vácuo de informação).",
        "50 somente quando houver fontes conflitantes no contexto.",
        "80-100 apenas quando múltiplas fontes confirmarem explicitamente a afirmação.",
        "Retorne APENAS um número inteiro de 0 a 100, sem texto adicional, símbolos ou explicações.",
        f"Afirmação: {claim.strip()}",
    ]
    if evidence_snippets:
        instructions.append("Contexto:")
        instructions.extend(evidence_snippets)
    else:
        instructions.append(
            "Contexto: (nenhum fornecido; trate como ausência total de evidências)")
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
    last_error: str | None = None
    max_retries = max(1, GEMINI_MAX_RETRIES)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GEMINI_API_URL,
                params={"key": api_key},
                json=payload,
                timeout=GEMINI_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                logger.warning("Gemini API status %s: %s",
                               resp.status_code, resp.text[:200])
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            for cand in candidates:
                parts = (cand.get("content") or {}).get("parts") or []
                for part in parts:
                    raw_text = (part.get("text") or "").strip()
                    if not raw_text:
                        continue
                    match = re.search(r"\d+(?:\.\d+)?", raw_text)
                    if not match:
                        continue
                    try:
                        value = float(match.group())
                    except ValueError:
                        continue
                    score = max(0.0, min(100.0, value))
                    result = {
                        "score": score,
                        "raw_response": raw_text,
                        "context_used": list(evidence_snippets),
                    }
                    _store_gemini_result(claim, result)
                    return copy.deepcopy(result)
            # nenhum número identificado neste candidato; continuar tentativas
        except requests.exceptions.Timeout:
            last_error = f"timeout após {GEMINI_TIMEOUT_SECONDS:.1f}s"
            logger.warning(
                "Gemini API timeout (tentativa %s/%s, %.1fs)",
                attempt,
                max_retries,
                GEMINI_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            logger.warning(
                "Gemini API erro de rede (tentativa %s/%s): %s",
                attempt,
                max_retries,
                exc,
            )
        except Exception:
            last_error = "exceção inesperada"
            logger.exception("Gemini API erro inesperado")
        if attempt < max_retries:
            time.sleep(min(1.5 * attempt, 4.0))
    if last_error:
        logger.warning(
            "Gemini API indisponível após %s tentativas: %s", max_retries, last_error)
    fallback = {"score": 0.0, "raw_response": "0",
                "context_used": list(evidence_snippets)}
    _store_gemini_result(claim, fallback)
    return copy.deepcopy(fallback)


def _gemini_verdict_from_score(score: float | None) -> str:
    if score is None:
        return "unproven"
    if score < 30:
        return "refuted"
    if score > 70:
        return "supported"
    return "unproven"


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
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sobre", "entre", "até", "após", "antes", "como", "que", "se", "sua", "seu", "suas", "seus", "é", "foi", "ser", "são", "era", "ao", "à", "às", "aos", "mais", "menos", "muito", "muita", "muitas", "muitos", "já", "não", "sim", "também", "ou", "onde", "quando", "porque", "porquê", "qual", "quais", "qualquer", "toda", "todo", "todas", "todos", "há", "teve", "ter", "tem", "têm", "desde", "contra", "meu", "minha", "meus", "minhas",
    # en
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "from", "by", "with", "without", "as", "at", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "into", "their", "there", "here", "not", "yes", "no", "also", "any", "all", "more", "less",
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


def _claim_percent_from_fact_checks(evidences: List[Dict[str, Any]]) -> float | None:
    """Return 100 for true verdict, 0 for false, otherwise None."""
    for ev in evidences or []:
        verdict = _fact_check_verdict(ev.get("rating"))
        if verdict == "true":
            return 100.0
        if verdict == "false":
            return 0.0
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
FORUM_DOMAINS = {"reddit.com", "quora.com",
                 "stackexchange.com", "stackoverflow.com"}
VIDEO_DOMAINS = {"youtube.com", "youtu.be",
                 "vimeo.com", "dailymotion.com", "tiktok.com"}
WIKI_DOMAINS = {"wikipedia.org", "wikinews.org",
                "wikiversity.org", "wikibooks.org"}
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
        pre_sim = r.get("_precomputed_similarity")
        if isinstance(pre_sim, (int, float)):
            sim = float(pre_sim)
        else:
            sim = _cosine_sim(sbert, claim, ref_text) if ref_text else 0.0
        url = r.get("url") or ""
        publisher = r.get("publisher") or ""
        percent_override = r.get("percent_true")
        if isinstance(percent_override, (int, float)):
            base_score = max(0.0, min(1.0, percent_override / 100.0))
            trust = 0.75
            social = False
            ov = _overlap_ratio(claim, ref_text)
            ov_bucket = ">=50%" if ov >= 0.50 else "40-49%" if ov >= 0.40 else "<40%"
            pass50 = ov >= 0.50
            pass40 = ov >= 0.40
            ov_factor = 1.0
            comb_eff = base_score
            final_eff = base_score
        else:
            base_score = _rating_to_score(r.get("rating"))
            url = r.get("url") or ""
            publisher = r.get("publisher") or ""
            trust = _domain_weight(url)
            social = _is_social_source(url, publisher)
            if social:
                trust = min(trust, 0.2)
            ov = _overlap_ratio(claim, ref_text)
            if ov >= 0.50:
                ov_factor = 1.0
                ov_bucket = ">=50%"
                pass50 = True
                pass40 = True
            elif ov >= 0.40:
                ov_factor = 0.95
                ov_bucket = "40-49%"
                pass50 = False
                pass40 = True
            else:
                ov_factor = 0.85
                ov_bucket = "<40%"
                pass50 = False
                pass40 = False
            if trust >= 0.85:
                if ov_bucket == "40-49%":
                    ov_factor = max(ov_factor, 0.95)
                elif ov_bucket == "<40%":
                    ov_factor = max(ov_factor, 0.90)
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
        r2.pop("_precomputed_similarity", None)
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
    sum_top_scores = float(sum(it.get("score", 0.0)
                           for it in top))  # score está em [0,1]
    avg_top10_pct = (sum_top_scores / 10.0) * 100.0
    return avg_top10_pct, top


def verify_with_external_sources(text: str, sbert: SentenceTransformer) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any], Dict]:
    debug_log = {
        "claims": [],
        "apis_used": [],
        "entities": None,
        "total_claims": 0,
        "final_score": 0.0
    }

    claims = extract_claims(text)
    if not claims:
        claims = [text]

    debug_log["total_claims"] = len(claims)
    debug_log["claims"] = [
        {"text": c, "score": 0.0, "percent": 0.0} for c in claims]

    # Ler chaves somente do .env local
    google_key = ENV.get("FACT_CHECK_API_KEY") or ENV.get(
        "GOOGLE_FACTCHECK_API_KEY")
    newsapi_key = ENV.get("NEWSAPI_KEY") or ENV.get("NEWS_API_KEY")
    serpapi_key = ENV.get("SERPAPI_KEY")
    bing_key = ENV.get("BING_SEARCH_KEY")
    gemini_key = ENV.get("GEMINI_API_KEY") or ENV.get("GOOGLE_GEMINI_API_KEY")
    entity_block: Dict[str, Any] = {}

    details_all: List[Dict[str, Any]] = []
    claim_percentages: List[float] = []

    # Track APIs
    api_tracker = {
        "Google Fact Check": {"success": bool(google_key), "results_count": 0, "enabled": bool(google_key)},
        "NewsAPI": {"success": bool(newsapi_key), "results_count": 0, "enabled": bool(newsapi_key)},
        "SerpAPI": {"success": bool(serpapi_key), "results_count": 0, "enabled": bool(serpapi_key)},
        "Bing Search": {"success": bool(bing_key), "results_count": 0, "enabled": bool(bing_key)},
        "Gemini AI": {"success": bool(gemini_key), "results_count": 0, "enabled": bool(gemini_key)}
    }

    for i, c in enumerate(claims):
        _level1_evidences_raw, news_general_context = _query_fact_checks_for_text(
            c, google_key, newsapi_key)

        # Track Google Fact Check
        if google_key and _level1_evidences_raw:
            api_tracker["Google Fact Check"]["results_count"] += len(
                [e for e in _level1_evidences_raw if e.get("provider") == "google_factcheck"])

        # Track NewsAPI
        if newsapi_key:
            newsapi_results = len([e for e in _level1_evidences_raw if e.get(
                "provider") == "newsapi"]) + len(news_general_context)
            api_tracker["NewsAPI"]["results_count"] += newsapi_results

        level1_evidences = _filter_fact_checks_by_similarity(
            c, _level1_evidences_raw, sbert)
        gemini_block: Dict[str, Any] | None = None
        context_hits: List[Dict[str, Any]] = []
        nivel_utilizado = NIVEL_1 if level1_evidences else NIVEL_2
        nivel1_total = len(level1_evidences)
        nivel2_total = 0

        if level1_evidences:
            results = list(level1_evidences)
        else:
            nivel_utilizado = NIVEL_2
            nivel1_total = 0
            if news_general_context:
                context_hits.extend(news_general_context[:5])
            serp_hits = query_serpapi(c, serpapi_key)
            if serp_hits:
                context_hits.extend(serp_hits)
                api_tracker["SerpAPI"]["results_count"] += len(serp_hits)
            bing_hits = query_bing(c, bing_key)
            if bing_hits:
                context_hits.extend(bing_hits)
                api_tracker["Bing Search"]["results_count"] += len(bing_hits)
            nivel2_total = len(context_hits)
            results = list(context_hits)
            if gemini_key:
                gemini_res = query_gemini_factcheck(
                    c, gemini_key, context=context_hits)
                if gemini_res and isinstance(gemini_res.get("score"), (int, float)):
                    api_tracker["Gemini AI"]["results_count"] += 1
                    percent_true = max(
                        0.0, min(100.0, float(gemini_res["score"])))
                    verdict_label = _gemini_verdict_from_score(percent_true)
                    context_summary = _summarize_context_hits(context_hits)
                    gemini_block = {
                        "provider": "gemini",
                        "title": f"Estimativa cética Gemini ({percent_true:.0f}%)",
                        "url": "https://ai.google/gemini",
                        "publisher": "Gemini AI",
                        "rating": f"{percent_true:.0f}% apoio contextual",
                        "fact_checker": "Gemini AI",
                        "claim_text": c,
                        "percent_true": percent_true,
                        "gemini_confidence": None,
                        "gemini_verdict": verdict_label,
                        "gemini_evidence_used": gemini_res.get("context_used") or context_summary,
                        "gemini_raw_response": gemini_res.get("raw_response"),
                        "context_hits": context_summary,
                    }
                    results.append(gemini_block)
                else:
                    api_tracker["Gemini AI"]["success"] = False

        score_c, top = evaluate_claim_against_results(c, results, sbert)
        claim_percent = _claim_percent_from_fact_checks(level1_evidences)
        if claim_percent is None and gemini_block:
            claim_percent = gemini_block.get("percent_true")
        if claim_percent is None:
            claim_percent = 50.0
        claim_percent = max(0.0, min(100.0, float(claim_percent)))
        claim_percentages.append(claim_percent)

        # Update debug log for this claim
        debug_log["claims"][i]["score"] = round(score_c, 1)
        debug_log["claims"][i]["percent"] = round(claim_percent, 1)
        debug_log["claims"][i]["nivel"] = nivel_utilizado

        details_all.append({
            "afirmacao": c,
            "nivel": nivel_utilizado,
            "score_afirmacao": round(score_c, 1),
            "percent_afirmacao": round(claim_percent, 1),
            "evidencias": top,
            "nivel1_total": nivel1_total,
            "nivel2_total": nivel2_total,
            "gemini_investigativo": gemini_block,
            "contexto_buscas": _summarize_context_hits(context_hits) if nivel_utilizado == NIVEL_2 else [],
        })
        # Evitar rate limits agressivos
        time.sleep(0.3)

    fonte_score = sum(claim_percentages) / \
        len(claim_percentages) if claim_percentages else 0.0
    fonte_score = max(0.0, min(100.0, fonte_score))

    try:
        entity_block = verify_entities_with_serpapi(text, serpapi_key)
        if entity_block:
            entity_items = []
            for ent in entity_block.get("entities", []):
                entity_items.append({
                    "name": ent.get("entity"),
                    "status": ent.get("status"),
                    "score": ent.get("score_percent", 0.0)
                })
            debug_log["entities"] = {
                "items": entity_items,
                "media_percent": entity_block.get("media_percent", 0.0),
                "total": len(entity_items)
            }
    except Exception:
        logger.exception("Erro ao verificar entidades via SerpAPI")
        entity_block = {}

    if entity_block:
        avg_pct = entity_block.get("media_percent")
        if isinstance(avg_pct, (int, float)):
            fonte_score = min(fonte_score, float(avg_pct))

    debug_log["final_score"] = fonte_score

    # Compile API tracking
    for api_name, api_info in api_tracker.items():
        debug_log["apis_used"].append({
            "name": api_name,
            "success": api_info["success"] and api_info["results_count"] > 0,
            "results_count": api_info["results_count"],
            "enabled": api_info["enabled"]
        })

    return fonte_score, details_all, entity_block, debug_log
