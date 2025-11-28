from fact_checker import (
    load_vector_store,
    load_classifier,
    historical_consistency_for_text,
    bert_probability_true_for_text,
    fuse_scores,
    classify_text,
    blend_score_with_entities,
    faiss_claim_corroboration,
    ENTITY_BERT_WEIGHT,
)
from flask import Flask, send_from_directory, request, jsonify
import os
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports do pipeline otimizado

# External sources (mantido separado por ser grande)
try:
    from external_sources import verify_with_external_sources
    HAS_EXTERNAL = True
except ImportError:
    logger.warning(
        "external_sources.py não disponível. Fontes externas desabilitadas.")
    HAS_EXTERNAL = False

    def verify_with_external_sources(text, sbert):
        return None, [], {}

# Configurações Wikipedia
WIKI_API_URL = "https://pt.wikipedia.org/w/api.php"
WIKI_NEIGHBOR_LIMIT = 5
WIKI_ABS_MAX_RESULTS = 50
WIKI_SIMILARITY_THRESHOLD = 0.7
WIKI_REQUEST_HEADERS = {
    "User-Agent": "GoldenFake/1.0 (+https://github.com/JPeeeedrs/goldenfake)"
}

# Categorias para filtrar
WIKI_CATEGORY_EXCLUDE_PREFIXES = (
    "artigo", "artigos", "página", "páginas", "lista", "listas",
    "wikipédia", "wikipedia", "wikiprojeto", "cs1", "!",
    "todos os artigos", "predefinições", "predefinicoes",
)

WIKI_CATEGORY_EXCLUDE_SUBSTRINGS = (
    "wikificação", "wikificacao", "manutenção", "manutencao",
    "fontes", "cs1", "stub", "esboço", "esboco", "ajuda",
    "artigo destacado", "artigos destacados",
)

_WIKI_INFO_CACHE = {}

app = Flask(__name__, static_folder='static')


@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


# CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except ImportError:
    logger.warning("flask-cors não instalado. CORS pode não funcionar.")

# Carregar modelos na inicialização
try:
    INDEX, METADATA, VCFG = load_vector_store()
    CLF, LE, SBERT, CCFG = load_classifier()
    logger.info("Modelos carregados com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar modelos: {e}")
    raise

# Defaults
DEFAULTS = {
    "k": 20,
    "w_hist": 0.4,
    "w_bert": 0.3,
    "w_fontes": 0.3,
    "entity_bert_weight": ENTITY_BERT_WEIGHT,
    "max_tokens": 512,
    "overlap_tokens": 128,
    "hist_agg": "max",
    "bert_agg": "mean",
    "use_wiki": False,
    "wiki_titles": 3,
    "w_faiss": 0.5,
    "w_wiki": 0.5,
}


def _as_bool(val, default=False):
    """Converte valor para boolean."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _flatten_external_evidence(details):
    """Achata lista de evidências externas para o frontend."""
    items = []
    seen = set()

    for cl in details or []:
        for ev in (cl.get("evidencias") or []):
            url = ev.get("url") or ""
            key = url or f"{ev.get('title') or ''}|{ev.get('publisher') or ''}"

            if key in seen:
                continue
            seen.add(key)

            # Extrair porcentagem
            sim = ev.get("similaridade")
            sc = ev.get("score")

            try:
                pct = round(float(sim) * 100.0, 1) if sim is not None else None
            except Exception:
                pct = None

            if pct is None and sc is not None:
                try:
                    pct = round(float(sc) * 100.0, 1)
                except Exception:
                    pct = None

            if pct is None:
                pct = 0.0

            # Análise de rating
            rating = ev.get("rating")
            rating_text = str(rating or "").strip().lower()

            verdict = None
            if isinstance(rating, str):
                if any(tok in rating_text for tok in ("false", "falso", "fake", "enganoso")):
                    verdict = "false"
                    pct = 0.0
                elif any(tok in rating_text for tok in ("true", "verdadeiro", "correct", "correto")):
                    verdict = "true"
                    pct = 100.0

            items.append({
                "title": ev.get("title"),
                "url": url,
                "publisher": ev.get("publisher"),
                "provider": ev.get("provider"),
                "percent": pct,
                "similaridade": sim,
                "confianca_fonte": ev.get("confianca_fonte"),
                "overlap_bucket": ev.get("overlap_bucket") or "N/A",
                "is_social": ev.get("is_social", False),
                "fact_checker": bool(ev.get("fact_checker")),
                "rating": rating,
                "source_tags": ev.get("source_tags") or [],
                "gemini_percent_true": ev.get("percent_true"),
            })

    items.sort(key=lambda x: x.get("percent") or 0, reverse=True)
    return items


def _normalize_category_title(title):
    """Remove prefixo 'Categoria:' se presente."""
    if not title:
        return None
    if title.startswith("Categoria:"):
        return title.split(":", 1)[-1]
    return title


def _clean_wiki_categories(categories):
    """Filtra categorias irrelevantes da Wikipedia."""
    cleaned = []

    for cat in categories or []:
        if not cat:
            continue

        norm = cat.strip()
        if not norm:
            continue

        lower = norm.lower()
        skip = False

        # Verificar prefixos
        for prefix in WIKI_CATEGORY_EXCLUDE_PREFIXES:
            if lower.startswith(prefix):
                skip = True
                break

        # Verificar substrings
        if not skip:
            for token in WIKI_CATEGORY_EXCLUDE_SUBSTRINGS:
                if token in lower:
                    skip = True
                    break

        if skip:
            continue

        cleaned.append(norm)
        if len(cleaned) >= 6:
            break

    # Se não sobrou nada, retornar primeiras 3 originais
    if not cleaned:
        return (categories or [])[:3]

    return cleaned


def _clamp_wiki_limit(k, wiki_titles):
    """Limita número de artigos Wikipedia."""
    candidates = [wiki_titles, k, WIKI_NEIGHBOR_LIMIT]
    for cand in candidates:
        if isinstance(cand, (int, float)) and cand and cand > 0:
            val = int(cand)
            return max(1, min(val, WIKI_ABS_MAX_RESULTS))
    return WIKI_NEIGHBOR_LIMIT


def _fetch_wikipedia_metadata(page_ids):
    """Busca metadados de páginas Wikipedia."""
    page_ids = [str(pid) for pid in page_ids if pid]
    if not page_ids:
        return {}

    missing = [pid for pid in page_ids if pid not in _WIKI_INFO_CACHE]

    # Processar em lotes de 20
    for i in range(0, len(missing), 20):
        chunk = missing[i:i+20]

        params = {
            "action": "query",
            "format": "json",
            "prop": "info|categories",
            "inprop": "url|displaytitle",
            "cllimit": 20,
            "pageids": "|".join(chunk),
        }

        try:
            resp = requests.get(
                WIKI_API_URL,
                params=params,
                timeout=4,
                headers=WIKI_REQUEST_HEADERS,
            )
            resp.raise_for_status()

            pages = resp.json().get("query", {}).get("pages", {}) or {}

            for pid, data in pages.items():
                cats = []
                for cat in data.get("categories") or []:
                    norm = _normalize_category_title(cat.get("title"))
                    if norm:
                        cats.append(norm)

                cats = _clean_wiki_categories(cats)

                _WIKI_INFO_CACHE[pid] = {
                    "title": data.get("title"),
                    "displaytitle": data.get("displaytitle"),
                    "fullurl": data.get("fullurl"),
                    "categories": cats,
                }
        except Exception as e:
            logger.warning(f"Erro ao buscar metadata Wikipedia: {e}")
            for pid in chunk:
                _WIKI_INFO_CACHE.setdefault(pid, {})

    return {pid: _WIKI_INFO_CACHE.get(pid, {}).copy() for pid in page_ids}


def _build_wiki_sources(neighbors, metadata, limit=WIKI_NEIGHBOR_LIMIT,
                        similarity_threshold=WIKI_SIMILARITY_THRESHOLD):
    """Constrói seção de fontes Wikipedia."""
    section = {
        "artigos_wikipedia_similares": [],
        "categorias_expandido": {},
        "total_encontrado": 0,
        "limite_exibido": 0,
        "limite_configurado": limit,
        "limiar_similaridade": similarity_threshold,
        "relaxado_por_falta": False,
    }

    if not neighbors or not metadata:
        return section

    # Agrupar por artigo (pegar melhor similaridade)
    best_by_article = {}

    for idx, sim in neighbors:
        if idx is None or idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx] or {}
        article_id = str(meta.get("id")) if meta.get(
            "id") is not None else None

        if not article_id:
            continue

        sim_val = float(sim or 0.0)
        current = best_by_article.get(article_id)

        if current is None or sim_val > current:
            best_by_article[article_id] = sim_val

    if not best_by_article:
        return section

    # Ordenar por similaridade
    sorted_articles = sorted(best_by_article.items(),
                             key=lambda x: x[1], reverse=True)

    # Filtrar por threshold
    filtered = [(aid, score)
                for aid, score in sorted_articles if score >= similarity_threshold]

    if not filtered:
        # Relaxar threshold se nada passou
        filtered = sorted_articles[:limit]
        section["relaxado_por_falta"] = True

    section["total_encontrado"] = len(filtered)
    top = filtered[:limit]
    section["limite_exibido"] = len(top)

    if not top:
        return section

    # Buscar metadados
    meta_map = _fetch_wikipedia_metadata([aid for aid, _ in top])

    articles = []
    for aid, score in top:
        info = meta_map.get(aid) or {}
        cats = info.get("categories") or []

        articles.append({
            "id": aid,
            "titulo": info.get("displaytitle") or info.get("title") or f"Artigo {aid}",
            "similaridade": round(float(score), 3),
            "categoria": cats[0] if cats else None,
            "categorias": cats,
            "url": info.get("fullurl") or f"https://pt.wikipedia.org/?curid={aid}",
        })

    section["artigos_wikipedia_similares"] = articles

    # Agrupar por categoria
    cat_map = {}
    for art in articles:
        cats = art.get("categorias") or []
        if not cats:
            cat_map.setdefault("Sem categoria", []).append(art)
            continue

        for cat in cats:
            cat_map.setdefault(cat, []).append(art)

    # Ordenar artigos em cada categoria
    for cat, items in cat_map.items():
        items.sort(key=lambda x: x.get("similaridade", 0), reverse=True)

    section["categorias_expandido"] = cat_map

    return section


def _build_frontend_view(payload):
    """Constrói visão simplificada para o frontend."""
    confirm = payload.get("confirmacao_fontes") or {}
    fontes_individuais = confirm.get("fontes_individuais") or []
    entidades_block = confirm.get("entidades_verificadas") or {}
    wiki_section = payload.get("fontes_externas") or {}
    historico = payload.get("historico") or {}
    bert_block = payload.get("bert") or {}
    final_block = payload.get("final") or {}

    # Evidências principais
    evidencias_principais = []
    for item in fontes_individuais[:8]:
        evidencias_principais.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "publisher": item.get("publisher"),
            "provider": item.get("provider"),
            "percent": item.get("percent"),
            "tags": item.get("source_tags"),
        })

    # Entidades
    entidades_lista = []
    for ent in entidades_block.get("entidades") or []:
        entidades_lista.append({
            "entidade": ent.get("entidade"),
            "status": ent.get("status"),
            "rotulo": ent.get("rotulo"),
            "score": ent.get("score"),
            "percent": ent.get("percent"),
            "provider": ent.get("provider"),
        })

    return {
        "texto_analisado": payload.get("texto_analisado"),
        "score_final": final_block.get("score"),
        "rotulo_final": final_block.get("rotulo"),
        "componentes": {
            "historico": {
                "consistencia": historico.get("consistencia"),
                "aggregate": historico.get("aggregate"),
                "k": historico.get("k"),
            },
            "bert": {
                "rotulo": bert_block.get("rotulo"),
                "prob_true": bert_block.get("prob_true"),
                "entity_avg_percent": bert_block.get("entity_avg_percent"),
            },
            "fontes": {
                "fonte_score": confirm.get("fonte_score"),
                "total_fontes": len(fontes_individuais),
            },
        },
        "fontes": {
            "evidencias_principais": evidencias_principais,
            "entidades": {
                "resumo": {
                    "total": entidades_block.get("total"),
                    "fortes": entidades_block.get("fortes"),
                    "fracas": entidades_block.get("fracas"),
                    "ausentes": entidades_block.get("ausentes"),
                    "media_percent": entidades_block.get("media_percent"),
                },
                "lista": entidades_lista,
            },
        },
        "wikipedia": {
            "total_encontrado": wiki_section.get("total_encontrado"),
            "limite_exibido": wiki_section.get("limite_exibido"),
            "artigos": wiki_section.get("artigos_wikipedia_similares"),
        },
        "parametros": {
            "pesos": payload.get("pesos"),
            "chunking": payload.get("chunking"),
        },
    }


def analyze_text_payload(text, k, w_hist, w_bert, w_fontes,
                         entity_bert_weight, max_tokens, overlap_tokens,
                         hist_agg, bert_agg, use_wiki, wiki_titles,
                         w_faiss, w_wiki):
    """Função principal de análise."""
    text = (text or "").strip()
    if not text:
        return {"error": "texto vazio"}, 400

    wiki_limit = _clamp_wiki_limit(k, wiki_titles)

    # Consistência histórica (FAISS)
    hist_score, neighbors = historical_consistency_for_text(
        INDEX, SBERT, text,
        k=k,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        aggregate=hist_agg
    )

    # Corroboração FAISS
    faiss_corroboration = faiss_claim_corroboration(neighbors, METADATA, text)
    hist_score_raw = hist_score
    corroboration_multiplier = None
    corroboration_score = None

    if faiss_corroboration:
        corroboration_score = faiss_corroboration.get("corroboration_score")
        if corroboration_score is not None:
            try:
                raw_ratio = float(corroboration_score)
                raw_ratio = max(0.0, min(1.0, raw_ratio))
                # Aderência parcial ainda deve preservar parcela relevante do score histórico.
                multiplier = 0.5 + 0.5 * raw_ratio
                hist_score = hist_score * multiplier
                corroboration_multiplier = multiplier
                corroboration_score = raw_ratio
            except (TypeError, ValueError):
                pass

    # BERT
    bert_score_true_raw = bert_probability_true_for_text(
        CLF, LE, SBERT, CCFG, text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        aggregate=bert_agg
    )

    bert_score_true = bert_score_true_raw
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    # Fontes externas
    fonte_score, fonte_details, entity_block = verify_with_external_sources(
        text, SBERT)

    entity_avg = None
    if entity_block:
        bert_score_true, entity_avg = blend_score_with_entities(
            bert_score_true_raw, entity_block, entity_bert_weight
        )
        bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    fontes_individuais = _flatten_external_evidence(fonte_details)

    # Score final
    final_score = fuse_scores(
        hist_score, bert_score_true, fonte_score,
        w_hist, w_bert, w_fontes
    )

    final_label, _ = classify_text(final_score)

    # Montar payload
    historico_block = {
        # Proteção: só arredonda se não for None
        "consistencia": round(hist_score, 1) if hist_score is not None else 0.0,
        "consistencia_raw": round(hist_score_raw, 1) if hist_score_raw is not None else 0.0,
        "k": k,
        "aggregate": hist_agg,
        "vizinhos": neighbors,
    }

    if faiss_corroboration:
        historico_block["corroboracao"] = faiss_corroboration

    if corroboration_multiplier is not None:
        historico_block["corroboracao_multiplicador"] = round(
            float(corroboration_multiplier), 4)
    if corroboration_score is not None:
        historico_block["corroboracao_score"] = round(
            float(corroboration_score), 4)

    payload = {
        "texto_analisado": text,
        "historico": historico_block,
        "bert": {
            "rotulo": bert_label,
            # Proteção
            "prob_true": round(bert_score_true, 1) if bert_score_true is not None else 0.0,
            # Proteção
            "prob_true_raw": round(bert_score_true_raw, 1) if bert_score_true_raw is not None else 0.0,
            "entity_avg_percent": (round(entity_avg, 1) if entity_avg is not None else None),
            "entity_blend_weight": round(entity_bert_weight, 2),
            "aggregate": bert_agg,
        },
        "confirmacao_fontes": {
            # --- AQUI É O PONTO CRÍTICO ---
            # Se fonte_score for None (erro/internet caiu), retornamos None para o frontend saber
            "fonte_score": round(fonte_score, 1) if fonte_score is not None else None,
            "detalhes": fonte_details,
            "fontes_individuais": fontes_individuais,
            "entidades_verificadas": entity_block,
        },
        "final": {
            "rotulo": final_label,
            "score": round(final_score, 1),
        },
        "pesos": {
            "historico": w_hist,
            "bert": w_bert,
            "fontes": w_fontes,
            "entity_bert_weight": entity_bert_weight,
        },
        "chunking": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
    }

    # Wikipedia sources
    wiki_section = _build_wiki_sources(neighbors, METADATA, limit=wiki_limit)
    payload["fontes_externas"] = wiki_section
    payload["wikipedia"] = {
        "matches": wiki_section.get("artigos_wikipedia_similares", []),
        "titles_limit": wiki_limit,
    }

    # Frontend view
    payload["frontend_view"] = _build_frontend_view(payload)

    return payload, 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Endpoint principal de análise."""
    data = request.get_json(silent=True) or request.form.to_dict() or {}

    text = data.get("text") or data.get("texto") or ""

    def _get_num(key, typ, default):
        try:
            return typ(data.get(key, default))
        except Exception:
            return default

    k = _get_num("k", int, DEFAULTS["k"])
    w_hist = _get_num("w_hist", float, DEFAULTS["w_hist"])
    w_bert = _get_num("w_bert", float, DEFAULTS["w_bert"])
    w_fontes = _get_num("w_fontes", float, DEFAULTS["w_fontes"])
    entity_weight = _get_num("entity_bert_weight", float,
                             DEFAULTS["entity_bert_weight"])
    max_tokens = _get_num("max_tokens", int, DEFAULTS["max_tokens"])
    overlap_tokens = _get_num("overlap_tokens", int,
                              DEFAULTS["overlap_tokens"])

    hist_agg = str(data.get("hist_agg", DEFAULTS["hist_agg"]))
    if hist_agg not in ("max", "mean"):
        hist_agg = DEFAULTS["hist_agg"]

    bert_agg = str(data.get("bert_agg", DEFAULTS["bert_agg"]))
    if bert_agg not in ("max", "mean"):
        bert_agg = DEFAULTS["bert_agg"]

    use_wiki = _as_bool(data.get("use_wiki", DEFAULTS["use_wiki"]))
    wiki_titles = _get_num("wiki_titles", int, DEFAULTS["wiki_titles"])
    w_faiss = _get_num("w_faiss", float, DEFAULTS["w_faiss"])
    w_wiki = _get_num("w_wiki", float, DEFAULTS["w_wiki"])

    result, status = analyze_text_payload(
        text=text,
        k=k,
        w_hist=w_hist,
        w_bert=w_bert,
        w_fontes=w_fontes,
        entity_bert_weight=entity_weight,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        hist_agg=hist_agg,
        bert_agg=bert_agg,
        use_wiki=use_wiki,
        wiki_titles=wiki_titles,
        w_faiss=w_faiss,
        w_wiki=w_wiki,
    )

    return jsonify(result), status


@app.route("/analyze_friendly", methods=["POST"])
def analyze_friendly():
    """Endpoint com resposta simplificada."""
    data = request.get_json(silent=True) or request.form.to_dict() or {}

    text = data.get("text") or data.get("texto") or ""

    if not text.strip():
        return jsonify({"error": "Por favor, insira um texto para análise."}), 400

    # Usar defaults
    result, status = analyze_text_payload(
        text=text,
        k=int(data.get("k", DEFAULTS["k"])),
        w_hist=float(data.get("w_hist", DEFAULTS["w_hist"])),
        w_bert=float(data.get("w_bert", DEFAULTS["w_bert"])),
        w_fontes=float(data.get("w_fontes", DEFAULTS["w_fontes"])),
        entity_bert_weight=float(
            data.get("entity_bert_weight", DEFAULTS["entity_bert_weight"])),
        max_tokens=int(data.get("max_tokens", DEFAULTS["max_tokens"])),
        overlap_tokens=int(
            data.get("overlap_tokens", DEFAULTS["overlap_tokens"])),
        hist_agg=data.get("hist_agg", DEFAULTS["hist_agg"]),
        bert_agg=data.get("bert_agg", DEFAULTS["bert_agg"]),
        use_wiki=_as_bool(data.get("use_wiki", DEFAULTS["use_wiki"])),
        wiki_titles=int(data.get("wiki_titles", DEFAULTS["wiki_titles"])),
        w_faiss=float(data.get("w_faiss", DEFAULTS["w_faiss"])),
        w_wiki=float(data.get("w_wiki", DEFAULTS["w_wiki"])),
    )

    if status != 200:
        return jsonify(result), status

    friendly_output = {
        "Texto Analisado": result["texto_analisado"],
        "Consistência Histórica": f"{result['historico']['consistencia']}%",
        "Análise BERT": {
            "Rótulo": result["bert"]["rotulo"],
            "Probabilidade de Verdadeiro": f"{result['bert']['prob_true']}%",
        },
        "Confirmação por Fontes Externas": f"{result['confirmacao_fontes']['fonte_score']}%",
        "Classificação Final": {
            "Rótulo": result["final"]["rotulo"],
            "Score": f"{result['final']['score']}%",
        },
    }

    return jsonify(friendly_output), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
