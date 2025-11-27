import os
import json
import numpy as np
import faiss
import joblib
import re
import requests
from functools import lru_cache
from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports opcionais
try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("pt_core_news_sm")
    except Exception:
        _SPACY_NLP = None
except Exception:
    spacy = None
    _SPACY_NLP = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VSTORE_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VSTORE_DIR, "faiss_index.bin")
META_PATH = os.path.join(VSTORE_DIR, "faiss_metadata.json")
VEC_CONFIG_PATH = os.path.join(VSTORE_DIR, "faiss_config.json")

MODEL_DIR = os.path.join(BASE_DIR, "models")
CLS_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
CLS_CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
STYLE_SCALER_PATH = os.path.join(MODEL_DIR, "style_scaler.joblib")

# Fallback para diretório treino/models
if not all(os.path.isfile(p) for p in [CLS_PATH, LE_PATH, CLS_CONFIG_PATH]):
    _ALT_DIR = os.path.join(BASE_DIR, "treino", "models")
    if os.path.isdir(_ALT_DIR):
        _alt_cls = os.path.join(_ALT_DIR, "classifier.joblib")
        _alt_le = os.path.join(_ALT_DIR, "label_encoder.joblib")
        _alt_cfg = os.path.join(_ALT_DIR, "config.json")
        if all(os.path.isfile(p) for p in [_alt_cls, _alt_le, _alt_cfg]):
            MODEL_DIR = _ALT_DIR
            CLS_PATH = _alt_cls
            LE_PATH = _alt_le
            CLS_CONFIG_PATH = _alt_cfg
            STYLE_SCALER_PATH = os.path.join(MODEL_DIR, "style_scaler.joblib")

# Features de estilo
SENSATIONAL_LEXICON = [
    "urgente", "chocante", "escândalo", "bomba", "imperdível",
    "revelado", "exclusivo", "alerta", "incrível", "verdadeiro?",
    "mentira", "fraude", "golpe", "boato", "polêmico", "assustador"
]
PUNCT_SET = set("!?")

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 128

# Cache do style scaler
_STYLE_SCALER = None

# Peso para blend com entidades
ENTITY_BERT_WEIGHT = 0.35


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0


def extract_style_features(text: str) -> np.ndarray:
    """Extrai 8 features de estilo."""
    t = text or ""
    n_chars = len(t)
    letters = [c for c in t if c.isalpha()]
    n_letters = len(letters)
    n_upper = sum(1 for c in letters if c.isupper())
    
    words = re.findall(r"\b\w+\b", t, flags=re.UNICODE)
    n_words = len(words)
    words_alpha = [w for w in words if any(ch.isalpha() for ch in w)]
    n_allcaps_words = sum(1 for w in words_alpha if len(w) >= 3 and w.isupper())
    
    avg_word_len = _safe_div(sum(len(w) for w in words_alpha), len(words_alpha))
    ttr = _safe_div(len(set(w.lower() for w in words_alpha)), len(words_alpha))
    
    punct_count = sum(1 for c in t if c in PUNCT_SET)
    punct_ratio = _safe_div(punct_count, max(n_chars, 1))
    upper_ratio = _safe_div(n_upper, max(n_letters, 1))
    allcaps_ratio = _safe_div(n_allcaps_words, max(n_words, 1))
    
    exclam = min(t.count("!"), 10) / 10.0
    quest = min(t.count("?"), 10) / 10.0
    
    low = t.lower()
    lex_count = sum(1 for kw in SENSATIONAL_LEXICON if kw in low)
    lex_density = _safe_div(lex_count, max(n_words, 1))
    
    return np.array([
        upper_ratio, allcaps_ratio, punct_ratio,
        exclam, quest, min(avg_word_len, 20) / 20.0,
        ttr, min(lex_density, 1.0)
    ], dtype=np.float32)


def _get_style_scaler():
    """Carrega o scaler de estilo (cache)."""
    global _STYLE_SCALER
    if _STYLE_SCALER is None and os.path.isfile(STYLE_SCALER_PATH):
        try:
            _STYLE_SCALER = joblib.load(STYLE_SCALER_PATH)
        except Exception:
            _STYLE_SCALER = None
    return _STYLE_SCALER


def chunk_text_optimized(model: SentenceTransformer, text: str,
                        max_tokens: int = _DEFAULT_MAX_TOKENS,
                        overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS) -> List[str]:
    """Divide texto em chunks otimizado."""
    try:
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        stride = max(1, max_tokens - overlap_tokens)
        
        for start in range(0, len(tokens), stride):
            window = tokens[start:start + max_tokens]
            if not window:
                break
            chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
    
    except Exception:
        return [text]


def load_vector_store():
    """Carrega índice FAISS e metadados."""
    if not all(os.path.isfile(p) for p in [INDEX_PATH, META_PATH, VEC_CONFIG_PATH]):
        raise FileNotFoundError("Vector store não encontrado. Execute build_faiss_index.py")
    
    index = faiss.read_index(INDEX_PATH)
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    with open(VEC_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    logger.info(f"FAISS carregado: {index.ntotal} vetores")
    return index, metadata, cfg


def load_classifier():
    """Carrega classificador XGBoost calibrado."""
    if not all(os.path.isfile(p) for p in [CLS_PATH, LE_PATH, CLS_CONFIG_PATH]):
        raise FileNotFoundError("Modelo não encontrado. Execute train_classifier.py")
    
    clf = joblib.load(CLS_PATH)
    le = joblib.load(LE_PATH)
    
    with open(CLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    embed_model = cfg.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sbert = SentenceTransformer(embed_model)
    sbert.max_seq_length = int(cfg.get("max_seq_length", 512))
    
    logger.info(f"Classificador carregado: {type(clf).__name__}")
    return clf, le, sbert, cfg


def embed_query(model: SentenceTransformer, text: str, normalize: bool = True) -> np.ndarray:
    """Gera embedding único normalizado."""
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )
    return emb.astype(np.float32)


def embed_query_with_style(model: SentenceTransformer, text: str, cfg: Dict) -> np.ndarray:
    """Gera embedding com features de estilo opcionais."""
    base = embed_query(model, text, normalize=True)
    
    if not cfg.get("style_features"):
        return base
    
    scaler = _get_style_scaler()
    if scaler is None:
        return base
    
    style = extract_style_features(text).reshape(1, -1)
    style_scaled = scaler.transform(style)
    
    # Aplicar peso do estilo
    style_weight = float(cfg.get("style_weight", 0.15))
    style_weighted = style_scaled * style_weight
    
    return np.hstack([base, style_weighted.astype(np.float32)])


def historical_consistency_for_text(index, model: SentenceTransformer, text: str,
                                   k: int = 8,
                                   max_tokens: int = _DEFAULT_MAX_TOKENS,
                                   overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                   aggregate: str = "max") -> Tuple[float, List]:
    """Calcula consistência histórica com FAISS usando chunking e agregação."""
    chunks = chunk_text_optimized(model, text, max_tokens, overlap_tokens)
    
    if not chunks:
        return 0.0, []
    
    scores = []
    all_neighbors = []
    
    for chunk in chunks:
        q_emb = embed_query(model, chunk, normalize=True)
        
        if index.ntotal == 0:
            return 0.0, []
        
        D, I = index.search(q_emb, k)
        sims = np.clip(D[0], 0.0, 1.0)
        score = float(np.mean(sims)) * 100.0
        
        scores.append(score)
        all_neighbors.append(list(zip(I[0].tolist(), sims.tolist())))
    
    if not scores:
        return 0.0, []
    
    if aggregate == "mean":
        final_score = float(np.mean(scores))
        # Retornar vizinhos do melhor chunk
        best_idx = int(np.argmax(scores))
        return final_score, all_neighbors[best_idx]
    else:  # max
        best_idx = int(np.argmax(scores))
        return scores[best_idx], all_neighbors[best_idx]


def bert_probability_true_for_text(clf, le, model: SentenceTransformer, cfg: Dict,
                                  text: str,
                                  max_tokens: int = _DEFAULT_MAX_TOKENS,
                                  overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                  aggregate: str = "mean") -> float:
    """Calcula probabilidade 'true' usando BERT com chunking."""
    chunks = chunk_text_optimized(model, text, max_tokens, overlap_tokens)
    
    if not chunks:
        return 0.0
    
    # Processar todos os chunks em batch
    embeddings = [embed_query_with_style(model, chunk, cfg) for chunk in chunks]
    X = np.vstack(embeddings)
    
    # Predição em batch
    proba = clf.predict_proba(X)
    
    # Identificar índice da classe 'true'
    try:
        idx_true = list(le.classes_).index("true")
    except ValueError:
        idx_true = int(np.argmax(proba[0]))
    
    probs_true = proba[:, idx_true] * 100.0
    
    if aggregate == "max":
        return float(np.max(probs_true))
    else:  # mean
        return float(np.mean(probs_true))


def blend_score_with_entities(base_score: float,
                             entity_block: Dict[str, Any] | None,
                             weight: float = ENTITY_BERT_WEIGHT) -> Tuple[float, float | None]:
    """Combina score base com média de entidades verificadas."""
    if not entity_block:
        return base_score, None
    
    avg_pct = entity_block.get("media_percent")
    total = entity_block.get("total", 0)
    
    if avg_pct is None or total <= 0:
        return base_score, None
    
    avg_pct = max(0.0, min(100.0, float(avg_pct)))
    weight = max(0.0, min(1.0, float(weight)))
    
    blended = (1.0 - weight) * base_score + weight * avg_pct
    return blended, avg_pct


def fuse_scores(hist_score: float, bert_score: float, fonte_score: float | None = None,
               w_hist: float = 0.4, w_bert: float = 0.3, w_fontes: float = 0.3) -> float:
    """Fusão ponderada de scores."""
    weights = np.array([w_hist, w_bert, w_fontes], dtype=np.float32)
    scores = np.array([
        hist_score if hist_score is not None else 0.0,
        bert_score if bert_score is not None else 0.0,
        fonte_score if fonte_score is not None else 0.0
    ], dtype=np.float32)
    
    # Normalizar pesos
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    
    return float(np.dot(weights, scores))


def classify_text(final_score: float) -> Tuple[str, float]:
    """Classifica com base no score final."""
    if final_score >= 50.0:
        return "VERDADEIRO", final_score
    else:
        return "FALSO", 100.0 - final_score


# Wikipedia dynamic retrieval (opcional - pode ser movido para módulo separado)
def _wiki_opensearch_titles(query: str, limit: int = 3) -> List[Tuple[str, str]]:
    """Busca títulos na Wikipedia PT."""
    try:
        url = "https://pt.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json"
        }
        r = requests.get(url, params=params, timeout=3)
        r.raise_for_status()
        data = r.json()
        
        titles = data[1] if len(data) > 1 else []
        links = data[3] if len(data) > 3 else []
        
        return [(titles[i], links[i]) for i in range(min(len(titles), len(links)))]
    except Exception:
        return []


def wikipedia_dynamic_neighbors(model: SentenceTransformer, query_text: str,
                               titles_limit: int = 3, k: int = 8,
                               max_tokens: int = _DEFAULT_MAX_TOKENS) -> Tuple[float, List[Dict]]:
    """Busca dinâmica na Wikipedia e retorna score de similaridade."""
    titles = _wiki_opensearch_titles(query_text, limit=titles_limit)
    if not titles:
        return 0.0, []
    
    # Simplified: apenas retornar score base e títulos
    # A implementação completa requer fetch do conteúdo das páginas
    q_emb = embed_query(model, query_text, normalize=True)
    
    details = []
    for title, url in titles:
        details.append({
            "title": title,
            "url": url,
            "similaridade": 0.5  # Placeholder
        })
    
    score = 50.0  # Score base para Wikipedia
    return score, details


def combined_historical_consistency_for_text(index, model: SentenceTransformer,
                                            text: str, k: int = 8,
                                            max_tokens: int = _DEFAULT_MAX_TOKENS,
                                            overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                            aggregate: str = "max",
                                            wiki_titles: int = 3,
                                            w_faiss: float = 0.7,
                                            w_wiki: float = 0.3) -> Tuple[float, Dict]:
    """Combina FAISS e Wikipedia."""
    faiss_score, neighbors = historical_consistency_for_text(
        index, model, text, k, max_tokens, overlap_tokens, aggregate
    )
    
    wiki_score, wiki_details = wikipedia_dynamic_neighbors(
        model, text, wiki_titles, k, max_tokens
    )
    
    # Normalizar pesos
    if wiki_score <= 0:
        combined = faiss_score
        wf, ww = 1.0, 0.0
    elif faiss_score <= 0:
        combined = wiki_score
        wf, ww = 0.0, 1.0
    else:
        total = w_faiss + w_wiki
        wf = w_faiss / total
        ww = w_wiki / total
        combined = faiss_score * wf + wiki_score * ww
    
    return combined, {
        "faiss": {"score": round(faiss_score, 1), "vizinhos": neighbors},
        "wikipedia": {"score": round(wiki_score, 1), "matches": wiki_details},
        "pesos": {"faiss": round(wf, 2), "wikipedia": round(ww, 2)}
    }


# Corroboração FAISS (simplificada)
@lru_cache(maxsize=256)
def fetch_wikipedia_article(page_id: str) -> Dict | None:
    """Busca artigo da Wikipedia (com cache)."""
    try:
        url = "https://pt.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "pageids": str(page_id)
        }
        r = requests.get(url, params=params, timeout=4)
        r.raise_for_status()
        
        data = r.json().get("query", {}).get("pages", {})
        page = data.get(str(page_id))
        
        if not page or page.get("missing"):
            return None
        
        return {
            "id": page_id,
            "title": page.get("title"),
            "text": page.get("extract", ""),
            "url": f"https://pt.wikipedia.org/?curid={page_id}"
        }
    except Exception:
        return None


def faiss_claim_corroboration(neighbors: List[Tuple[int, float]],
                              metadata: List[Dict],
                              claim_text: str,
                              top_articles: int = 3) -> Dict | None:
    """Verifica corroboração da claim nos artigos mais similares do FAISS."""
    if not neighbors or not metadata:
        return None
    
    # Extrair keywords simples da claim
    claim_tokens = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
    
    results = []
    max_corroboration = 0.0
    
    for idx, similarity in neighbors[:top_articles]:
        if idx < 0 or idx >= len(metadata):
            continue
        
        entry = metadata[idx]
        page_id = entry.get("id")
        
        if not page_id:
            continue
        
        article = fetch_wikipedia_article(str(page_id))
        
        if not article:
            results.append({
                "page_id": str(page_id),
                "similaridade": round(float(similarity), 3),
                "status": "unavailable"
            })
            continue
        
        # Verificar overlap de tokens
        article_text = article.get("text", "").lower()
        article_tokens = set(re.findall(r'\b\w{4,}\b', article_text))
        
        matched = claim_tokens & article_tokens
        ratio = len(matched) / len(claim_tokens) if claim_tokens else 0.0
        
        status = "corroborated" if ratio >= 0.6 else \
                 "partially_corroborated" if ratio >= 0.3 else "not_corroborated"
        
        max_corroboration = max(max_corroboration, ratio)
        
        results.append({
            "page_id": str(page_id),
            "title": article.get("title"),
            "url": article.get("url"),
            "similaridade": round(float(similarity), 3),
            "status": status,
            "corroboration_ratio": round(ratio, 3)
        })
    
    overall_status = "corroborated" if max_corroboration >= 0.6 else \
                     "partially_corroborated" if max_corroboration >= 0.3 else "not_corroborated"
    
    return {
        "status": overall_status,
        "corroboration_score": round(max_corroboration, 3),
        "articles": results
    }


# CLI simplificada
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fact Checker otimizado")
    parser.add_argument("--text", type=str, help="Texto a analisar")
    parser.add_argument("--k", type=int, default=8, help="Vizinhos FAISS")
    parser.add_argument("--w_hist", type=float, default=0.4)
    parser.add_argument("--w_bert", type=float, default=0.3)
    parser.add_argument("--w_fontes", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    text = args.text or input("Texto para análise:\n")
    if not text.strip():
        print("Texto vazio")
        return
    
    # Carregar modelos
    index, metadata, _ = load_vector_store()
    clf, le, sbert, cfg = load_classifier()
    
    # Análise
    hist_score, neighbors = historical_consistency_for_text(index, sbert, text, k=args.k)
    bert_score = bert_probability_true_for_text(clf, le, sbert, cfg, text)
    
    # Fonte score (placeholder - requer external_sources.py)
    fonte_score = 50.0
    
    final_score = fuse_scores(hist_score, bert_score, fonte_score,
                            args.w_hist, args.w_bert, args.w_fontes)
    label, _ = classify_text(final_score)
    
    result = {
        "texto": text,
        "historico": {"score": round(hist_score, 1), "k": args.k},
        "bert": {"score": round(bert_score, 1)},
        "fontes": {"score": round(fonte_score, 1)},
        "final": {"label": label, "score": round(final_score, 1)}
    }
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\nHistórico: {result['historico']['score']:.1f}%")
        print(f"BERT: {result['bert']['score']:.1f}%")
        print(f"Fontes: {result['fontes']['score']:.1f}%")
        print(f"\nFinal: {label} ({final_score:.1f}%)")


if __name__ == "__main__":
    main()