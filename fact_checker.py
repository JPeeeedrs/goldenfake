import os
import json
import argparse
import numpy as np
import faiss
import joblib
import re
import unicodedata
import requests
from functools import lru_cache
from typing import Any, Dict, List, Tuple

try:  # spaCy é opcional em runtime
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

_SPACY_NLP = None
from sentence_transformers import SentenceTransformer
from external_sources import verify_with_external_sources

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

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
# --- fallback para treino/models quando a pasta raiz models não existir ou faltar arquivos ---
if not (os.path.isdir(MODEL_DIR) and all(os.path.isfile(p) for p in [CLS_PATH, LE_PATH, CLS_CONFIG_PATH])):
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

# --- Estilo (opcional) ---
SENSATIONAL_LEXICON = [
    "urgente", "chocante", "escândalo", "bomba", "imperdível", "você não vai acreditar",
    "revelado", "exclusivo", "alerta", "atenção", "incrível", "chocado", "verdadeiro?",
    "mentira", "fraude", "golpe", "boato", "polêmico", "assustador", "impactante",
]
PUNCT_SET = set(list("!?"))
_STYLE_SCALER = None  # cache

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 128

# --- Online adaptive classifier (incremental fine-tuning) ---
try:
    # sklearn is in requirements; import lazily to avoid heavy deps on tools using this file as a lib
    from sklearn.linear_model import SGDClassifier  # type: ignore
except Exception:  # pragma: no cover
    SGDClassifier = None  # type: ignore

ONLINE_ADAPTOR_PATH = os.path.join(MODEL_DIR, "online_adaptor.joblib")

def _parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


ENTITY_BERT_WEIGHT = max(0.0, min(1.0, _parse_float_env("ENTITY_BERT_WEIGHT", 0.35)))


class OnlineAdaptiveClassifier:
    """Incremental adaptor trained on user feedback, keeping base model intact.
    Uses SGDClassifier (logistic regression) with partial_fit.
    """

    def __init__(self, le, sbert: SentenceTransformer, cfg: dict):
        self.le = le
        self.sbert = sbert
        self.cfg = cfg or {}
        self.inc_clf = None
        self.n_updates = 0
        # Persist a snapshot of classes for partial_fit's classes= param on first call
        self._classes_idx = np.arange(
            len(list(le.classes_))) if hasattr(le, "classes_") else None
        # cache style scaler usage via embed_query_with_style

    def is_trained(self) -> bool:
        return self.inc_clf is not None and getattr(self.inc_clf, "classes_", None) is not None

    def _ensure_init(self):
        if self.inc_clf is None:
            if SGDClassifier is None:
                raise RuntimeError("SGDClassifier indisponível.")
            # log_loss gives probabilities
            self.inc_clf = SGDClassifier(
                loss="log_loss", alpha=1e-4, penalty="l2", max_iter=1, learning_rate="optimal", random_state=42)

    def _embed_texts_with_style(self, texts: list[str]) -> np.ndarray:
        # Build features compatible with the base classifier (SBERT + optional style)
        Xs = []
        for t in texts:
            Xs.append(embed_query_with_style(self.sbert, t, self.cfg))
        return np.vstack(Xs)

    def update(self, texts: list[str], labels_str: list[str], sample_weight: np.ndarray | None = None):
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return False
        if not labels_str or len(labels_str) != len(texts):
            raise ValueError("labels_str deve ter o mesmo tamanho de texts")
        # Mapear rótulos string para índices via label encoder
        y_idx = self.le.transform([str(y).lower() for y in labels_str])
        X = self._embed_texts_with_style(texts)
        self._ensure_init()
        if not self.is_trained():
            try:
                self.inc_clf.partial_fit(X, y_idx, classes=self._classes_idx)
            except Exception:
                self.inc_clf.partial_fit(X, y_idx, classes=self._classes_idx)
        else:
            try:
                if sample_weight is not None and getattr(self.inc_clf, "partial_fit", None):
                    self.inc_clf.partial_fit(X, y_idx, sample_weight=sample_weight)
                else:
                    self.inc_clf.partial_fit(X, y_idx)
            except TypeError:
                self.inc_clf.partial_fit(X, y_idx)
        self.n_updates += len(texts)
        self.save()
        return True

    def predict_proba_for_chunks(self, chunks: list[str]) -> float | None:
        if not self.is_trained() or not chunks:
            return None
        X = self._embed_texts_with_style(chunks)
        try:
            P = self.inc_clf.predict_proba(X)
        except Exception:
            return None
        try:
            idx_true = list(self.le.classes_).index("true")
        except ValueError:
            idx_true = int(np.argmax(P[0]))
        probs = P[:, idx_true] * 100.0
        return float(np.mean(probs))

    def prob_true_for_text(self, text: str,
                           max_tokens: int = _DEFAULT_MAX_TOKENS,
                           overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                           aggregate: str = "mean") -> float | None:
        if not self.is_trained() or not (text and text.strip()):
            return None
        ids = _token_ids(self.sbert, text)
        if len(ids) <= max_tokens:
            return self.predict_proba_for_chunks([text])
        chunks = split_text_into_token_chunks(
            self.sbert, text, max_tokens, overlap_tokens)
        if not chunks:
            return self.predict_proba_for_chunks([text])
        # compute per-chunk probs
        X = self._embed_texts_with_style(chunks)
        try:
            P = self.inc_clf.predict_proba(X)
        except Exception:
            return None
        try:
            idx_true = list(self.le.classes_).index("true")
        except ValueError:
            idx_true = int(np.argmax(P[0]))
        probs = P[:, idx_true] * 100.0
        if aggregate == "max":
            return float(np.max(probs))
        return float(np.mean(probs))

    def alpha(self) -> float:
        """Peso do adaptador na fusão com o modelo base. Cresce com n_updates, limitado a 0.8."""
        n = float(self.n_updates)
        # 0 -> 0.0, 10 -> ~0.39, 50 -> ~0.78, limite 0.8
        a = 1.0 - np.exp(-n / 25.0)
        return float(min(0.8, max(0.0, a)))

    def save(self):
        try:
            joblib.dump({
                "inc_clf": self.inc_clf,
                "n_updates": self.n_updates,
                "classes": list(self.le.classes_),
            }, ONLINE_ADAPTOR_PATH)
        except Exception:
            pass

    def load_from_disk(self):
        if not os.path.isfile(ONLINE_ADAPTOR_PATH):
            return False
        try:
            data = joblib.load(ONLINE_ADAPTOR_PATH)
            self.inc_clf = data.get("inc_clf")
            self.n_updates = int(data.get("n_updates") or 0)
            return True
        except Exception:
            return False


def load_online_adaptor(le, sbert: SentenceTransformer, cfg: dict) -> OnlineAdaptiveClassifier:
    """Cria/Carrega adaptador incremental persistente em models/online_adaptor.joblib."""
    onl = OnlineAdaptiveClassifier(le, sbert, cfg)
    onl.load_from_disk()
    return onl


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def blend_score_with_entities(base_score: float,
                              entity_block: Dict[str, Any] | None,
                              weight: float | None = None) -> Tuple[float, float | None]:
    """Funde o score base com a média de entidades (0-100) usando peso beta."""
    try:
        base = float(base_score)
    except Exception:
        base = 0.0
    if not entity_block:
        return base, None
    avg = entity_block.get("media_percent")
    total = entity_block.get("total") or 0
    if avg is None or total <= 0:
        return base, None
    try:
        avg_float = float(avg)
    except Exception:
        return base, None
    avg_float = max(0.0, min(100.0, avg_float))
    w = ENTITY_BERT_WEIGHT if weight is None else float(weight)
    w = max(0.0, min(1.0, w))
    if w == 0.0:
        return base, avg_float
    blended = (1.0 - w) * base + w * avg_float
    return blended, avg_float


def extract_style_features(text: str) -> np.ndarray:
    t = text or ""
    n_chars = len(t)
    letters = [c for c in t if c.isalpha()]
    n_letters = len(letters)
    n_upper = sum(1 for c in letters if c.isupper())
    words = re.findall(r"\b\w+\b", t, flags=re.UNICODE)
    n_words = len(words)
    words_alpha = [w for w in words if any(ch.isalpha() for ch in w)]
    n_allcaps_words = sum(1 for w in words_alpha if len(w)
                          >= 3 and w.isupper())
    avg_word_len = _safe_div(sum(len(w)
                             for w in words_alpha), len(words_alpha))
    ttr = _safe_div(len(set(w.lower() for w in words_alpha)), len(words_alpha))
    punct_count = sum(1 for c in t if c in PUNCT_SET)
    punct_ratio = _safe_div(punct_count, max(n_chars, 1))
    upper_ratio = _safe_div(n_upper, max(n_letters, 1))
    allcaps_ratio = _safe_div(n_allcaps_words, max(n_words, 1))
    exclam = t.count("!")
    quest = t.count("?")
    lex_count = 0
    low = t.lower()
    for kw in SENSATIONAL_LEXICON:
        if kw in low:
            lex_count += 1
    lex_density = _safe_div(lex_count, max(n_words, 1))
    feats = np.array([
        upper_ratio,
        allcaps_ratio,
        punct_ratio,
        min(exclam, 10) / 10,
        min(quest, 10) / 10,
        min(avg_word_len, 20) / 20,
        ttr,
        min(lex_density, 1.0),
    ], dtype=float)
    return feats


def _get_style_scaler():
    global _STYLE_SCALER
    if _STYLE_SCALER is None and os.path.isfile(STYLE_SCALER_PATH):
        try:
            _STYLE_SCALER = joblib.load(STYLE_SCALER_PATH)
        except Exception:
            _STYLE_SCALER = None
    return _STYLE_SCALER


# --- Tokenization helpers (chunking) ---

def _token_ids(model: SentenceTransformer, text: str) -> list[int]:
    try:
        tok = model.tokenizer
        # SentenceTransformer mantém max_seq_length separado; aumentamos manualmente o limite do tokenizer
        max_len = getattr(tok, "model_max_length", None)
        if isinstance(max_len, int) and max_len < 4096:
            try:
                tok.model_max_length = 16384
                if hasattr(tok, "init_kwargs"):
                    tok.init_kwargs["model_max_length"] = tok.model_max_length
            except Exception:
                pass
        encoded = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(ids, list):
            return ids
        # último recurso com encode tradicional (com truncation desativado)
        return tok.encode(text, add_special_tokens=False, truncation=False)
    except Exception:
        # Fallback: approximate by words as "tokens"
        return text.split()


def split_text_into_token_chunks(model: SentenceTransformer, text: str,
                                 max_tokens: int = _DEFAULT_MAX_TOKENS,
                                 overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS) -> list[str]:
    ids = _token_ids(model, text)
    if not ids:
        return []
    stride = max(1, max_tokens - max(0, overlap_tokens))
    chunks: list[str] = []
    # Use tokenizer decode if available; otherwise join words
    use_decode = hasattr(model, 'tokenizer') and hasattr(
        model.tokenizer, 'decode')
    for start in range(0, len(ids), stride):
        window = ids[start:start + max_tokens]
        if not window:
            break
        if use_decode:
            chunk_text = model.tokenizer.decode(
                window, skip_special_tokens=True)
        else:
            if isinstance(window[0], str):
                chunk_text = " ".join(window)
            else:
                # Last-resort: slice original text by char length proportionally
                # Not ideal, but ensures some chunk content
                approx = int(
                    len(text) * min(1.0, (start + len(window)) / max(1, len(ids))))
                prev = int(len(text) * (start / max(1, len(ids))))
                chunk_text = text[prev:approx]
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def load_vector_store():
    if not (os.path.isfile(INDEX_PATH) and os.path.isfile(META_PATH) and os.path.isfile(VEC_CONFIG_PATH)):
        raise FileNotFoundError(
            "Vector store não encontrado. Execute build_faiss_index.py primeiro.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(VEC_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return index, metadata, cfg


def load_classifier():
    if not (os.path.isfile(CLS_PATH) and os.path.isfile(LE_PATH) and os.path.isfile(CLS_CONFIG_PATH)):
        raise FileNotFoundError(
            "Modelo não encontrado. Execute train_classifier.py primeiro.")
    clf = joblib.load(CLS_PATH)
    le = joblib.load(LE_PATH)
    with open(CLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sbert = SentenceTransformer(
        cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
    # Usar 512 tokens por chunk por padrão
    try:
        sbert.max_seq_length = int(cfg.get("max_seq_length", 512))
    except Exception:
        sbert.max_seq_length = 512
    return clf, le, sbert, cfg


def embed_query(model, text: str, normalize=True):
    emb = model.encode([text], convert_to_numpy=True,
                       normalize_embeddings=normalize)
    return emb.astype("float32")


def embed_query_with_style(model, text: str, cfg: dict) -> np.ndarray:
    """Retorna embedding para classificador: SBERT + (opcional) estilo normalizado."""
    base = embed_query(model, text)
    if not cfg or not cfg.get("style_features"):
        return base
    scaler = _get_style_scaler()
    if scaler is None:
        return base
    style = extract_style_features(text).reshape(1, -1)
    style_s = scaler.transform(style)
    # aplicar o mesmo fator usado no treino, se existir
    style_scale = float(cfg.get("style_scale", 1.0))
    style_s = style_s * style_scale
    combo = np.hstack([base, style_s.astype(base.dtype)])
    return combo


def historical_consistency(index, q_emb, k=8):
    if index.ntotal == 0:
        return 0.0, []
    D, I = index.search(q_emb, k)
    sims = D[0]
    sims = np.clip(sims, 0.0, 1.0)
    score = float(np.mean(sims)) * 100.0
    return score, list(zip(I[0].tolist(), sims.tolist()))


def bert_probability_true(clf, le, q_emb):
    probs = clf.predict_proba(q_emb)[0]
    try:
        idx_true = list(le.classes_).index("true")
    except ValueError:
        idx_true = int(np.argmax(probs))
    return float(probs[idx_true]) * 100.0


# --- Chunk-aware wrappers ---

def historical_consistency_for_text(index, model: SentenceTransformer, text: str,
                                    k: int = 8,
                                    max_tokens: int = _DEFAULT_MAX_TOKENS,
                                    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                    aggregate: str = "max"):
    """Calcula consistência histórica por chunks e agrega.
    aggregate: 'max' ou 'mean'
    Retorna (score_aggregado, detalhes_do_melhor_chunk)
    """
    ids = _token_ids(model, text)
    if len(ids) <= max_tokens:
        q_emb = embed_query(model, text)
        return historical_consistency(index, q_emb, k=k)

    chunks = split_text_into_token_chunks(
        model, text, max_tokens, overlap_tokens)
    if not chunks:
        q_emb = embed_query(model, text)
        return historical_consistency(index, q_emb, k=k)

    scores = []
    details = []
    for ch in chunks:
        q_emb = embed_query(model, ch)
        sc, det = historical_consistency(index, q_emb, k=k)
        scores.append(sc)
        details.append((sc, ch, det))

    if not scores:
        return 0.0, []

    if aggregate == "mean":
        agg_score = float(np.mean(scores))
        # Retornar detalhes do melhor chunk para referência
        best = max(details, key=lambda x: x[0])
        return agg_score, best[2]
    else:  # 'max'
        best = max(details, key=lambda x: x[0])
        return best[0], best[2]


def bert_probability_true_for_text(clf, le, model: SentenceTransformer, cfg: dict, text: str,
                                   max_tokens: int = _DEFAULT_MAX_TOKENS,
                                   overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                   aggregate: str = "mean") -> float:
    """Calcula probabilidade 'true' por chunks e agrega (média por padrão)."""
    ids = _token_ids(model, text)
    if len(ids) <= max_tokens:
        q_emb = embed_query_with_style(model, text, cfg)
        return bert_probability_true(clf, le, q_emb)

    chunks = split_text_into_token_chunks(
        model, text, max_tokens, overlap_tokens)
    if not chunks:
        q_emb = embed_query_with_style(model, text, cfg)
        return bert_probability_true(clf, le, q_emb)

    probs = []
    # Batch process to be efficient
    # Build embeddings first
    embs = []
    for ch in chunks:
        embs.append(embed_query_with_style(model, ch, cfg))
    # embs is a list of arrays shape (1, d); stack
    X = np.vstack(embs)
    P = clf.predict_proba(X)
    try:
        idx_true = list(le.classes_).index("true")
    except ValueError:
        idx_true = int(np.argmax(P[0]))
    probs = P[:, idx_true] * 100.0

    if aggregate == "max":
        return float(np.max(probs))
    # default mean
    return float(np.mean(probs))


def fuse_scores(hist_score, bert_score, fonte_score=None, w_hist=0.6, w_bert=0.4, w_fontes=0.0):
    weights = np.array([w_hist, w_bert, w_fontes], dtype=float)
    scores = np.array([
        hist_score if hist_score is not None else 0.0,
        bert_score if bert_score is not None else 0.0,
        fonte_score if (fonte_score is not None) else 0.0,
    ], dtype=float)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return float(np.dot(weights, scores))


def classify_text(final_score):
    return ("VERDADEIRO", final_score) if final_score >= 50.0 else ("FALSO", 100.0 - final_score)


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Assumes rows are L2-normalized if using normalize_embeddings=True
    return A @ B.T


def _wiki_opensearch_titles(query: str, limit: int = 3) -> list[tuple[str, str]]:
    """Busca títulos na Wikipedia (pt) via Opensearch. Retorna lista (title, url)."""
    try:
        url = "https://pt.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
        }
        r = requests.get(url, params=params, timeout=3)
        r.raise_for_status()
        data = r.json()
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        links = data[3] if isinstance(data, list) and len(data) > 3 else []
        out = []
        for i, t in enumerate(titles):
            link = links[i] if i < len(
                links) else f"https://pt.wikipedia.org/wiki/{requests.utils.quote(t)}"
            out.append((t, link))
        return out
    except Exception:
        return []


def _wiki_fetch_page_text(title: str, max_chars: int = 60000) -> str:
    """Obtém HTML da página e extrai texto básico (parágrafos e cabeçalhos)."""
    try:
        endpoint = f"https://pt.wikipedia.org/api/rest_v1/page/html/{requests.utils.quote(title)}"
        r = requests.get(endpoint, headers={
                         "User-Agent": "GoldenFred/1.0"}, timeout=4)
        r.raise_for_status()
        html = r.text
        if not BeautifulSoup:
            # fallback sem bs4: remover tags simples
            text = html
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        soup = BeautifulSoup(html, "html.parser")
        # Remover elementos menos úteis
        for tag in soup.find_all(['sup', 'table', 'aside', 'span', 'figure']):
            tag.extract()
        body = soup.find('body') or soup
        parts: list[str] = []
        for el in body.find_all(['h2', 'h3', 'h4', 'p', 'li']):
            txt = el.get_text(separator=" ", strip=True)
            if not txt:
                continue
            parts.append(txt)
            if sum(len(p) for p in parts) > max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception:
        return ""


def wikipedia_dynamic_neighbors(model: SentenceTransformer, query_text: str,
                                titles_limit: int = 3,
                                k: int = 8,
                                max_tokens: int = _DEFAULT_MAX_TOKENS,
                                overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS):
    """Busca páginas no Wikipedia (dinâmico) e retorna score e matches semelhantes ao FAISS.
    Retorna (score_0_100, detalhes_matches_dict_list)
    """
    titles = _wiki_opensearch_titles(query_text, limit=titles_limit)
    if not titles:
        return 0.0, []

    # Concatenar chunks de todas páginas
    all_chunks: list[tuple[str, str, str]] = []  # (title, url, chunk)
    for title, url in titles:
        txt = _wiki_fetch_page_text(title)
        if not txt:
            continue
        ids = _token_ids(model, txt)
        if len(ids) <= max_tokens:
            chunks = [txt]
        else:
            chunks = split_text_into_token_chunks(
                model, txt, max_tokens, overlap_tokens)
        for ch in chunks:
            if ch and ch.strip():
                all_chunks.append((title, url, ch))

    if not all_chunks:
        return 0.0, []

    # Embed query and chunks
    q = embed_query(model, query_text)  # normalized
    X = []
    for _, _, ch in all_chunks:
        X.append(embed_query(model, ch))
    Xmat = np.vstack(X)  # shape (N, d)

    sims = (Xmat @ q.T).reshape(-1)  # cosine since normalized
    # Clip to [0,1] and compute top-k mean similar to FAISS behavior
    sims = np.clip(sims, 0.0, 1.0)
    if k > 0 and len(sims) > k:
        topk_idx = np.argpartition(-sims, k)[:k]
        topk = sims[topk_idx]
    else:
        topk = sims
    score = float(np.mean(topk) * 100.0)

    # Build details sorted by similarity
    order = np.argsort(-sims)
    details = []
    for i in order[:min(len(order), max(8, k))]:
        title, url, ch = all_chunks[i]
        details.append({
            "title": title,
            "url": url,
            "similaridade": float(sims[i]),
        })

    return score, details


def combined_historical_consistency_for_text(index, model: SentenceTransformer, text: str,
                                             k: int = 8,
                                             max_tokens: int = _DEFAULT_MAX_TOKENS,
                                             overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                                             aggregate: str = "max",
                                             wiki_titles: int = 3,
                                             w_faiss: float = 0.5,
                                             w_wiki: float = 0.5):
    """Combina FAISS (histórico local) com recuperação dinâmica do Wikipedia.
    Retorna (score_combined, detalhes: {faiss, wikipedia})
    """
    faiss_score, faiss_neighbors = historical_consistency_for_text(
        index, model, text, k=k, max_tokens=max_tokens, overlap_tokens=overlap_tokens, aggregate=aggregate
    )
    wiki_score, wiki_details = wikipedia_dynamic_neighbors(
        model, text, titles_limit=wiki_titles, k=k, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )
    # Normalizar pesos se uma das pontuações estiver ausente
    if wiki_score <= 0.0:
        combined = faiss_score
        wf, ww = 1.0, 0.0
    elif faiss_score <= 0.0:
        combined = wiki_score
        wf, ww = 0.0, 1.0
    else:
        s = max(1e-6, w_faiss + w_wiki)
        wf, ww = (w_faiss / s), (w_wiki / s)
        combined = float(faiss_score * wf + wiki_score * ww)

    return combined, {
        "faiss": {"score": round(faiss_score, 1), "vizinhos": faiss_neighbors},
        "wikipedia": {"score": round(wiki_score, 1), "matches": wiki_details},
        "pesos": {"faiss": round(wf, 2), "wikipedia": round(ww, 2)},
    }


WIKI_EXTRACT_URL = "https://pt.wikipedia.org/w/api.php"
WIKI_PAGE_URL_TEMPLATE = "https://pt.wikipedia.org/?curid={page_id}"
WIKI_EXTRACT_HEADERS = {
    "User-Agent": "GoldenFake/1.0 (+https://github.com/JPeeeedrs/goldenfake)"
}

_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquele", "as", "com", "como", "da", "das", "de",
    "do", "dos", "e", "em", "entre", "era", "essa", "esse", "estao", "esta",
    "estar", "foi", "ha", "isso", "isto", "ja", "mais", "mas", "na", "nas", "no",
    "nos", "nossa", "neste", "o", "os", "para", "pela", "pelas", "pelo", "pelos",
    "por", "que", "se", "sem", "ser", "sua", "são", "tambem", "tem", "uma",
    "umas", "uns", "vai", "via", "voc", "voce", "sou", "somos", "sao", "estamos",
    "estao", "eram", "serao", "seria", "seriam", "teria", "teriam", "tiveram",
    "havia", "haviam", "existem", "existe", "existia", "existiu", "cria", "criar",
    "criam", "criou", "criando", "feito", "fazer", "faz", "fazia", "fazem", "feito",
    "feito", "feitos", "muito", "pouco", "tudo", "todos", "todas", "cada", "outra",
    "outro", "outros", "outras", "sempre", "nunca", "the", "and", "this", "that",
    "who", "when", "where", "from", "into", "you"
}

_GENERIC_ENTITY_TOKENS = {
    "governo", "sociedade", "economistas", "analistas", "pessoas", "ano", "anos",
    "problema", "crise", "escandalo", "rombo", "situacao", "pais", "populacao",
    "setor", "mercado", "empresa", "empresas", "politica", "economia", "reforma",
    "programa", "beneficio", "pagamento", "trabalhadores", "federal", "nacional",
    "publico", "privado", "setor privado", "setor publico",
}

CORROBORATION_LOW_THRESHOLD = 0.3  # legacy constants (mantidos para tuning futuro, atualmente não usados)
CORROBORATION_LOW_PENALTY = 0.5


def _get_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:  # type: ignore
        _SPACY_NLP = None
        return None
    try:
        _SPACY_NLP = spacy.load("pt_core_news_sm")  # type: ignore
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP


def _extract_entities(text: str | None,
                      allowed_labels: set[str] | None = None) -> set[str]:
    if not text:
        return set()
    nlp = _get_spacy_model()

    if nlp is None:
        return _fallback_entity_candidates(text)

    try:
        doc = nlp(text)
    except Exception:
        return set()

    allowed = allowed_labels or {
        "PER", "ORG", "LOC", "GPE", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "NORP", "PERSON", "ORGANIZATION", "LOCATION",
    }

    spacy_entities: set[str] = set()
    for ent in doc.ents:
        label = ent.label_ or ""
        if allowed and label not in allowed:
            continue
        normalized = _normalize_text_for_match(ent.text)
        if not _should_keep_entity(normalized):
            continue
        spacy_entities.add(normalized)

    if len(spacy_entities) >= 3:
        return spacy_entities

    fallback_candidates = _fallback_entity_candidates(text)
    return spacy_entities | fallback_candidates


def _should_keep_entity(token: str | None) -> bool:
    if not token:
        return False
    if token in _STOPWORDS or token in _GENERIC_ENTITY_TOKENS:
        return False
    if token.replace(" ", "").isdigit():
        return False
    if len(token) <= 2:
        return False
    return True


def _fallback_entity_candidates(text: str | None) -> set[str]:
    if not text:
        return set()
    candidates: set[str] = set()
    for raw, norm in _tokenize_claim(text):
        if not _should_keep_entity(norm):
            continue
        if raw.isupper() or raw[:1].isupper():
            candidates.add(norm)
        if len(candidates) >= 12:
            break
    return candidates


def _normalize_text_for_match(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower()


def _tokenize_claim(text: str | None) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for raw in re.findall(r"\b[\w\-]+\b", text or "", flags=re.UNICODE):
        norm = _normalize_text_for_match(raw)
        if norm:
            tokens.append((raw, norm))
    return tokens


def _score_tokens(tokens: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    scored: list[tuple[float, str, str]] = []
    fallback: list[tuple[float, str, str]] = []
    for raw, norm in tokens:
        base = len(norm)
        if base == 0:
            continue
        boost = 0.0
        if any(ch.isdigit() for ch in raw):
            boost += 2.5
        if "-" in raw:
            boost += 1.0
        score = base + boost
        if norm in _STOPWORDS or base <= 3:
            fallback.append((max(1.0, score * 0.5), norm, raw))
        else:
            scored.append((score, norm, raw))
    if not scored:
        scored = fallback
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def extract_claim_keywords(text: str | None,
                           max_terms: int = 8,
                           max_phrases: int = 3) -> list[dict[str, str | float]]:
    tokens = _tokenize_claim(text or "")
    ranked = _score_tokens(tokens)
    keywords: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for score, norm, raw in ranked:
        if not norm or norm in seen:
            continue
        keywords.append({
            "token": raw,
            "normalized": norm,
            "score": round(float(score), 2),
        })
        seen.add(norm)
        if len(keywords) >= max_terms:
            break

    phrase_candidates: list[tuple[float, str, str]] = []
    for i in range(len(tokens) - 1):
        raw1, norm1 = tokens[i]
        raw2, norm2 = tokens[i + 1]
        if norm1 in _STOPWORDS or norm2 in _STOPWORDS:
            continue
        phrase_norm = f"{norm1} {norm2}"
        if len(phrase_norm) <= 7:
            continue
        phrase_raw = f"{raw1} {raw2}"
        phrase_score = len(norm1) + len(norm2) + 1.0
        phrase_candidates.append((phrase_score, phrase_norm, phrase_raw))
    phrase_candidates.sort(key=lambda item: (-item[0], item[1]))

    added = 0
    for score, norm, raw in phrase_candidates:
        if norm in seen:
            continue
        keywords.append({
            "token": raw,
            "normalized": norm,
            "score": round(float(score), 2),
            "is_phrase": True,
        })
        seen.add(norm)
        added += 1
        if added >= max_phrases:
            break

    return keywords


@lru_cache(maxsize=4096)
def fetch_wikipedia_article(page_id: str | int) -> dict | None:
    pid = str(page_id)
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": pid,
    }
    try:
        resp = requests.get(WIKI_EXTRACT_URL, params=params,
                            headers=WIKI_EXTRACT_HEADERS, timeout=4)
        resp.raise_for_status()
        data = resp.json().get("query", {}).get("pages", {})
        page = data.get(pid)
        if not page or page.get("missing") is not None:
            return None
        return {
            "id": pid,
            "title": page.get("title"),
            "text": page.get("extract") or "",
            "url": WIKI_PAGE_URL_TEMPLATE.format(page_id=pid),
        }
    except Exception:
        return None


def _match_keywords(article_text: str, keywords: list[dict[str, str | float]]):
    normalized_article = _normalize_text_for_match(article_text)
    matched: list[str] = []
    missing: list[str] = []
    for kw in keywords:
        norm_kw = kw.get("normalized")
        token = kw.get("token")
        if not norm_kw or not token:
            continue
        if norm_kw in normalized_article:
            matched.append(str(token))
        else:
            missing.append(str(token))
    if not keywords:
        status = "insufficient_keywords"
    elif matched:
        status = "corroborated"
    else:
        status = "not_corroborated"
    return {
        "status": status,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


def _is_fuzzy_match(entity_a: str | None, entity_b: str | None) -> bool:
    a = (entity_a or "").strip().lower()
    b = (entity_b or "").strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def faiss_claim_corroboration(neighbors: list[tuple[int, float]],
                              metadata: list[dict],
                              claim_text: str,
                              top_articles: int = 3) -> dict | None:
    if not neighbors or not metadata:
        return None
    keywords = extract_claim_keywords(claim_text)
    query_entities = _extract_entities(claim_text)
    total_query_ents = len(query_entities)
    results: list[dict] = []
    overall_status = "insufficient_keywords" if not keywords else "not_corroborated"
    best_corroboration = 0.0

    for neighbor in neighbors[:top_articles]:
        idx, similarity = neighbor
        if idx is None or idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx] or {}
        page_id = entry.get("id")
        if not page_id:
            continue
        chunk_index = entry.get("chunk_index")
        article = fetch_wikipedia_article(str(page_id))
        if not article:
            results.append({
                "page_id": str(page_id),
                "chunk_index": chunk_index,
                "similaridade": round(float(similarity or 0.0), 3),
                "status": "unavailable",
            })
            continue

        article_text = article.get("text", "") or ""
        match = _match_keywords(article_text, keywords)
        article_entities = _extract_entities(article_text, None)
        article_text_lower = article_text.lower()

        matched_entities: set[str] = set()
        if query_entities:
            for q_ent in query_entities:
                found = False
                for a_ent in article_entities:
                    if _is_fuzzy_match(q_ent, a_ent):
                        matched_entities.add(q_ent)
                        found = True
                        break
                if not found and q_ent.lower() in article_text_lower:
                    matched_entities.add(q_ent)

        missing_entities = sorted(list(query_entities - matched_entities)) if query_entities else []
        matched_list = sorted(list(matched_entities)) if matched_entities else []

        ratio = 0.0
        if total_query_ents > 0:
            matched_count = len(matched_entities)
            ratio = matched_count / total_query_ents
            if ratio > 0.5:
                ratio = min(1.0, ratio * 1.2)
        corroboration_ratio = round(float(ratio), 4) if total_query_ents else None

        if ratio > best_corroboration:
            best_corroboration = ratio

        current_status = "not_corroborated"
        if ratio >= 0.6:
            current_status = "corroborated"
        elif ratio >= 0.3:
            current_status = "partially_corroborated"

        results.append({
            "page_id": str(page_id),
            "title": article.get("title"),
            "url": article.get("url"),
            "chunk_index": chunk_index,
            "similaridade": round(float(similarity or 0.0), 3),
            "status": current_status,
            "matched_keywords": match.get("matched_keywords"),
            "missing_keywords": match.get("missing_keywords"),
            "matched_entities": matched_list,
            "missing_entities": missing_entities,
            "article_entities": sorted(article_entities),
            "corroboration_ratio": corroboration_ratio,
        })

        if current_status == "corroborated":
            overall_status = "corroborated"
        elif current_status == "partially_corroborated" and overall_status != "corroborated":
            overall_status = "partially_corroborated"

    if total_query_ents == 0:
        best_corroboration = 1.0

    if overall_status != "corroborated" and best_corroboration > 0.5:
        overall_status = "partially_corroborated"

    return {
        "status": overall_status,
        "keywords": keywords,
        "query_entities": sorted(query_entities),
        "corroboration_score": round(float(best_corroboration), 4),
        "articles": results,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Verificador de fatos com FAISS + BERT + Fontes Externas")
    parser.add_argument(
        "--text", type=str, help="Texto a analisar. Se omitido, será solicitado via stdin.")
    parser.add_argument("--k", type=int, default=8,
                        help="Número de vizinhos no FAISS")
    parser.add_argument("--w_hist", type=float, default=0.6,
                        help="Peso da consistência histórica")
    parser.add_argument("--w_bert", type=float, default=0.4,
                        help="Peso da classificação BERT")
    parser.add_argument("--w_fontes", type=float, default=0.0,
                        help="Peso das fontes externas")
    # novos parâmetros de chunking/agregação para CLI
    parser.add_argument("--max_tokens", type=int, default=_DEFAULT_MAX_TOKENS,
                        help="Tamanho máximo do chunk em tokens")
    parser.add_argument("--overlap_tokens", type=int, default=_DEFAULT_OVERLAP_TOKENS,
                        help="Sobreposição entre chunks em tokens")
    parser.add_argument("--hist_agg", choices=["max", "mean"],
                        default="max", help="Agregação da consistência histórica")
    parser.add_argument(
        "--bert_agg", choices=["mean", "max"], default="mean", help="Agregação da probabilidade BERT")
    parser.add_argument("--json", action="store_true",
                        help="Imprimir saída em JSON detalhado")
    args = parser.parse_args()

    text = args.text or input("Digite o texto que deseja analisar:\n")
    text = text.strip()
    if not text:
        print("Texto vazio.")
        return

    index, metadata, vcfg = load_vector_store()
    clf, le, sbert, ccfg = load_classifier()
    online_adaptor = load_online_adaptor(le, sbert, ccfg)

    # Usar wrappers com chunking (512 tokens + overlap por padrão), mantendo comportamento para textos curtos
    hist_score, neighbors = historical_consistency_for_text(
        index, sbert, text, k=args.k, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, aggregate=args.hist_agg
    )

    bert_score_true_raw = bert_probability_true_for_text(
        clf, le, sbert, ccfg, text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, aggregate=args.bert_agg
    )
    bert_score_true = bert_score_true_raw
    entity_avg = None
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    # Consultar fontes externas (opcional; requer chaves em .env para melhor cobertura)
    fonte_score, fonte_details, entity_block = verify_with_external_sources(text, sbert)
    if entity_block:
        bert_score_true, entity_avg = blend_score_with_entities(
            bert_score_true_raw, entity_block, ENTITY_BERT_WEIGHT)
        bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    # Adicionar probabilidade do adaptador online, se treinado
    online_score = online_adaptor.prob_true_for_text(
        text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, aggregate=args.bert_agg)
    alpha_online = online_adaptor.alpha() if online_adaptor.is_trained() else 0.0

    final_score = fuse_scores(hist_score, bert_score_true, fonte_score,
                              args.w_hist, args.w_bert * (1 - alpha_online), args.w_fontes)
    if online_score is not None:
        final_score = fuse_scores(
            final_score, online_score, None, 1 - alpha_online, alpha_online, 0.0)
    final_label, _ = classify_text(final_score)

    result = {
        "texto_analisado": text,
        "historico": {
            "consistencia": round(hist_score, 1),
            "k": args.k,
            "vizinhos": neighbors,
            "aggregate": args.hist_agg,
        },
        "bert": {
            "rotulo": bert_label,
            "prob_true": round(bert_score_true, 1),
            "prob_true_raw": round(bert_score_true_raw, 1),
            "entity_avg_percent": (round(entity_avg, 1) if entity_avg is not None else None),
            "entity_blend_weight": round(ENTITY_BERT_WEIGHT, 2),
            "aggregate": args.bert_agg,
        },
        "confirmacao_fontes": {
            "fonte_score": round(fonte_score, 1),
            "detalhes": fonte_details,
            "entidades_verificadas": entity_block,
        },
        "online_adaptor": {
            "prob_true": round(online_score, 1) if online_score is not None else None,
            "alpha": round(alpha_online, 2),
        },
        "final": {
            "rotulo": final_label,
            "score": round(final_score, 1),
        },
        "pesos": {"historico": args.w_hist, "bert": args.w_bert, "fontes": args.w_fontes},
        "chunking": {"max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens},
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("==== GoldenFred - Resultado ====")
        print(
            f"Consistência histórica: {result['historico']['consistencia']:.1f}% (k={args.k}, agg={args.hist_agg})")
        print(
            f"BERT: {result['bert']['rotulo']} (Prob. verdadeiro: {result['bert']['prob_true']:.1f}% | agg={args.bert_agg})")
        print(
            f"Fontes externas: {result['confirmacao_fontes']['fonte_score']:.1f}%")
        if online_score is not None:
            print(
                f"Adaptador online: {result['online_adaptor']['prob_true']:.1f}% (alpha={result['online_adaptor']['alpha']:.2f})")
        print("--------------------------------")
        print(
            f"Classificação final: {final_label} | Score: {result['final']['score']:.1f}%")


if __name__ == "__main__":
    main()
