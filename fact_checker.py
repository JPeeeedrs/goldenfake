import os
import json
import argparse
import numpy as np
import faiss
import joblib
import re
import requests
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

# --- Chunking config padrão ---
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 64

# --- Online adaptive classifier (incremental fine-tuning) ---
try:
    # sklearn is in requirements; import lazily to avoid heavy deps on tools using this file as a lib
    from sklearn.linear_model import SGDClassifier  # type: ignore
except Exception:  # pragma: no cover
    SGDClassifier = None  # type: ignore

ONLINE_ADAPTOR_PATH = os.path.join(MODEL_DIR, "online_adaptor.joblib")


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
        self._classes_idx = np.arange(len(list(le.classes_))) if hasattr(le, "classes_") else None
        # cache style scaler usage via embed_query_with_style

    def is_trained(self) -> bool:
        return self.inc_clf is not None and getattr(self.inc_clf, "classes_", None) is not None

    def _ensure_init(self):
        if self.inc_clf is None:
            if SGDClassifier is None:
                raise RuntimeError("SGDClassifier indisponível.")
            # log_loss gives probabilities
            self.inc_clf = SGDClassifier(loss="log_loss", alpha=1e-4, penalty="l2", max_iter=1, learning_rate="optimal", random_state=42)

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
            # primeira chamada precisa de classes
            try:
                self.inc_clf.partial_fit(X, y_idx, classes=self._classes_idx)
            except Exception:
                # fallback sem weights
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
        # persistir
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
        # probabilidade da classe "true" segundo label encoder
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
        chunks = split_text_into_token_chunks(self.sbert, text, max_tokens, overlap_tokens)
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


def extract_style_features(text: str) -> np.ndarray:
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
        return tok.encode(text, add_special_tokens=False)
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
    use_decode = hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'decode')
    for start in range(0, len(ids), stride):
        window = ids[start:start + max_tokens]
        if not window:
            break
        if use_decode:
            chunk_text = model.tokenizer.decode(window, skip_special_tokens=True)
        else:
            if isinstance(window[0], str):
                chunk_text = " ".join(window)
            else:
                # Last-resort: slice original text by char length proportionally
                # Not ideal, but ensures some chunk content
                approx = int(len(text) * min(1.0, (start + len(window)) / max(1, len(ids))))
                prev = int(len(text) * (start / max(1, len(ids))))
                chunk_text = text[prev:approx]
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def load_vector_store():
    if not (os.path.isfile(INDEX_PATH) and os.path.isfile(META_PATH) and os.path.isfile(VEC_CONFIG_PATH)):
        raise FileNotFoundError("Vector store não encontrado. Execute build_faiss_index.py primeiro.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(VEC_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return index, metadata, cfg


def load_classifier():
    if not (os.path.isfile(CLS_PATH) and os.path.isfile(LE_PATH) and os.path.isfile(CLS_CONFIG_PATH)):
        raise FileNotFoundError("Modelo não encontrado. Execute train_classifier.py primeiro.")
    clf = joblib.load(CLS_PATH)
    le = joblib.load(LE_PATH)
    with open(CLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sbert = SentenceTransformer(cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
    # Garantir janelas de até 512 tokens por padrão para futuras entradas longas
    try:
        sbert.max_seq_length = int(cfg.get("max_seq_length", _DEFAULT_MAX_TOKENS))
    except Exception:
        sbert.max_seq_length = _DEFAULT_MAX_TOKENS
    return clf, le, sbert, cfg


def embed_query(model, text: str, normalize=True):
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=normalize)
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

    chunks = split_text_into_token_chunks(model, text, max_tokens, overlap_tokens)
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

    chunks = split_text_into_token_chunks(model, text, max_tokens, overlap_tokens)
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
            link = links[i] if i < len(links) else f"https://pt.wikipedia.org/wiki/{requests.utils.quote(t)}"
            out.append((t, link))
        return out
    except Exception:
        return []


def _wiki_fetch_page_text(title: str, max_chars: int = 60000) -> str:
    """Obtém HTML da página e extrai texto básico (parágrafos e cabeçalhos)."""
    try:
        endpoint = f"https://pt.wikipedia.org/api/rest_v1/page/html/{requests.utils.quote(title)}"
        r = requests.get(endpoint, headers={"User-Agent": "GoldenFred/1.0"}, timeout=4)
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
            chunks = split_text_into_token_chunks(model, txt, max_tokens, overlap_tokens)
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


def main():
    parser = argparse.ArgumentParser(description="Verificador de fatos com FAISS + BERT + Fontes Externas")
    parser.add_argument("--text", type=str, help="Texto a analisar. Se omitido, será solicitado via stdin.")
    parser.add_argument("--k", type=int, default=8, help="Número de vizinhos no FAISS")
    parser.add_argument("--w_hist", type=float, default=0.6, help="Peso da consistência histórica")
    parser.add_argument("--w_bert", type=float, default=0.4, help="Peso da classificação BERT")
    parser.add_argument("--w_fontes", type=float, default=0.0, help="Peso das fontes externas")
    # novos parâmetros de chunking/agregação para CLI
    parser.add_argument("--max_tokens", type=int, default=_DEFAULT_MAX_TOKENS, help="Tamanho máximo do chunk em tokens")
    parser.add_argument("--overlap_tokens", type=int, default=_DEFAULT_OVERLAP_TOKENS, help="Sobreposição entre chunks em tokens")
    parser.add_argument("--hist_agg", choices=["max", "mean"], default="max", help="Agregação da consistência histórica")
    parser.add_argument("--bert_agg", choices=["mean", "max"], default="mean", help="Agregação da probabilidade BERT")
    parser.add_argument("--json", action="store_true", help="Imprimir saída em JSON detalhado")
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

    bert_score_true = bert_probability_true_for_text(
        clf, le, sbert, ccfg, text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, aggregate=args.bert_agg
    )
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    # Consultar fontes externas (opcional; requer chaves em .env para melhor cobertura)
    fonte_score, fonte_details = verify_with_external_sources(text, sbert)

    # Adicionar probabilidade do adaptador online, se treinado
    online_score = online_adaptor.prob_true_for_text(text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, aggregate=args.bert_agg)
    alpha_online = online_adaptor.alpha() if online_adaptor.is_trained() else 0.0

    final_score = fuse_scores(hist_score, bert_score_true, fonte_score, args.w_hist, args.w_bert * (1 - alpha_online), args.w_fontes)
    if online_score is not None:
        final_score = fuse_scores(final_score, online_score, None, 1 - alpha_online, alpha_online, 0.0)
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
            "aggregate": args.bert_agg,
        },
        "confirmacao_fontes": {
            "fonte_score": round(fonte_score, 1),
            "detalhes": fonte_details,
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
        print(f"Consistência histórica: {result['historico']['consistencia']:.1f}% (k={args.k}, agg={args.hist_agg})")
        print(f"BERT: {result['bert']['rotulo']} (Prob. verdadeiro: {result['bert']['prob_true']:.1f}% | agg={args.bert_agg})")
        print(f"Fontes externas: {result['confirmacao_fontes']['fonte_score']:.1f}%")
        if online_score is not None:
            print(f"Adaptador online: {result['online_adaptor']['prob_true']:.1f}% (alpha={result['online_adaptor']['alpha']:.2f})")
        print("--------------------------------")
        print(f"Classificação final: {final_label} | Score: {result['final']['score']:.1f}%")


if __name__ == "__main__":
    main()
