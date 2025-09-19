import os
import json
import argparse
import numpy as np
import faiss
import joblib
import re
from sentence_transformers import SentenceTransformer
from external_sources import verify_with_external_sources

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


def main():
    parser = argparse.ArgumentParser(description="Verificador de fatos com FAISS + BERT + Fontes Externas")
    parser.add_argument("--text", type=str, help="Texto a analisar. Se omitido, será solicitado via stdin.")
    parser.add_argument("--k", type=int, default=8, help="Número de vizinhos no FAISS")
    parser.add_argument("--w_hist", type=float, default=0.6, help="Peso da consistência histórica")
    parser.add_argument("--w_bert", type=float, default=0.4, help="Peso da classificação BERT")
    parser.add_argument("--w_fontes", type=float, default=0.0, help="Peso das fontes externas")
    parser.add_argument("--json", action="store_true", help="Imprimir saída em JSON detalhado")
    args = parser.parse_args()

    text = args.text or input("Digite o texto que deseja analisar:\n")
    text = text.strip()
    if not text:
        print("Texto vazio.")
        return

    index, metadata, vcfg = load_vector_store()
    clf, le, sbert, ccfg = load_classifier()

    # Usar wrappers com chunking (512 tokens + overlap), mantendo comportamento para textos curtos
    hist_score, neighbors = historical_consistency_for_text(index, sbert, text, k=args.k)

    bert_score_true = bert_probability_true_for_text(clf, le, sbert, ccfg, text)
    bert_label = "provavelmente verdadeiro" if bert_score_true >= 50.0 else "provavelmente falso"

    fonte_score, fonte_details = verify_with_external_sources(text, sbert)

    final_score = fuse_scores(hist_score, bert_score_true, fonte_score, args.w_hist, args.w_bert, args.w_fontes)
    final_label, final_conf = classify_text(final_score)

    print("Texto analisado:", text[:300] + ("..." if len(text) > 300 else ""))
    print(f"Consistência histórica: {hist_score:.0f}%")
    print(f"Classificação BERT: \"{bert_label}\" ({bert_score_true:.0f}%)")
    print(f"Confirmação por fontes externas: {fonte_score:.0f}%")
    print(f"Resultado final: \"{final_label}\" com confiança de {final_score:.0f}%")

    result_json = {
        "texto_analisado": text,
        "historico": {"consistencia": round(hist_score, 1), "k": args.k},
        "bert": {"rotulo": bert_label, "prob_true": round(bert_score_true, 1)},
        "confirmacao_fontes": {"fonte_score": round(fonte_score, 1), "detalhes": fonte_details},
        "final": {"rotulo": final_label, "score": round(final_score, 1)},
        "pesos": {"historico": args.w_hist, "bert": args.w_bert, "fontes": args.w_fontes},
    }
    if args.json:
        print("\nJSON:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
