import os
import json
import joblib
import numpy as np
import re
import argparse
from typing import List, Tuple, Iterable, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer

# Tentar carregar spaCy e um modelo PT; fallback para regex se indisponível
try:
    import spacy  # type: ignore
    try:
        _NLP = spacy.load("pt_core_news_sm")  # pode não estar instalado
    except Exception:
        _NLP = None
except Exception:  # spaCy não instalado
    spacy = None
    _NLP = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_full_texts.json")
# Salvar modelos no diretório raiz do projeto (../models)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(MODEL_DIR, exist_ok=True)

SENSATIONAL_LEXICON = [
    "urgente", "chocante", "escândalo", "bomba", "imperdível", "você não vai acreditar",
    "revelado", "exclusivo", "alerta", "atenção", "incrível", "chocado", "verdadeiro?",
    "mentira", "fraude", "golpe", "boato", "polêmico", "assustador", "impactante",
]

PUNCT_SET = set(list("!?"))


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
    # Normalizações simples e clamping
    feats = np.array([
        upper_ratio,          # 0
        allcaps_ratio,        # 1
        punct_ratio,          # 2
        min(exclam, 10) / 10, # 3
        min(quest, 10) / 10,  # 4
        min(avg_word_len, 20) / 20, # 5
        ttr,                  # 6
        min(lex_density, 1.0) # 7
    ], dtype=float)
    return feats


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Apenas entradas com texto e label válido
    data = [x for x in data if isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("text").strip() and x.get("label")]
    texts = [x["text"].strip() for x in data]
    labels = [str(x["label"]).lower() for x in data]
    return texts, labels


def embed_texts(model, texts: List[str], batch_size=64, normalize=True):
    try:
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=True)
    except TypeError:
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
    return embs


# --- Debias helpers ---

def load_protected_terms(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        terms = cfg.get("protected_terms") or []
        return [t.strip() for t in terms if t and isinstance(t, str)]
    except Exception:
        return []


# Regex fallback para nomes próprios (sequências de palavras Capitalizadas)
# Considera letras acentuadas comuns em PT
_CAP = r"A-ZÁÀÂÃÉÊÍÓÔÕÚÇ"
_LOW = r"a-záàâãéêíóôõúç"
PROPER_SEQ_RE = re.compile(rf"\b([{_CAP}][{_LOW}]+(?:\s+[{_CAP}][{_LOW}]+){{0,3}})\b")


def extract_entities(text: str, nlp, types: Iterable[str]) -> List[str]:
    if nlp is None:
        return []
    try:
        doc = nlp(text)
        out = [ent.text for ent in doc.ents if ent.label_ in set(types)]
        return out
    except Exception:
        return []


def extract_proper_names_regex(text: str) -> List[str]:
    # Encontra sequências capitalizadas; pode incluir falsos positivos (ex.: início de sentença)
    return [m.group(0) for m in PROPER_SEQ_RE.finditer(text)]


def get_protected_mentions(text: str, terms: List[str], mode: str, nlp, ent_types: Iterable[str]) -> List[str]:
    mentions: List[str] = []
    mode = (mode or "auto").lower()
    if mode in ("list", "both") and terms:
        for t in terms:
            if not t:
                continue
            if re.search(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE):
                mentions.append(t)
    if mode in ("auto", "both"):
        ents = extract_entities(text, nlp, ent_types)
        if not ents:  # fallback para regex
            ents = extract_proper_names_regex(text)
        mentions.extend(ents)
    # Normalizar e deduplicar por lower/strip
    norm_seen = set()
    uniq: List[str] = []
    for m in mentions:
        k = m.strip().lower()
        if not k:
            continue
        if k not in norm_seen:
            norm_seen.add(k)
            uniq.append(m.strip())
    return uniq


def contains_protected_dynamic(text: str, terms: List[str], mode: str, nlp, ent_types: Iterable[str]) -> bool:
    return len(get_protected_mentions(text, terms, mode, nlp, ent_types)) > 0


def mask_mentions(text: str, mentions: List[str], placeholder: str = "[ENT]") -> str:
    if not mentions:
        return text
    # Ordenar por tamanho desc para evitar sobreposição
    ordered = sorted(set(mentions), key=lambda s: len(s), reverse=True)
    out = text
    for m in ordered:
        # substituir com word boundary quando aplicável, mantendo case-insensitive
        patt = re.compile(rf"\b{re.escape(m)}\b", flags=re.IGNORECASE)
        out = patt.sub(placeholder, out)
    return out


def augment_with_masked(texts: List[str], labels: np.ndarray, terms: List[str], mode: str, nlp, ent_types: Iterable[str], alpha: float = 0.5) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Retorna (texts_aug, labels_aug, weights_aug). Cria cópias mascaradas de amostras que contêm menções protegidas
    detectadas automaticamente (NER/regex) e/ou da lista.
    alpha: peso relativo das cópias (1.0 mantém igual, 0.5 metade do peso).
    """
    texts_out = list(texts)
    labels_out = list(labels)
    weights_out = [1.0] * len(texts)

    for i, txt in enumerate(texts):
        mentions = get_protected_mentions(txt, terms, mode, nlp, ent_types)
        if mentions:
            masked = mask_mentions(txt, mentions)
            if masked != txt:
                texts_out.append(masked)
                labels_out.append(labels[i])
                weights_out.append(alpha)
    return texts_out, np.array(labels_out), np.array(weights_out, dtype=float)


def compute_protected_sample_weights(texts: List[str], labels: np.ndarray, terms: List[str], mode: str, nlp, ent_types: Iterable[str]) -> np.ndarray:
    """
    Balanceia a distribuição de rótulos dentro do subconjunto que contém menções protegidas.
    Para amostras fora do subconjunto, peso = 1.0.
    Para amostras dentro do subconjunto e com rótulo y, peso = N_sub / (2 * count_sub_y).
    """
    n = len(texts)
    in_sub = np.array([contains_protected_dynamic(t, terms, mode, nlp, ent_types) for t in texts], dtype=bool)
    weights = np.ones(n, dtype=float)

    if in_sub.any():
        idxs = np.where(in_sub)[0]
        y_sub = labels[idxs]
        classes, counts = np.unique(y_sub, return_counts=True)
        total = len(idxs)
        count_map = {int(c): int(k) for c, k in zip(classes, counts)}
        # garantir 2 classes
        for c in np.unique(labels):
            count_map.setdefault(int(c), 0)
        for i in idxs:
            c = int(labels[i])
            denom = 2 * max(count_map.get(c, 1), 1)
            weights[i] = total / denom
    return weights


def main():
    parser = argparse.ArgumentParser(description="Treino classificador com opções de debias (auto nomes próprios/NER) e estilo")
    parser.add_argument("--debias", choices=["none", "mask", "weight", "both"], default="both", help="Estratégia de debias")
    parser.add_argument("--protected_mode", choices=["list", "auto", "both"], default="auto", help="Fonte de menções protegidas: lista, auto (NER/regex) ou ambos")
    parser.add_argument("--entity_types", type=str, default="PERSON,ORG,GPE", help="Tipos de entidades (spaCy) para mascarar, separados por vírgula")
    parser.add_argument("--protected_terms_file", type=str, default=os.path.join(BASE_DIR, "debias_config.json"), help="Arquivo JSON com protected_terms")
    parser.add_argument("--augment_alpha", type=float, default=0.5, help="Peso das cópias mascaradas (se mask/both)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_style", action="store_true", help="Adicionar features de estilo ao classificador")
    parser.add_argument("--style_scale", type=float, default=0.2, help="Fator para reduzir a influência das features de estilo após padronização")
    parser.add_argument("--calibrate", action="store_true", help="Aplicar CalibratedClassifierCV (isotonic, cv=5) para calibrar probabilidades")
    args = parser.parse_args()

    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset não encontrado: {DATASET_PATH}")
        return

    texts, labels = load_dataset(DATASET_PATH)
    if not texts:
        print("Nenhum dado válido no dataset.")
        return

    print(f"Amostras totais: {len(texts)}")

    # Codificar labels
    le = LabelEncoder()
    y_all = le.fit_transform(labels)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(texts, y_all, test_size=args.test_size, random_state=args.random_state, stratify=y_all)

    print("Carregando modelo de embeddings...")
    sbert = SentenceTransformer(EMBED_MODEL_NAME)

    # Debias: carregar termos protegidos e modo
    ent_types = [t.strip() for t in args.entity_types.split(",") if t.strip()]
    if args.protected_mode in ("list", "both"):
        protected_terms = load_protected_terms(args.protected_terms_file)
        terms_info = protected_terms if protected_terms else "nenhum"
    else:
        protected_terms = []
        terms_info = "ignorado (modo auto)"
    print(f"Debias: {args.debias} | protected_mode: {args.protected_mode} | ent_types: {ent_types} | termos: {terms_info}")

    sample_weight = np.ones(len(X_train_txt), dtype=float)

    texts_train_for_embed = list(X_train_txt)
    y_train_for_fit = np.array(y_train)

    if args.debias in ("mask", "both"):
        texts_train_for_embed, y_train_for_fit, aug_weights = augment_with_masked(
            texts_train_for_embed, y_train_for_fit, protected_terms, args.protected_mode, _NLP, ent_types, alpha=args.augment_alpha
        )
        # expandir sample_weight se houve cópias
        if len(aug_weights) > len(X_train_txt):
            sample_weight = np.concatenate([sample_weight, aug_weights[len(X_train_txt):]])

    if args.debias in ("weight", "both"):
        w = compute_protected_sample_weights(texts_train_for_embed, y_train_for_fit, protected_terms, args.protected_mode, _NLP, ent_types)
        if w.shape[0] != len(texts_train_for_embed):
            w = np.ones(len(texts_train_for_embed), dtype=float)
        sample_weight = w

    print("Gerando embeddings de treino...")
    X_train_sbert = embed_texts(sbert, texts_train_for_embed)

    if args.use_style:
        X_train_style = np.vstack([extract_style_features(t) for t in texts_train_for_embed])
        style_scaler = StandardScaler()
        X_train_style_s = style_scaler.fit_transform(X_train_style) * float(args.style_scale)
        X_train = np.hstack([X_train_sbert, X_train_style_s])
    else:
        style_scaler = None
        X_train = X_train_sbert

    print("Treinando classificador (LogisticRegression)...")
    # Evitar dupla ponderação: se usamos sample_weight (debias weight/both), não usar class_weight
    use_sample_w = args.debias in ("weight", "both")
    cls_weight = None if use_sample_w else "balanced"
    base_clf = LogisticRegression(max_iter=2000, class_weight=cls_weight, C=0.1, solver="lbfgs")

    if args.calibrate:
        clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=5)
    else:
        clf = base_clf

    try:
        clf.fit(X_train, y_train_for_fit, sample_weight=(sample_weight if use_sample_w else None))
    except TypeError:
        clf.fit(X_train, y_train_for_fit)

    print("Avaliando no conjunto de teste...")
    X_test_sbert = embed_texts(sbert, X_test_txt)
    if args.use_style:
        X_test_style = np.vstack([extract_style_features(t) for t in X_test_txt])
        X_test_style_s = style_scaler.transform(X_test_style) * float(args.style_scale)
        X_test = np.hstack([X_test_sbert, X_test_style_s])
    else:
        X_test = X_test_sbert
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Salvar artefatos
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    if args.use_style and style_scaler is not None:
        joblib.dump(style_scaler, os.path.join(MODEL_DIR, "style_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": EMBED_MODEL_NAME,
            "normalize": True,
            "debias": args.debias,
            "protected_mode": args.protected_mode,
            "entity_types": ent_types,
            "protected_terms_file": os.path.basename(args.protected_terms_file),
            "spacy": bool(_NLP is not None),
            "style_features": bool(args.use_style),
            "style_feature_count": 8,
            "style_scale": float(args.style_scale),
            "calibrated": bool(args.calibrate)
        }, f, ensure_ascii=False, indent=2)

    print("Modelo salvo em models/ (classifier.joblib, label_encoder.joblib, config.json, style_scaler.joblib opcional)")


if __name__ == "__main__":
    main()
