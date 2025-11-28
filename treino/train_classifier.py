import os
import json
import joblib
import numpy as np
import re
import argparse
from typing import List, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar spaCy (conforme instalação: python -m spacy download pt_core_news_sm)
try:
    import spacy
    try:
        _NLP = spacy.load("pt_core_news_sm")
        logger.info("✓ spaCy pt_core_news_sm carregado com sucesso!")
    except Exception as e:
        _NLP = None
        logger.warning(
            f"⚠ Modelo spaCy pt_core_news_sm não disponível ({e}). Usando fallback regex.")
except Exception as e:
    spacy = None
    _NLP = None
    logger.warning(f"⚠ spaCy não instalado ({e}). Usando fallback regex.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_full_texts.json")
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(MODEL_DIR, exist_ok=True)

# Léxico sensacionalista expandido (conforme sugestão Gemini)
SENSATIONAL_LEXICON = [
    # Urgência/Atenção
    "urgente", "chocante", "escândalo", "bomba", "imperdível", "alerta",
    "grave", "atenção", "compartilhem", "divulguem", "viral",

    # Teoria da Conspiração/Segredo
    "revelado", "escondido", "censurado", "mídia não mostra", "ninguém fala",
    "verdade oculta", "plano secreto", "bastidores", "farsa",

    # Certeza Absoluta (Fake News odeia dúvida)
    "comprovado", "sem dúvida", "absoluta verdade", "confirmado", "fato",

    # Emoção Negativa/Ataque
    "vergonha", "absurdo", "criminoso", "traição", "covardia", "destruição",
    "ameaça", "perigo",

    # Termos originais mantidos
    "exclusivo", "incrível", "verdadeiro?", "mentira", "fraude", "golpe",
    "boato", "polêmico", "assustador"
]
PUNCT_SET = set("!?")


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0


def extract_style_features(text: str) -> np.ndarray:
    """Extrai 8 features de estilo textual normalizadas."""
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

    exclam = min(t.count("!"), 10) / 10.0
    quest = min(t.count("?"), 10) / 10.0

    # Contagem de léxico sensacionalista
    low = t.lower()
    lex_count = sum(1 for kw in SENSATIONAL_LEXICON if kw in low)
    lex_density = _safe_div(lex_count, max(n_words, 1))

    return np.array([
        upper_ratio,
        allcaps_ratio,
        punct_ratio,
        exclam,
        quest,
        min(avg_word_len, 20) / 20.0,
        ttr,
        min(lex_density, 1.0)
    ], dtype=np.float32)


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    """Carrega dataset com validação."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validação e limpeza
    valid_data = []
    for x in data:
        if not isinstance(x, dict):
            continue
        text = x.get("text")
        label = x.get("label")
        if isinstance(text, str) and text.strip() and label:
            valid_data.append(
                {"text": text.strip(), "label": str(label).lower()})

    if not valid_data:
        raise ValueError("Nenhum dado válido encontrado no dataset")

    texts = [x["text"] for x in valid_data]
    labels = [x["label"] for x in valid_data]

    logger.info(f"Dataset carregado: {len(texts)} amostras válidas")
    return texts, labels


def chunk_text_optimized(model: SentenceTransformer, text: str,
                         max_tokens: int = 512,
                         overlap_tokens: int = 128) -> List[str]:
    """Divide texto em chunks com overlap, otimizado."""
    try:
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(
            text, add_special_tokens=False, truncation=False)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        stride = max(1, max_tokens - overlap_tokens)

        for start in range(0, len(tokens), stride):
            window = tokens[start:start + max_tokens]
            if not window:
                break
            chunk_text = tokenizer.decode(
                window, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else [text]

    except Exception as e:
        logger.warning(f"Erro no chunking: {e}. Retornando texto completo.")
        return [text]


def embed_texts_batch(model: SentenceTransformer, texts: List[str],
                      batch_size: int = 32,
                      max_tokens: int = 512,
                      overlap_tokens: int = 128,
                      normalize: bool = True) -> np.ndarray:
    """Gera embeddings com chunking e agregação por média, otimizado para batch."""
    all_embeddings = []

    for text in texts:
        chunks = chunk_text_optimized(model, text, max_tokens, overlap_tokens)

        # Processar chunks em batch
        chunk_embs = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        # Média dos embeddings dos chunks
        if len(chunk_embs.shape) == 1:
            chunk_embs = chunk_embs.reshape(1, -1)

        emb = np.mean(chunk_embs, axis=0)
        all_embeddings.append(emb)

    return np.vstack(all_embeddings).astype(np.float32)


def extract_entities_simple(text: str, nlp) -> List[str]:
    """Extração simples de entidades nomeadas."""
    if nlp is None:
        # Fallback: regex para nomes próprios (simples E compostos)
        # {0,3} permite nomes simples (José) e compostos (José Silva, Jair Messias Bolsonaro)
        pattern = r'\b([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+){0,3})\b'
        matches = re.findall(pattern, text)
        return list(set(matches))[:10]

    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in {
            "PERSON", "ORG", "GPE", "LOC"}]
        return list(set(entities))[:10]
    except Exception:
        return []


def mask_entities_in_text(text: str, entities: List[str], placeholder: str = "[ENT]") -> str:
    """Mascara entidades no texto."""
    if not entities:
        return text

    masked = text
    # Ordenar por tamanho decrescente para evitar sobreposição
    for entity in sorted(set(entities), key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(entity)}\b", flags=re.IGNORECASE)
        masked = pattern.sub(placeholder, masked)

    return masked


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Calcula pesos balanceados por classe."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = np.ones(len(y), dtype=np.float32)

    for cls, count in zip(unique, counts):
        class_weight = total / (len(unique) * count)
        weights[y == cls] = class_weight

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento otimizado do classificador")
    parser.add_argument("--debias", choices=["none", "mask", "weight", "both"],
                        default="mask", help="Estratégia de debias")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_style", action="store_true",
                        help="Adicionar features de estilo")
    parser.add_argument("--style_weight", type=float, default=1.0,
                        help="Peso das features de estilo (1.0 = deixar XGBoost decidir, recomendado)")
    parser.add_argument("--n_estimators", type=int, default=300,
                        help="Número de árvores no XGBoost")
    parser.add_argument("--max_depth", type=int, default=7,
                        help="Profundidade máxima das árvores")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="Taxa de aprendizado")
    args = parser.parse_args()

    # Carregar dados
    texts, labels = load_dataset(DATASET_PATH)

    # Codificar labels
    le = LabelEncoder()
    y_all = le.fit_transform(labels)

    # Split estratificado
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_all
    )

    logger.info(f"Treino: {len(X_train_txt)} | Teste: {len(X_test_txt)}")

    # Carregar modelo de embeddings
    logger.info("Carregando modelo de embeddings...")
    sbert = SentenceTransformer(EMBED_MODEL_NAME)
    sbert.max_seq_length = 512

    # Aplicar debias se solicitado
    texts_train = list(X_train_txt)
    y_train_final = y_train.copy()
    sample_weights = np.ones(len(texts_train), dtype=np.float32)

    if args.debias in ("mask", "both"):
        logger.info("Aplicando mascaramento de entidades...")
        augmented_texts = []
        augmented_labels = []
        augmented_weights = []

        for i, text in enumerate(X_train_txt):
            entities = extract_entities_simple(text, _NLP)
            if entities:
                masked = mask_entities_in_text(text, entities)
                if masked != text:
                    augmented_texts.append(masked)
                    augmented_labels.append(y_train[i])
                    # Peso maior para amostras mascaradas
                    augmented_weights.append(0.7)

        if augmented_texts:
            texts_train.extend(augmented_texts)
            y_train_final = np.concatenate([y_train_final, augmented_labels])
            sample_weights = np.concatenate(
                [sample_weights, augmented_weights])
            logger.info(
                f"Adicionadas {len(augmented_texts)} amostras mascaradas")

    if args.debias in ("weight", "both"):
        logger.info("Aplicando balanceamento de classes...")
        class_weights = compute_class_weights(y_train_final)
        sample_weights = sample_weights * class_weights

    # Gerar embeddings
    logger.info("Gerando embeddings de treino...")
    X_train_emb = embed_texts_batch(sbert, texts_train, batch_size=32)

    logger.info("Gerando embeddings de teste...")
    X_test_emb = embed_texts_batch(sbert, X_test_txt, batch_size=32)

    # Adicionar features de estilo se solicitado
    if args.use_style:
        logger.info("Extraindo features de estilo...")
        X_train_style = np.vstack([extract_style_features(t)
                                  for t in texts_train])
        X_test_style = np.vstack([extract_style_features(t)
                                 for t in X_test_txt])

        # Normalizar features de estilo
        style_scaler = StandardScaler()
        X_train_style_scaled = style_scaler.fit_transform(X_train_style)
        X_test_style_scaled = style_scaler.transform(X_test_style)

        # Aplicar peso às features de estilo (se style_weight=1.0, não altera nada)
        # XGBoost baseado em árvores consegue lidar com escalas diferentes
        if args.style_weight != 1.0:
            X_train_style_final = X_train_style_scaled * args.style_weight
            X_test_style_final = X_test_style_scaled * args.style_weight
        else:
            # Com peso 1.0, deixar XGBoost decidir importância sem artificialmente reduzir
            X_train_style_final = X_train_style_scaled
            X_test_style_final = X_test_style_scaled

        # Concatenar
        X_train = np.hstack(
            [X_train_emb, X_train_style_final]).astype(np.float32)
        X_test = np.hstack(
            [X_test_emb, X_test_style_final]).astype(np.float32)

        logger.info(f"Dimensão final: {X_train.shape[1]} features")
    else:
        style_scaler = None
        X_train = X_train_emb
        X_test = X_test_emb

    # Treinar XGBoost otimizado
    logger.info("Treinando XGBoost...")
    base_clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        # Early stopping configurado no construtor (XGBoost 2.0+)
        early_stopping_rounds=20,
        callbacks=None
    )

    # NÃO aplicar calibração - ela estava comprimindo muito as probabilidades
    # Treinar diretamente com o XGBoost + Early Stopping
    logger.info("Ajustando modelo (sem calibração, com early stopping)...")
    clf = base_clf
    clf.fit(
        X_train,
        y_train_final,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],  # Monitora o conjunto de teste
        verbose=False
    )

    # Avaliar
    logger.info("Avaliando no conjunto de teste...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(y_test, y_pred,
          target_names=le.classes_, digits=4))

    # Calcular AUC-ROC
    try:
        auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"\nAUC-ROC: {auc:.4f}")
    except Exception:
        pass

    # Gerar gráfico de importância das features (Raio-X do modelo)
    try:
        from xgboost import plot_importance
        import matplotlib
        matplotlib.use('Agg')  # Backend sem GUI
        import matplotlib.pyplot as plt

        logger.info("Gerando gráfico de importância das features...")
        plt.figure(figsize=(10, 8))
        plot_importance(clf, max_num_features=20,
                        height=0.5, importance_type='weight')
        plt.title("O que a IA mais valorizou para detectar Fake News",
                  fontsize=14, weight='bold')
        plt.xlabel("Importância (frequência de uso nas decisões)")
        plt.tight_layout()
        importance_path = os.path.join(MODEL_DIR, "feature_importance.png")
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Gráfico de importância salvo: {importance_path}")
        print("   → Features f0-f383: embeddings BERT")
        if args.use_style:
            print("   → Features f384-f391: estilo (upper_ratio, allcaps, punct, exclam, quest, avg_word_len, ttr, lex_density)")
    except Exception as e:
        logger.warning(f"Não foi possível gerar gráfico de importância: {e}")

    # Salvar artefatos
    logger.info("Salvando modelos...")
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    if args.use_style and style_scaler is not None:
        joblib.dump(style_scaler, os.path.join(
            MODEL_DIR, "style_scaler.joblib"))

    # Salvar configuração
    config = {
        "embed_model": EMBED_MODEL_NAME,
        "normalize": True,
        "debias": args.debias,
        "style_features": args.use_style,
        "style_feature_count": 8,
        "style_weight": args.style_weight,
        "max_seq_length": 512,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "calibrated": False,
        "calibration_method": "none"
    }

    with open(os.path.join(MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Salvar textos de treino (sem máscaras) para FAISS
    with open(os.path.join(MODEL_DIR, "train_texts.json"), "w", encoding="utf-8") as f:
        json.dump(list(X_train_txt), f, ensure_ascii=False, indent=2)

    logger.info(f"Modelos salvos em {MODEL_DIR}")
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)


if __name__ == "__main__":
    main()
