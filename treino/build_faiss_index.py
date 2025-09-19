import os
import json
import math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_full_texts.json")
VSTORE_DIR = os.path.join(BASE_DIR, "vector_store")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = os.path.join(VSTORE_DIR, "faiss_index.bin")
META_PATH = os.path.join(VSTORE_DIR, "faiss_metadata.json")
CONFIG_PATH = os.path.join(VSTORE_DIR, "faiss_config.json")

os.makedirs(VSTORE_DIR, exist_ok=True)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filtrar registros válidos com texto
    valid = [x for x in data if isinstance(x, dict) and x.get("text") and isinstance(x.get("text"), str) and x.get("label")]
    return valid


def embed_texts(model, texts, batch_size=64, normalize=True):
    # Tenta usar normalização nativa; se não houver, normaliza na mão
    try:
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=True)
    except TypeError:
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
    return embs


def build_index(embeddings):
    d = embeddings.shape[1]
    # Cosine via inner product com vetores normalizados
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def main():
    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset não encontrado: {DATASET_PATH}")
        return

    data = load_dataset(DATASET_PATH)

    # Manter somente notícias verdadeiras como base histórica
    data_true = [x for x in data if str(x.get("label")).lower() == "true"]

    if not data_true:
        print("Nenhuma notícia 'true' encontrada com texto válido.")
        return

    texts = [x["text"].strip() for x in data_true if x["text"].strip()]

    if not texts:
        print("Nenhum texto válido após limpeza.")
        return

    print(f"Total de notícias 'true' para indexar: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(model, texts)

    index = build_index(embeddings)

    faiss.write_index(index, INDEX_PATH)

    # Salvar metadados simples: apenas label e talvez um trecho do texto
    metadata = [
        {
            "id": i,
            "label": "true",
            "preview": t[:300]
        }
        for i, t in enumerate(texts)
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    config = {
        "model_name": MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "normalize": True,
        "index_type": "IndexFlatIP",
        "count": int(index.ntotal)
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Index salvo em: {INDEX_PATH}")
    print(f"Metadados: {META_PATH}")
    print(f"Config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
