import faiss
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time

# --- Configuração de Entrada (Seus arquivos) ---
EMBEDDINGS_DIR = Path(__file__).parent.resolve() / "embeddings"
METADATA_FILE = EMBEDDINGS_DIR / "metadata_final.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings_final.npz"

# --- Configuração de Saída (Nomes de arquivo que você quer) ---
OUTPUT_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index.bin"
OUTPUT_METADATA_FILE = EMBEDDINGS_DIR / "faiss_metadata.json"
OUTPUT_CONFIG_FILE = EMBEDDINGS_DIR / "faiss_config.json"
OUTPUT_PROGRESS_FILE = EMBEDDINGS_DIR / "faiss_build.progress.json"

# Modelo usado para gerar os embeddings (para o config.json)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# -----------------------------------------------------------------

# --- ETAPA 1: Carregar Metadados de Entrada ---
print(f"Carregando metadados de {METADATA_FILE}...")
with open(METADATA_FILE, 'r') as f:
    # 'metadata' é um dicionário: {"id_artigo": {...}, ...}
    metadata = json.load(f)
print(f"Metadados para {len(metadata)} IDs carregados.")

# --- ETAPA 2: Carregar Embeddings ---
print(f"Carregando arquivo de embeddings {EMBEDDINGS_FILE}...")
start_time = time.time()
npz_file = np.load(EMBEDDINGS_FILE)

# index_map será o 'faiss_metadata.json' de saída (uma lista)
index_map = [] 
embedding_list = []
d = 0 # Dimensão dos embeddings

print("Lendo embeddings do arquivo NPZ...")

# Itera sobre os metadados para garantir a ordem correta
for article_id in tqdm(metadata.keys(), desc="Lendo arrays do NPZ"):
    if article_id not in npz_file:
        print(f"Aviso: Artigo ID {article_id} dos metadados não encontrado no NPZ.")
        continue

    # Carrega o array de embeddings para este artigo
    embs_array = npz_file[article_id]
    
    if d == 0:
        # Detecta a dimensão do embedding no primeiro item
        d = embs_array.shape[1]
        print(f"Dimensão do embedding detectada: {d}")

    # Adiciona os embeddings à nossa lista principal
    embedding_list.append(embs_array.astype('float32'))
    
    # Cria o mapa de volta para o artigo/chunk
    # Este será o nosso 'faiss_metadata.json'
    num_chunks = embs_array.shape[0]
    for i in range(num_chunks):
        index_map.append({
            "id": article_id,
            "chunk_index": i
        })

end_time = time.time()
print(f"Todos os {len(embedding_list)} arrays de artigos carregados em {end_time - start_time:.2f} segundos.")

# --- ETAPA 3: Construir Matriz e Índice FAISS ---
print("Empilhando todos os embeddings em uma única matriz (pode levar um momento)...")
all_embeddings = np.vstack(embedding_list)

print(f"Matriz final criada com shape: {all_embeddings.shape}")
print(f"Total de vetores: {all_embeddings.shape[0]:,}")
print(f"Dimensão do vetor: {all_embeddings.shape[1]}")

if all_embeddings.shape[0] != len(index_map):
    raise ValueError("Erro: O número de embeddings não bate com o mapa do índice!")

print("Construindo o índice FAISS (IndexFlatIP)...")
index = faiss.IndexFlatIP(d)

print("Adicionando vetores ao índice...")
index.add(all_embeddings)

total_vectors = index.ntotal
print(f"Total de vetores no índice: {total_vectors}")

# --- ETAPA 4: Salvar (SÓ UMA VEZ, com seus nomes de arquivo) ---

# 1. Salvar o índice binário
print(f"Salvando índice em {OUTPUT_INDEX_FILE}...")
faiss.write_index(index, str(OUTPUT_INDEX_FILE))

# 2. Salvar o mapa (o index_map que criamos)
print(f"Salvando mapa do índice em {OUTPUT_METADATA_FILE}...")
with open(OUTPUT_METADATA_FILE, 'w') as f:
    json.dump(index_map, f) # Salva sem indentação para ser mais rápido/compacto

# 3. Salvar o arquivo de configuração
print(f"Salvando configuração em {OUTPUT_CONFIG_FILE}...")
config_payload = {
    "model_name": MODEL_NAME,
    "dim": d,
    "index_type": "IndexFlatIP",
    "count": total_vectors,
    "normalize": True,
}
with open(OUTPUT_CONFIG_FILE, 'w') as f:
    json.dump(config_payload, f, indent=2)

# 4. Salvar o arquivo de progresso (completo)
print(f"Salvando progresso em {OUTPUT_PROGRESS_FILE}...")
progress_payload = {
    "processed_files": list(metadata.keys()), # Lista de todos os IDs de artigos
    "total_vectors": total_vectors,
}
with open(OUTPUT_PROGRESS_FILE, 'w') as f:
    json.dump(progress_payload, f, indent=2)

print("✓ Processo concluído! Todos os 4 arquivos foram criados.")