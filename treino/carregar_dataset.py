import os
import json
from collections import Counter

# Caminhos base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "full_texts")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_full_texts.json")


def carregar_textos(pasta, label):
    textos = []
    if not os.path.isdir(pasta):
        print(f"Aviso: pasta não encontrada: {pasta}")
        return textos
    for nome_arquivo in sorted(os.listdir(pasta)):
        caminho = os.path.join(pasta, nome_arquivo)
        if not os.path.isfile(caminho):
            continue
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            texto = f.read().strip()
            textos.append({"text": texto, "label": label})
    return textos

# Carregar fake e true
fake_path = os.path.join(DATASET_DIR, "fake")
true_path = os.path.join(DATASET_DIR, "true")

dados_fake = carregar_textos(fake_path, "fake")
dados_true = carregar_textos(true_path, "true")

# Unir e imprimir algumas estatísticas
dados_unidos = dados_fake + dados_true
contagem = Counter(item["label"] for item in dados_unidos)

print("Total de notícias:", len(dados_unidos))
print("Contagem por label:", dict(contagem))
print("Exemplo:", dados_unidos[0] if dados_unidos else None)

# Salvar em JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dados_unidos, f, ensure_ascii=False, indent=2)

print(f"\nArquivo salvo: {OUTPUT_FILE}")
