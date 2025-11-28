"""
Script para mesclar novas notícias TRUE ao dataset existente.
Balanceia o dataset mantendo proporção 50/50 entre fake e true.
"""

import json
import random


def load_json(file_path):
    """Carrega arquivo JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Salva arquivo JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("🔄 Mesclando datasets...\n")

    # Carregar dataset original
    original_file = "dataset_full_texts.json"
    print(f"📂 Carregando {original_file}...")
    original_data = load_json(original_file)

    # Contar fake e true originais
    fake_count = sum(1 for x in original_data if x.get("label") == "fake")
    true_count = sum(1 for x in original_data if x.get("label") == "true")

    print(
        f"  Original: {fake_count} fake + {true_count} true = {len(original_data)} total")

    # Carregar novas notícias TRUE coletadas
    new_file = "true_news_collected.json"
    try:
        print(f"\n📂 Carregando {new_file}...")
        new_true_news = load_json(new_file)

        # Limpar campos extras, manter só text e label
        cleaned_new = []
        for item in new_true_news:
            cleaned_new.append({
                "text": item["text"],
                "label": "true"
            })

        print(f"  Novas notícias TRUE: {len(cleaned_new)}")

    except FileNotFoundError:
        print(f"❌ Arquivo {new_file} não encontrado!")
        print("   Execute primeiro: python scrape_true_news.py")
        return

    # Calcular quantas adicionar para balancear
    # Queremos manter 50/50, então precisa ter fake_count notícias TRUE
    needed = fake_count - true_count

    if needed <= 0:
        print(f"\n✅ Dataset já está balanceado ou tem mais TRUE que FAKE")
        print(
            f"   Adicionando todas as {len(cleaned_new)} novas notícias TRUE...")
        to_add = cleaned_new
    else:
        print(f"\n⚖️ Precisamos de {needed} notícias TRUE para balancear")
        if len(cleaned_new) >= needed:
            print(
                f"   Selecionando {needed} das {len(cleaned_new)} coletadas...")
            to_add = random.sample(cleaned_new, needed)
        else:
            print(f"   Adicionando todas as {len(cleaned_new)} disponíveis...")
            to_add = cleaned_new

    # Mesclar
    merged_data = original_data + to_add

    # Embaralhar para misturar bem
    random.shuffle(merged_data)

    # Contar final
    final_fake = sum(1 for x in merged_data if x.get("label") == "fake")
    final_true = sum(1 for x in merged_data if x.get("label") == "true")

    print(f"\n📊 Dataset final:")
    print(f"  {final_fake} fake + {final_true} true = {len(merged_data)} total")
    print(
        f"  Proporção: {final_true/len(merged_data)*100:.1f}% TRUE / {final_fake/len(merged_data)*100:.1f}% FAKE")

    # Salvar backup do original
    backup_file = "dataset_full_texts_backup.json"
    print(f"\n💾 Salvando backup em {backup_file}...")
    save_json(original_data, backup_file)

    # Salvar novo dataset
    print(f"💾 Salvando dataset mesclado em {original_file}...")
    save_json(merged_data, original_file)

    print(f"\n✅ Concluído!")
    print(f"\n🎯 Próximo passo: Retreinar o modelo com 'python train_classifier.py --use_style --style_weight 0.05'")


if __name__ == "__main__":
    main()
