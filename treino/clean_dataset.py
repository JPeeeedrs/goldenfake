#!/usr/bin/env python3
"""
Script de Limpeza do Dataset GoldenFake
========================================

Remove:
1. Textos muito curtos (< 50 palavras)
2. Duplicatas exatas
3. Problemas de encoding (opcional)

Gera backup automático antes de limpar.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple


# Configurações
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset_full_texts.json"
BACKUP_DIR = BASE_DIR / "backups"
OUTPUT_PATH = BASE_DIR / "dataset_full_texts_cleaned.json"

MIN_WORDS = 50  # Mínimo de palavras para manter


class DatasetCleaner:
    """Limpador de dataset com logging detalhado."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.data = []
        self.removed = {
            'too_short': [],
            'duplicates': [],
            'encoding_issues': []
        }

    def load_data(self) -> None:
        """Carrega o dataset."""
        print(f"📂 Carregando dataset de {self.dataset_path}...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ {len(self.data)} amostras carregadas\n")

    def create_backup(self) -> Path:
        """Cria backup do dataset original."""
        BACKUP_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"dataset_backup_{timestamp}.json"

        print(f"💾 Criando backup em {backup_path}...")
        shutil.copy2(self.dataset_path, backup_path)
        print(f"✅ Backup criado\n")

        return backup_path

    def remove_short_texts(self) -> None:
        """Remove textos com menos de MIN_WORDS palavras."""
        print("="*70)
        print(f"1️⃣  REMOVENDO TEXTOS CURTOS (< {MIN_WORDS} palavras)")
        print("="*70)

        cleaned = []
        for i, item in enumerate(self.data):
            text = item.get('text', '')
            word_count = len(text.split())

            if word_count < MIN_WORDS:
                self.removed['too_short'].append({
                    'index': i,
                    'words': word_count,
                    'label': item.get('label'),
                    'text_preview': text[:100]
                })
            else:
                cleaned.append(item)

        removed_count = len(self.data) - len(cleaned)
        self.data = cleaned

        print(f"❌ Removidas: {removed_count} amostras")
        print(f"✅ Mantidas:  {len(self.data)} amostras")

        # Mostrar exemplos removidos
        if self.removed['too_short']:
            print(f"\n📋 Exemplos de textos removidos:")
            for item in self.removed['too_short'][:3]:
                print(
                    f"   - Índice {item['index']} ({item['words']} palavras, label: {item['label']})")
                print(f"     Texto: {item['text_preview']}...")
        print()

    def remove_duplicates(self) -> None:
        """Remove duplicatas exatas mantendo apenas a primeira ocorrência."""
        print("="*70)
        print("2️⃣  REMOVENDO DUPLICATAS EXATAS")
        print("="*70)

        seen_texts = {}
        cleaned = []

        for i, item in enumerate(self.data):
            text = item.get('text', '').strip().lower()
            text_hash = hash(text)

            if text_hash in seen_texts:
                # Duplicata encontrada
                first_index = seen_texts[text_hash]
                self.removed['duplicates'].append({
                    'kept_index': first_index,
                    'removed_index': i,
                    'label': item.get('label'),
                    'text_preview': text[:100]
                })
            else:
                # Primeira ocorrência
                seen_texts[text_hash] = i
                cleaned.append(item)

        removed_count = len(self.data) - len(cleaned)
        self.data = cleaned

        print(f"❌ Removidas: {removed_count} duplicatas")
        print(f"✅ Mantidas:  {len(self.data)} amostras únicas")

        # Mostrar duplicatas removidas
        if self.removed['duplicates']:
            print(f"\n📋 Duplicatas removidas:")
            for item in self.removed['duplicates']:
                print(
                    f"   - Mantido índice {item['kept_index']}, removido {item['removed_index']}")
                print(f"     Label: {item['label']}")
                print(f"     Texto: {item['text_preview']}...")
        print()

    def verify_balance(self) -> None:
        """Verifica balanceamento após limpeza."""
        print("="*70)
        print("3️⃣  VERIFICANDO BALANCEAMENTO")
        print("="*70)

        labels = [item.get('label', 'unknown') for item in self.data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(labels)
        print(f"📊 Distribuição após limpeza:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"   {label:10s}: {count:6d} ({pct:5.1f}%) {bar}")

        # Calcular razão
        if len(label_counts) >= 2:
            values = list(label_counts.values())
            max_val, min_val = max(values), min(values)
            ratio = max_val / min_val if min_val > 0 else float('inf')

            if ratio > 1.5:
                print(f"\n⚠️  Desbalanceamento: {ratio:.2f}x")
            else:
                print(f"\n✅ Dataset balanceado: {ratio:.2f}x")
        print()

    def verify_length_distribution(self) -> None:
        """Verifica distribuição de comprimentos após limpeza."""
        print("="*70)
        print("4️⃣  VERIFICANDO COMPRIMENTOS")
        print("="*70)

        import numpy as np

        by_label = defaultdict(list)
        for item in self.data:
            label = item.get('label', 'unknown')
            text = item.get('text', '')
            words = len(text.split())
            by_label[label].append(words)

        print(f"📏 Estatísticas de comprimento por label:")
        for label, lengths in sorted(by_label.items()):
            lengths = np.array(lengths)
            print(f"\n   {label.upper()}:")
            print(f"      Média:   {lengths.mean():8.1f} palavras")
            print(f"      Mediana: {np.median(lengths):8.1f} palavras")
            print(f"      Min:     {lengths.min():8d} palavras")
            print(f"      Max:     {lengths.max():8d} palavras")

        # Verificar diferença entre classes
        if len(by_label) >= 2:
            labels = list(by_label.keys())
            means = [np.mean(by_label[l]) for l in labels]
            max_mean, min_mean = max(means), min(means)
            ratio = max_mean / min_mean

            print(f"\n   Razão de tamanho entre classes: {ratio:.2f}x")
            if ratio > 2.0:
                print(f"   ⚠️  Ainda há diferença significativa!")
                print(
                    f"   💡 Considere coletar mais fake news longas ou truncar true news")
            else:
                print(f"   ✅ Diferença aceitável")
        print()

    def save_cleaned_dataset(self, output_path: Path) -> None:
        """Salva dataset limpo."""
        print("="*70)
        print("5️⃣  SALVANDO DATASET LIMPO")
        print("="*70)

        print(f"💾 Salvando em {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        print(f"✅ Dataset limpo salvo com {len(self.data)} amostras")

        # Salvar relatório de limpeza
        report_path = output_path.parent / "cleaning_report.json"
        report = {
            'original_size': len(self.data) + sum(len(v) for v in self.removed.values()),
            'cleaned_size': len(self.data),
            'removed_too_short': len(self.removed['too_short']),
            'removed_duplicates': len(self.removed['duplicates']),
            'removed_encoding': len(self.removed['encoding_issues']),
            'total_removed': sum(len(v) for v in self.removed.values())
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Relatório de limpeza salvo em {report_path}")
        print()

    def print_summary(self) -> None:
        """Imprime resumo final."""
        print("="*70)
        print("📊 RESUMO DA LIMPEZA")
        print("="*70)

        total_removed = sum(len(v) for v in self.removed.values())

        print(f"\n✂️  Total removido: {total_removed} amostras")
        print(f"   - Textos curtos: {len(self.removed['too_short'])}")
        print(f"   - Duplicatas:    {len(self.removed['duplicates'])}")
        print(f"   - Encoding:      {len(self.removed['encoding_issues'])}")

        print(f"\n✅ Dataset final: {len(self.data)} amostras limpas")

        reduction = (total_removed / (len(self.data) + total_removed)) * 100
        print(f"📉 Redução: {reduction:.1f}%")

        print(f"\n{'='*70}")
        print("🎉 LIMPEZA CONCLUÍDA COM SUCESSO!")
        print("="*70)

        print(f"\n💡 PRÓXIMOS PASSOS:")
        print(f"   1. Revisar o dataset limpo: {OUTPUT_PATH}")
        print(
            f"   2. Verificar o relatório: {OUTPUT_PATH.parent}/cleaning_report.json")
        print(f"   3. Se estiver OK, substituir o dataset original:")
        print(f"      mv {OUTPUT_PATH} {DATASET_PATH}")
        print(f"   4. Re-treinar o modelo com o dataset limpo")
        print()

    def run(self, create_backup: bool = True) -> None:
        """Executa limpeza completa."""
        print("\n" + "="*70)
        print("🧹 LIMPEZA DO DATASET - GoldenFake")
        print("="*70 + "\n")

        self.load_data()

        if create_backup:
            self.create_backup()

        self.remove_short_texts()
        self.remove_duplicates()
        self.verify_balance()
        self.verify_length_distribution()
        self.save_cleaned_dataset(OUTPUT_PATH)
        self.print_summary()


def main():
    """Função principal com confirmação do usuário."""
    import sys

    print("\n" + "="*70)
    print("🧹 SCRIPT DE LIMPEZA DO DATASET")
    print("="*70)
    print(f"\nDataset original: {DATASET_PATH}")
    print(f"Dataset limpo:    {OUTPUT_PATH}")
    print(f"\n⚠️  AÇÕES QUE SERÃO EXECUTADAS:")
    print(f"   1. Criar backup automático em backups/")
    print(f"   2. Remover textos com < {MIN_WORDS} palavras")
    print(f"   3. Remover duplicatas exatas")
    print(f"   4. Gerar relatório de limpeza")
    print(f"\n{'='*70}\n")

    # Confirmar execução
    response = input("🔹 Deseja continuar? (s/N): ").strip().lower()

    if response not in ['s', 'sim', 'y', 'yes']:
        print("\n❌ Operação cancelada pelo usuário.")
        sys.exit(0)

    # Executar limpeza
    cleaner = DatasetCleaner(DATASET_PATH)
    cleaner.run(create_backup=True)


if __name__ == "__main__":
    main()
