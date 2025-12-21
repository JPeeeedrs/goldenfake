#!/usr/bin/env python3
"""
Análise Completa de Qualidade do Dataset para Detecção de Fake News
====================================================================

Analisa problemas que podem prejudicar:
- SBERT (embeddings semânticos)
- XGBoost (classificador)
- FAISS (busca por similaridade)

Gera relatório detalhado com problemas e soluções.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI

# Configurações
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset_full_texts.json"
OUTPUT_DIR = BASE_DIR / "dataset_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Thresholds
MIN_WORDS = 50          # Mínimo de palavras para ser útil ao SBERT
MAX_WORDS = 10000       # Máximo razoável (acima disso pode ser spam)
MIN_CHARS = 200         # Mínimo de caracteres
MAX_REPETITION = 0.7    # Máximo de palavras repetidas (70%)
MIN_DIVERSITY = 0.2     # Mínimo de diversidade vocabular (Type-Token Ratio)


class DatasetAnalyzer:
    """Analisador completo de qualidade do dataset."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.data = []
        self.problems = defaultdict(list)
        self.stats = {}

    def load_data(self):
        """Carrega o dataset."""
        print(f"📂 Carregando dataset de {self.dataset_path}...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ {len(self.data)} amostras carregadas\n")

    def analyze_basic_structure(self):
        """1. Análise Básica de Estrutura"""
        print("="*70)
        print("1️⃣  ANÁLISE DE ESTRUTURA BÁSICA")
        print("="*70)

        # Verificar campos obrigatórios
        required_fields = ['text', 'label']
        optional_fields = ['title', 'source', 'url', 'date']

        missing_fields = defaultdict(int)
        for i, item in enumerate(self.data):
            for field in required_fields:
                if field not in item:
                    missing_fields[field] += 1
                    self.problems['missing_required'].append({
                        'index': i,
                        'field': field,
                        'item': item
                    })

        if missing_fields:
            print(f"❌ Campos obrigatórios faltando:")
            for field, count in missing_fields.items():
                print(
                    f"   - '{field}': {count} amostras ({count/len(self.data)*100:.1f}%)")
        else:
            print(f"✅ Todos os campos obrigatórios presentes")

        # Verificar campos opcionais
        optional_present = {field: sum(1 for item in self.data if field in item and item[field])
                            for field in optional_fields}
        print(f"\n📋 Campos opcionais presentes:")
        for field, count in optional_present.items():
            pct = count/len(self.data)*100
            symbol = "✅" if pct > 80 else "⚠️" if pct > 50 else "❌"
            print(f"   {symbol} '{field}': {count}/{len(self.data)} ({pct:.1f}%)")

        self.stats['fields'] = {
            'missing_required': dict(missing_fields),
            'optional_present': optional_present
        }
        print()

    def analyze_labels(self):
        """2. Análise de Labels (Balanceamento)"""
        print("="*70)
        print("2️⃣  ANÁLISE DE LABELS (BALANCEAMENTO)")
        print("="*70)

        labels = [item.get('label', 'missing') for item in self.data]
        label_counts = Counter(labels)

        print(f"📊 Distribuição de labels:")
        total = len(labels)
        for label, count in label_counts.most_common():
            pct = count/total*100
            bar = "█" * int(pct/2)
            print(f"   {label:10s}: {count:6d} ({pct:5.1f}%) {bar}")

        # Verificar balanceamento
        if len(label_counts) > 1:
            values = list(label_counts.values())
            max_val, min_val = max(values), min(values)
            imbalance_ratio = max_val / \
                min_val if min_val > 0 else float('inf')

            if imbalance_ratio > 1.5:
                print(f"\n⚠️  DESBALANCEAMENTO DETECTADO!")
                print(f"   Razão máx/mín: {imbalance_ratio:.2f}x")
                print(f"   Recomendação: Coletar mais amostras da classe minoritária")
                print(f"   ou aplicar técnicas de balanceamento (SMOTE, undersampling)")
                self.problems['imbalance'].append({
                    'ratio': imbalance_ratio,
                    'distribution': dict(label_counts)
                })
            else:
                print(f"\n✅ Dataset balanceado (razão {imbalance_ratio:.2f}x)")

        self.stats['labels'] = dict(label_counts)
        print()

    def analyze_text_length(self):
        """3. Análise de Comprimento de Texto"""
        print("="*70)
        print("3️⃣  ANÁLISE DE COMPRIMENTO DE TEXTO")
        print("="*70)

        lengths_words = []
        lengths_chars = []

        for i, item in enumerate(self.data):
            text = item.get('text', '')
            words = len(text.split())
            chars = len(text)

            lengths_words.append(words)
            lengths_chars.append(chars)

            # Detectar textos problemáticos
            if words < MIN_WORDS:
                self.problems['too_short'].append({
                    'index': i,
                    'words': words,
                    'label': item.get('label'),
                    'text_preview': text[:100]
                })
            if words > MAX_WORDS:
                self.problems['too_long'].append({
                    'index': i,
                    'words': words,
                    'label': item.get('label')
                })
            if chars < MIN_CHARS:
                self.problems['too_short_chars'].append({
                    'index': i,
                    'chars': chars
                })

        # Estatísticas
        lengths_words = np.array(lengths_words)
        print(f"📏 Estatísticas de palavras:")
        print(f"   Média:    {lengths_words.mean():8.1f} palavras")
        print(f"   Mediana:  {np.median(lengths_words):8.1f} palavras")
        print(f"   Desvio:   {lengths_words.std():8.1f} palavras")
        print(f"   Mínimo:   {lengths_words.min():8d} palavras")
        print(f"   Máximo:   {lengths_words.max():8d} palavras")

        # Percentis
        p25, p75 = np.percentile(lengths_words, [25, 75])
        print(f"   P25:      {p25:8.1f} palavras")
        print(f"   P75:      {p75:8.1f} palavras")

        # Problemas detectados
        print(f"\n⚠️  Textos problemáticos:")
        print(
            f"   Muito curtos (< {MIN_WORDS} palavras): {len(self.problems['too_short'])} ({len(self.problems['too_short'])/len(self.data)*100:.1f}%)")
        print(
            f"   Muito longos (> {MAX_WORDS} palavras): {len(self.problems['too_long'])} ({len(self.problems['too_long'])/len(self.data)*100:.1f}%)")

        if len(self.problems['too_short']) > 0:
            print(f"\n   💡 SOLUÇÃO para textos curtos:")
            print(f"      - Remover amostras com < {MIN_WORDS} palavras")
            print(
                f"      - SBERT precisa de contexto suficiente para gerar embeddings úteis")

        if len(self.problems['too_long']) > 0:
            print(f"\n   💡 SOLUÇÃO para textos longos:")
            print(f"      - Truncar em {MAX_WORDS} palavras ou usar chunking")
            print(f"      - Textos muito longos podem conter múltiplas notícias")

        self.stats['length'] = {
            'mean_words': float(lengths_words.mean()),
            'median_words': float(np.median(lengths_words)),
            'std_words': float(lengths_words.std()),
            'min_words': int(lengths_words.min()),
            'max_words': int(lengths_words.max()),
            'p25': float(p25),
            'p75': float(p75)
        }

        # Gerar gráfico de distribuição
        self._plot_length_distribution(lengths_words)
        print()

    def analyze_text_quality(self):
        """4. Análise de Qualidade do Texto"""
        print("="*70)
        print("4️⃣  ANÁLISE DE QUALIDADE DO TEXTO")
        print("="*70)

        for i, item in enumerate(self.data):
            text = item.get('text', '')
            if not text:
                continue

            words = text.split()
            if not words:
                continue

            # 1. Diversidade vocabular (Type-Token Ratio)
            unique_words = len(set(w.lower() for w in words))
            ttr = unique_words / len(words) if words else 0

            if ttr < MIN_DIVERSITY:
                self.problems['low_diversity'].append({
                    'index': i,
                    'ttr': ttr,
                    'label': item.get('label'),
                    'text_preview': text[:100]
                })

            # 2. Palavras muito repetidas (possível spam)
            word_counts = Counter(w.lower() for w in words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            repetition_ratio = most_common_count / len(words)

            if repetition_ratio > MAX_REPETITION:
                self.problems['high_repetition'].append({
                    'index': i,
                    'word': most_common_word,
                    'ratio': repetition_ratio,
                    'label': item.get('label')
                })

            # 3. Texto vazio ou só espaços
            if not text.strip():
                self.problems['empty_text'].append({'index': i})

            # 4. Caracteres especiais excessivos (possível encoding issue)
            special_chars = len(re.findall(
                r'[^\w\s\.,;:!?\-]', text, re.UNICODE))
            special_ratio = special_chars / len(text) if text else 0

            if special_ratio > 0.1:  # Mais de 10% de caracteres especiais
                self.problems['encoding_issue'].append({
                    'index': i,
                    'special_ratio': special_ratio,
                    'text_preview': text[:100]
                })

            # 5. URLs excessivas (possível spam)
            urls = re.findall(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            if len(urls) > 5:
                self.problems['too_many_urls'].append({
                    'index': i,
                    'url_count': len(urls)
                })

        # Relatório
        print(f"🔍 Problemas de qualidade detectados:")
        print(
            f"   Baixa diversidade vocabular: {len(self.problems['low_diversity'])} ({len(self.problems['low_diversity'])/len(self.data)*100:.1f}%)")
        print(
            f"   Alta repetição de palavras:  {len(self.problems['high_repetition'])} ({len(self.problems['high_repetition'])/len(self.data)*100:.1f}%)")
        print(
            f"   Textos vazios:                {len(self.problems['empty_text'])} ({len(self.problems['empty_text'])/len(self.data)*100:.1f}%)")
        print(
            f"   Problemas de encoding:        {len(self.problems['encoding_issue'])} ({len(self.problems['encoding_issue'])/len(self.data)*100:.1f}%)")
        print(
            f"   URLs excessivas:              {len(self.problems['too_many_urls'])} ({len(self.problems['too_many_urls'])/len(self.data)*100:.1f}%)")

        if any(len(self.problems[k]) > 0 for k in ['low_diversity', 'high_repetition', 'empty_text']):
            print(f"\n   💡 SOLUÇÕES:")
            if self.problems['low_diversity']:
                print(
                    f"      - Baixa diversidade: Pode ser spam ou texto gerado automaticamente")
                print(f"        → Revisar e remover amostras suspeitas")
            if self.problems['high_repetition']:
                print(
                    f"      - Alta repetição: Textos com palavras muito repetidas prejudicam SBERT")
                print(f"        → Filtrar palavras stopwords ou remover amostras")
            if self.problems['empty_text']:
                print(f"      - Textos vazios: REMOVER imediatamente do dataset")
        print()

    def analyze_duplicates(self):
        """5. Análise de Duplicatas"""
        print("="*70)
        print("5️⃣  ANÁLISE DE DUPLICATAS")
        print("="*70)

        # Duplicatas exatas (texto completo)
        text_hashes = defaultdict(list)
        for i, item in enumerate(self.data):
            text = item.get('text', '').strip().lower()
            text_hash = hash(text)
            text_hashes[text_hash].append(i)

        exact_duplicates = {k: v for k, v in text_hashes.items() if len(v) > 1}

        print(f"🔎 Duplicatas detectadas:")
        print(
            f"   Textos idênticos: {len(exact_duplicates)} grupos ({sum(len(v)-1 for v in exact_duplicates.values())} duplicatas)")

        if exact_duplicates:
            print(f"\n   ⚠️  PROBLEMA CRÍTICO: Duplicatas no dataset!")
            print(
                f"      - Prejudica validação cruzada (mesma amostra em treino e teste)")
            print(f"      - Infla artificialmente a acurácia do modelo")
            print(f"      - FAISS terá vetores idênticos (desperdício de espaço)")
            print(f"\n   💡 SOLUÇÃO: Remover TODAS as duplicatas, manter apenas 1 cópia")

            # Mostrar exemplo
            first_dup_group = list(exact_duplicates.values())[0]
            print(f"\n   Exemplo de duplicata (índices {first_dup_group}):")
            print(
                f"   Texto: {self.data[first_dup_group[0]].get('text', '')[:100]}...")

            self.problems['exact_duplicates'] = exact_duplicates
        else:
            print(f"   ✅ Nenhuma duplicata exata encontrada")

        # Duplicatas near-duplicate (primeiras 200 chars)
        prefix_hashes = defaultdict(list)
        for i, item in enumerate(self.data):
            text = item.get('text', '').strip().lower()
            prefix = text[:200]
            prefix_hash = hash(prefix)
            prefix_hashes[prefix_hash].append(i)

        near_duplicates = {k: v for k,
                           v in prefix_hashes.items() if len(v) > 1}
        near_dup_count = sum(len(v)-1 for v in near_duplicates.values())

        print(
            f"   Near-duplicates (início similar): {len(near_duplicates)} grupos ({near_dup_count} similares)")

        if near_dup_count > 0:
            print(f"\n   ⚠️  Near-duplicates podem ser:")
            print(f"      - Mesma notícia de fontes diferentes")
            print(f"      - Versões editadas da mesma notícia")
            print(f"   💡 RECOMENDAÇÃO: Revisar manualmente e remover similares demais")

        self.stats['duplicates'] = {
            'exact': len(exact_duplicates),
            'exact_count': sum(len(v)-1 for v in exact_duplicates.values()),
            'near': len(near_duplicates),
            'near_count': near_dup_count
        }
        print()

    def analyze_label_by_length(self):
        """6. Análise de Comprimento por Label"""
        print("="*70)
        print("6️⃣  ANÁLISE DE COMPRIMENTO POR LABEL")
        print("="*70)

        by_label = defaultdict(list)
        for item in self.data:
            label = item.get('label', 'unknown')
            text = item.get('text', '')
            words = len(text.split())
            by_label[label].append(words)

        print(f"📊 Estatísticas por label:")
        for label, lengths in by_label.items():
            lengths = np.array(lengths)
            print(f"\n   {label.upper()}:")
            print(f"      Média:   {lengths.mean():8.1f} palavras")
            print(f"      Mediana: {np.median(lengths):8.1f} palavras")
            print(f"      Min:     {lengths.min():8d} palavras")
            print(f"      Max:     {lengths.max():8d} palavras")

        # Verificar se há diferença significativa
        if len(by_label) >= 2:
            labels = list(by_label.keys())
            means = [np.mean(by_label[l]) for l in labels]
            max_mean, min_mean = max(means), min(means)

            if max_mean / min_mean > 2.0:
                print(f"\n   ⚠️  DIFERENÇA SIGNIFICATIVA entre labels!")
                print(f"      Razão máx/mín: {max_mean/min_mean:.2f}x")
                print(f"\n   💡 PROBLEMA para XGBoost:")
                print(f"      - Modelo pode aprender a classificar apenas pelo tamanho")
                print(
                    f"      - Fake news curtas ≠ sempre fake, notícias longas ≠ sempre true")
                print(f"\n   💡 SOLUÇÃO:")
                print(f"      - Normalizar comprimentos (truncar ou padding)")
                print(f"      - Usar features de estilo (que você já tem!)")
                print(f"      - Treinar com amostras de tamanhos variados por classe")

        self.stats['length_by_label'] = {
            label: {
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'std': float(np.std(lengths))
            }
            for label, lengths in by_label.items()
        }
        print()

    def analyze_vocabulary_overlap(self):
        """7. Análise de Sobreposição Vocabular entre Classes"""
        print("="*70)
        print("7️⃣  ANÁLISE DE VOCABULÁRIO (FAKE vs TRUE)")
        print("="*70)

        vocab_by_label = defaultdict(Counter)

        for item in self.data:
            label = item.get('label', 'unknown')
            text = item.get('text', '').lower()
            words = re.findall(r'\b\w+\b', text)
            vocab_by_label[label].update(words)

        if 'fake' in vocab_by_label and 'true' in vocab_by_label:
            fake_words = set(vocab_by_label['fake'].keys())
            true_words = set(vocab_by_label['true'].keys())

            overlap = fake_words & true_words
            fake_unique = fake_words - true_words
            true_unique = true_words - fake_words

            overlap_ratio = len(overlap) / len(fake_words | true_words)

            print(f"📚 Análise vocabular:")
            print(
                f"   Vocabulário FAKE:    {len(fake_words):,} palavras únicas")
            print(
                f"   Vocabulário TRUE:    {len(true_words):,} palavras únicas")
            print(
                f"   Sobreposição:        {len(overlap):,} palavras ({overlap_ratio*100:.1f}%)")
            print(f"   Exclusivas FAKE:     {len(fake_unique):,} palavras")
            print(f"   Exclusivas TRUE:     {len(true_unique):,} palavras")

            # Top palavras exclusivas de cada classe
            fake_top = vocab_by_label['fake'].most_common(20)
            true_top = vocab_by_label['true'].most_common(20)

            print(f"\n   🔴 Top 10 palavras mais comuns em FAKE:")
            for word, count in fake_top[:10]:
                if word not in true_words or vocab_by_label['fake'][word] > vocab_by_label['true'][word] * 2:
                    print(f"      - {word}: {count} ocorrências")

            print(f"\n   🟢 Top 10 palavras mais comuns em TRUE:")
            for word, count in true_top[:10]:
                if word not in fake_words or vocab_by_label['true'][word] > vocab_by_label['fake'][word] * 2:
                    print(f"      - {word}: {count} ocorrências")

            if overlap_ratio > 0.9:
                print(
                    f"\n   ⚠️  ALTA SOBREPOSIÇÃO ({overlap_ratio*100:.1f}%)!")
                print(f"      - Dificulta diferenciação entre fake e true")
                print(f"      - SBERT pode gerar embeddings muito similares")
                print(f"\n   💡 ISSO É ESPERADO!")
                print(f"      - Fake news IMITA notícias verdadeiras propositalmente")
                print(f"      - Por isso você usa features de ESTILO e HISTÓRICO")

        self.stats['vocabulary'] = {
            'fake_size': len(fake_words) if 'fake' in vocab_by_label else 0,
            'true_size': len(true_words) if 'true' in vocab_by_label else 0,
            'overlap': len(overlap) if 'fake' in vocab_by_label and 'true' in vocab_by_label else 0
        }
        print()

    def generate_report(self):
        """8. Gerar Relatório Final"""
        print("="*70)
        print("8️⃣  RELATÓRIO FINAL")
        print("="*70)

        total_problems = sum(len(v) for v in self.problems.values())

        print(f"\n📋 RESUMO EXECUTIVO:")
        print(f"   Total de amostras: {len(self.data)}")
        print(f"   Problemas detectados: {total_problems}")

        if total_problems == 0:
            print(f"\n   ✅ DATASET DE ALTA QUALIDADE!")
            print(f"      Nenhum problema crítico encontrado.")
        else:
            print(f"\n   ⚠️  PROBLEMAS ENCONTRADOS:")

            critical = ['empty_text', 'exact_duplicates', 'too_short']
            high = ['low_diversity', 'high_repetition', 'imbalance']
            medium = ['too_long', 'encoding_issue', 'too_many_urls']

            critical_count = sum(len(self.problems.get(k, []))
                                 for k in critical)
            high_count = sum(len(self.problems.get(k, [])) for k in high)
            medium_count = sum(len(self.problems.get(k, [])) for k in medium)

            if critical_count > 0:
                print(f"\n   🔴 CRÍTICO ({critical_count} amostras):")
                for key in critical:
                    if key in self.problems and self.problems[key]:
                        count = len(self.problems[key])
                        print(f"      - {key}: {count} amostras")

            if high_count > 0:
                print(f"\n   🟡 ALTO ({high_count} amostras):")
                for key in high:
                    if key in self.problems and self.problems[key]:
                        count = len(self.problems[key])
                        print(f"      - {key}: {count} amostras")

            if medium_count > 0:
                print(f"\n   🟢 MÉDIO ({medium_count} amostras):")
                for key in medium:
                    if key in self.problems and self.problems[key]:
                        count = len(self.problems[key])
                        print(f"      - {key}: {count} amostras")

        # Recomendações de ação
        print(f"\n\n{'='*70}")
        print("🎯 PLANO DE AÇÃO RECOMENDADO")
        print("="*70)

        actions = []

        if self.problems.get('empty_text'):
            actions.append({
                'priority': 1,
                'action': 'Remover textos vazios',
                'code': 'dataset = [d for d in dataset if d.get("text", "").strip()]'
            })

        if self.problems.get('exact_duplicates'):
            actions.append({
                'priority': 1,
                'action': 'Remover duplicatas exatas',
                'code': 'dataset = remove_duplicates(dataset)'
            })

        if self.problems.get('too_short'):
            actions.append({
                'priority': 2,
                'action': f'Remover textos com < {MIN_WORDS} palavras',
                'code': f'dataset = [d for d in dataset if len(d["text"].split()) >= {MIN_WORDS}]'
            })

        if self.problems.get('low_diversity'):
            actions.append({
                'priority': 2,
                'action': 'Revisar textos com baixa diversidade',
                'code': 'Verificar manualmente amostras suspeitas de spam'
            })

        if self.problems.get('imbalance'):
            actions.append({
                'priority': 2,
                'action': 'Balancear classes',
                'code': 'Usar SMOTE, undersampling ou coletar mais dados'
            })

        if self.problems.get('too_long'):
            actions.append({
                'priority': 3,
                'action': f'Truncar textos longos em {MAX_WORDS} palavras',
                'code': f'text = " ".join(text.split()[:{MAX_WORDS}])'
            })

        for i, action in enumerate(sorted(actions, key=lambda x: x['priority']), 1):
            print(
                f"\n{i}. [{['🔴', '🟡', '🟢'][action['priority']-1]}] {action['action']}")
            print(f"   Código: {action['code']}")

        # Salvar relatório JSON
        report_path = OUTPUT_DIR / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'problems': {k: len(v) for k, v in self.problems.items()},
                'total_samples': len(self.data),
                'actions': actions
            }, f, indent=2, ensure_ascii=False)

        print(f"\n\n📄 Relatório completo salvo em: {report_path}")
        print(f"📊 Gráficos salvos em: {OUTPUT_DIR}/")
        print()

    def _plot_length_distribution(self, lengths: np.ndarray):
        """Gera gráfico de distribuição de comprimentos."""
        plt.figure(figsize=(12, 6))

        # Histograma
        plt.subplot(1, 2, 1)
        plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(lengths.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Média: {lengths.mean():.0f}')
        plt.axvline(np.median(lengths), color='green', linestyle='--',
                    linewidth=2, label=f'Mediana: {np.median(lengths):.0f}')
        plt.xlabel('Número de Palavras')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Comprimento dos Textos')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(lengths, vert=True)
        plt.ylabel('Número de Palavras')
        plt.title('Boxplot de Comprimento')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'length_distribution.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   📊 Gráfico salvo: {OUTPUT_DIR}/length_distribution.png")

    def run_full_analysis(self):
        """Executa análise completa."""
        print("\n" + "="*70)
        print("🔬 ANÁLISE DE QUALIDADE DO DATASET - GoldenFake")
        print("="*70 + "\n")

        self.load_data()
        self.analyze_basic_structure()
        self.analyze_labels()
        self.analyze_text_length()
        self.analyze_text_quality()
        self.analyze_duplicates()
        self.analyze_label_by_length()
        self.analyze_vocabulary_overlap()
        self.generate_report()

        print("="*70)
        print("✅ ANÁLISE CONCLUÍDA!")
        print("="*70 + "\n")


def main():
    analyzer = DatasetAnalyzer(DATASET_PATH)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
