#!/usr/bin/env python3
"""
Remove notícias em idiomas estrangeiros de arquivos coletados.
Detecta inglês, espanhol e outros idiomas automaticamente.
"""

import json
import sys
from pathlib import Path


def detect_language(text: str) -> tuple[str, float]:
    """
    Detecta o idioma do texto.
    Retorna (idioma, confiança) onde idioma é 'pt', 'en', 'es' ou 'other'
    """
    text_lower = text.lower()

    # Palavras exclusivas de português
    portuguese_indicators = [
        ' que ', ' não ', ' com ', ' para ', ' uma ', ' mais ', ' ser ',
        ' está ', ' foi ', ' são ', ' como ', ' por ', ' dos ', ' das ',
        ' pela ', ' pelo ', ' ou ', ' mas ', ' também ', ' até ', ' onde ',
        ' quando ', ' qual ', ' seu ', ' sua ', ' seus ', ' suas '
    ]

    # Palavras exclusivas de inglês
    english_indicators = [
        ' the ', ' and ', ' you ', ' are ', ' can ', ' this ', ' that ',
        ' what ', ' will ', ' with ', ' from ', ' have ', ' they ',
        ' were ', ' been ', ' was ', ' his ', ' her ', ' their ',
        'minutes later', 'trying to', 'corrupt judge', 'top gun',
        ' would ', ' could ', ' should ', ' which ', ' these ', ' those '
    ]

    # Palavras exclusivas de espanhol
    spanish_indicators = [
        ' de la ', ' en el ', ' es el ', ' por el ', ' con el ',
        ' está embarazada ', ' es falso ', ' señores generales ',
        ' comandantes militares ', ' atención señores ', ' el general ',
        ' la vacuna ', ' los Estados Unidos '
    ]

    # Contar indicadores
    pt_count = sum(1 for ind in portuguese_indicators if ind in text_lower)
    en_count = sum(1 for ind in english_indicators if ind in text_lower)
    es_count = sum(1 for ind in spanish_indicators if ind in text_lower)

    # Determinar idioma
    max_count = max(pt_count, en_count, es_count)

    if max_count == 0:
        return 'unknown', 0.0

    if pt_count == max_count and pt_count >= 3:
        confidence = pt_count / (pt_count + en_count + es_count)
        return 'pt', confidence
    elif en_count == max_count:
        confidence = en_count / (pt_count + en_count + es_count)
        return 'en', confidence
    elif es_count == max_count:
        confidence = es_count / (pt_count + en_count + es_count)
        return 'es', confidence

    return 'unknown', 0.0


def remove_foreign_articles(input_file: Path, output_file: Path = None,
                            backup: bool = True) -> dict:
    """
    Remove artigos em idiomas estrangeiros.

    Args:
        input_file: Arquivo JSON de entrada
        output_file: Arquivo de saída (padrão: sobrescrever o original)
        backup: Criar backup antes de modificar (padrão: True)

    Returns:
        Relatório com estatísticas da limpeza
    """
    print("\n" + "="*70)
    print("🧹 REMOÇÃO DE NOTÍCIAS EM IDIOMAS ESTRANGEIROS")
    print("="*70)

    # Carregar dados
    print(f"\n📂 Carregando {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ {len(data)} artigos carregados")

    # Criar backup
    if backup:
        backup_file = input_file.parent / \
            f"{input_file.stem}_backup{input_file.suffix}"
        print(f"\n💾 Criando backup em {backup_file}...")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ Backup criado")

    # Analisar idiomas
    print("\n🔍 Analisando idiomas...")
    results = []
    for i, article in enumerate(data):
        text = article.get('text', '')
        lang, confidence = detect_language(text)
        results.append({
            'index': i,
            'language': lang,
            'confidence': confidence,
            'article': article
        })

    # Separar por idioma
    portuguese = [r for r in results if r['language'] == 'pt']
    english = [r for r in results if r['language'] == 'en']
    spanish = [r for r in results if r['language'] == 'es']
    unknown = [r for r in results if r['language'] == 'unknown']

    print(f"\n📊 Distribuição de idiomas:")
    print(
        f"   🟢 Português: {len(portuguese)} ({len(portuguese)/len(data)*100:.1f}%)")
    print(
        f"   🔴 Inglês:    {len(english)} ({len(english)/len(data)*100:.1f}%)")
    print(
        f"   🟡 Espanhol:  {len(spanish)} ({len(spanish)/len(data)*100:.1f}%)")
    print(
        f"   ⚪ Incerto:   {len(unknown)} ({len(unknown)/len(data)*100:.1f}%)")

    # Mostrar exemplos de artigos estrangeiros
    foreign = english + spanish
    if foreign:
        print(f"\n⚠️  Artigos estrangeiros detectados ({len(foreign)}):")
        for r in foreign[:10]:  # Mostrar até 10
            article = r['article']
            preview = article['text'][:100]
            lang_name = {'en': 'Inglês', 'es': 'Espanhol'}[r['language']]
            print(
                f"\n   [{r['index']}] {lang_name} (confiança: {r['confidence']:.1%})")
            print(f"   Título: {article.get('title', 'N/A')[:80]}")
            print(f"   Texto:  {preview}...")

    # Confirmar remoção
    if foreign:
        print(f"\n⚠️  {len(foreign)} artigos serão REMOVIDOS")
        response = input("\n🔹 Confirmar remoção? (s/N): ").lower().strip()

        if response != 's':
            print("\n❌ Operação cancelada")
            return {
                'total': len(data),
                'removed': 0,
                'kept': len(data),
                'cancelled': True
            }

    # Remover estrangeiros
    cleaned_data = [r['article'] for r in portuguese]

    # Se há artigos incertos, perguntar o que fazer
    if unknown:
        print(f"\n⚠️  {len(unknown)} artigos com idioma INCERTO")
        print("   (poucas palavras de qualquer idioma detectadas)")
        response = input("   Manter artigos incertos? (S/n): ").lower().strip()

        if response != 'n':
            cleaned_data.extend([r['article'] for r in unknown])
            print(f"   ✅ {len(unknown)} artigos incertos mantidos")
        else:
            print(f"   ❌ {len(unknown)} artigos incertos removidos")

    # Salvar resultado
    if output_file is None:
        output_file = input_file

    print(f"\n💾 Salvando em {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    # Relatório final
    removed = len(data) - len(cleaned_data)

    print("\n" + "="*70)
    print("✅ LIMPEZA CONCLUÍDA")
    print("="*70)
    print(f"\n📊 Resultado:")
    print(f"   Total original:  {len(data)}")
    print(f"   Removidos:       {removed} ({removed/len(data)*100:.1f}%)")
    print(
        f"   Mantidos:        {len(cleaned_data)} ({len(cleaned_data)/len(data)*100:.1f}%)")

    if backup:
        print(f"\n💾 Backup salvo em: {backup_file}")

    return {
        'total': len(data),
        'removed': removed,
        'kept': len(cleaned_data),
        'languages': {
            'portuguese': len(portuguese),
            'english': len(english),
            'spanish': len(spanish),
            'unknown': len(unknown)
        }
    }


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print(
            "\n💡 Uso: python remove_foreign_news.py <arquivo.json> [--no-backup]")
        print("\nExemplo:")
        print("  python remove_foreign_news.py scraped_data/boatos_org_scraped.json")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"\n❌ Erro: Arquivo não encontrado: {input_file}")
        sys.exit(1)

    backup = '--no-backup' not in sys.argv

    report = remove_foreign_articles(input_file, backup=backup)

    if not report.get('cancelled'):
        print("\n🎉 Arquivo limpo com sucesso!")
        print("\n💡 Próximo passo:")
        print("   Adicionar ao dataset com merge_dataset.py")


if __name__ == "__main__":
    main()
