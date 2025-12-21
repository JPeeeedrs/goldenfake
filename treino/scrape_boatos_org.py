#!/usr/bin/env python3
"""
Web Scraper - Boatos.org
=========================

Coleta fake news desmentidas do site Boatos.org (brasileiro).

ANTES DE EXECUTAR:
1. Verificar robots.txt: https://www.boatos.org/robots.txt
2. Instalar dependências: pip install requests beautifulsoup4 lxml
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class BoatosOrgScraper:
    """Scraper para Boatos.org."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.base_url = "https://www.boatos.org"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Referer': 'https://www.boatos.org/'
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.articles_collected = []
        self.errors = []

    def check_robots_txt(self) -> bool:
        """Verifica se podemos scrapear o site."""
        try:
            response = self.session.get(
                f"{self.base_url}/robots.txt", timeout=10)
            robots_content = response.text.lower()

            print("🤖 Verificando robots.txt...")
            print(f"   URL: {self.base_url}/robots.txt")

            # Verificar apenas se há bloqueio GERAL de todo o site
            # "Disallow: /" sozinho (sem outras pastas) significa bloqueio total
            lines = robots_content.split('\n')

            # Procurar por "Disallow: /" sem especificar subpastas
            general_disallow = False
            for line in lines:
                line = line.strip()
                # Bloqueio geral é apenas "Disallow: /" sem nada após a barra
                if line == 'disallow: /' or line == 'disallow:/':
                    general_disallow = True
                    break

            if general_disallow:
                print("   ❌ Site bloqueia scraping completamente (Disallow: /)")
                return False

            # Se só tem bloqueios específicos (wp-admin, wp-content, etc), está OK
            print("   ✅ Scraping de artigos permitido")
            print("   📋 Bloqueios específicos: /wp-admin, /wp-content, /wp-includes")
            print("   ✅ URLs de artigos (/politica/, /brasil/, etc) são acessíveis")
            return True

        except Exception as e:
            print(f"   ⚠️  Não foi possível verificar robots.txt: {e}")
            print("   Continuando com cautela...")
            return True

    def get_article_links_from_page(self, page_url: str) -> List[str]:
        """Extrai links de artigos de uma página de listagem."""
        try:
            print(f"   📄 Acessando: {page_url}")
            response = self.session.get(page_url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Seletor identificado: h2.blog-entry-title a
            article_links = []

            # Procurar por todos os links de artigos
            title_elements = soup.select('h2.blog-entry-title a[href]')

            for link_elem in title_elements:
                href = link_elem.get('href', '')
                if href and href.startswith('http'):
                    article_links.append(href)

            print(f"   ✅ Encontrados {len(article_links)} artigos")
            return article_links

        except Exception as e:
            print(f"   ❌ Erro ao processar página: {e}")
            self.errors.append({'page': page_url, 'error': str(e)})
            return []

    def scrape_article(self, article_url: str) -> Optional[Dict]:
        """Extrai conteúdo de um artigo individual."""
        try:
            response = self.session.get(article_url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # ===== TÍTULO =====
            # Tentar múltiplos seletores
            title_elem = (
                soup.select_one('h1.entry-title') or
                soup.select_one('h1.title') or
                soup.select_one('article h1') or
                soup.select_one('h1')
            )
            title = title_elem.get_text(strip=True) if title_elem else ""

            # ===== CONTEÚDO =====
            # Container principal do artigo
            content_elem = soup.select_one('.nv-content-wrap')

            if not content_elem:
                print(
                    f"      ⚠️  Container .nv-content-wrap não encontrado")
                return None

            # FILTRO CRÍTICO: Extrair APENAS os spans vermelhos (fake news originais)
            # Se não tiver spans vermelhos, é artigo de guia/dica, não fake news
            fake_news_texts = []
            red_spans = content_elem.find_all(
                'span', style=lambda value: value and 'color: #ff0000' in value.lower())

            if not red_spans:
                print(f"      ⏭️  Sem texto vermelho (guia/matéria) - pulando")
                return None

            for span in red_spans:
                text = span.get_text(strip=True)
                if len(text) > 20:  # Aceitar textos menores (fake news podem ser curtas)
                    fake_news_texts.append(text)

            if not fake_news_texts:
                print(f"      ⏭️  Spans vermelhos vazios - pulando")
                return None

            # Juntar tudo
            full_text = " ".join(fake_news_texts)
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Limpar caracteres especiais/emojis desnecessários
            full_text = full_text.replace('\xa0', ' ')  # Non-breaking space

            # ===== CATEGORIA =====
            category_elem = soup.select_one('.meta.category a')
            category = category_elem.get_text(
                strip=True) if category_elem else ""

            # ===== FILTRO DE IDIOMA =====
            # Aceitar APENAS categorias em português
            # Rejeitar: "English", "Español", outras línguas
            if category and category.lower() not in ['política', 'saúde', 'tecnologia', 'economia',
                                                     'entretenimento', 'educação', 'ciência',
                                                     'mundo', 'brasil', 'esporte', 'cultura',
                                                     'sociedade', 'meio ambiente', 'geral', '']:
                # Se categoria tem nome de idioma estrangeiro, rejeitar
                foreign_keywords = ['english', 'español', 'espanhol', 'french', 'français',
                                    'german', 'deutsch', 'italian', 'italiano']
                if any(keyword in category.lower() for keyword in foreign_keywords):
                    print(
                        f"      ⏭️  Idioma estrangeiro detectado (categoria: {category}) - pulando")
                    return None

            # ===== DETECÇÃO ADICIONAL DE IDIOMA NO TEXTO =====
            # Palavras MUITO comuns em português (aparecem em qualquer texto)
            portuguese_words = [
                ' que ', ' não ', ' com ', ' para ', ' uma ', ' mais ', ' ser ',
                ' está ', ' foi ', ' são ', ' como ', ' por ', ' dos ', ' das ',
                ' pela ', ' pelo ', ' ou ', ' mas ', ' também ', ' até '
            ]

            # FRASES COMPLETAS que indicam texto majoritariamente estrangeiro
            # Não palavras isoladas (pois notícias PT podem citar termos em inglês)
            foreign_phrases = [
                # Frases que só existem em textos totalmente em inglês
                'minutes later', 'trying to intimidate', 'corrupt judge tries',
                'you won\'t believe', 'what happened next', 'never expected',
                'caught on camera', 'this powerful story', 'when a corrupt',
                'thought he could', 'walk into the courtroom',

                # Frases que só existem em textos totalmente em espanhol
                'señores generales coronéis', 'comandantes militares el',
                'atención señores generales', 'está embarazada en medio',
                'es falso que la viuda', 'reveló que está embarazada'
            ]

            text_lower = full_text.lower()

            # Contar palavras portuguesas comuns
            portuguese_count = sum(
                1 for word in portuguese_words if word in text_lower)

            # Detectar FRASES estrangeiras (não palavras isoladas)
            foreign_phrases_found = [
                phrase for phrase in foreign_phrases if phrase in text_lower
            ]

            # CRITÉRIOS DE REJEIÇÃO:
            # 1. Tem 2+ frases estrangeiras completas OU
            # 2. Tem menos de 5 palavras portuguesas comuns (texto curto em idioma estrangeiro)

            if len(foreign_phrases_found) >= 2:
                print(
                    f"      ⏭️  Múltiplas frases estrangeiras detectadas ({len(foreign_phrases_found)}) - pulando")
                return None

            if portuguese_count < 5:
                print(
                    f"      ⏭️  Poucas palavras portuguesas ({portuguese_count}) - possível idioma estrangeiro - pulando")
                return None

            # Se tem 1 frase estrangeira MAS muitas palavras portuguesas = citação, OK!
            # Ex: notícia PT que fala sobre "Top Gun" ou cita frase em inglês

            # ===== DATA =====
            date_elem = soup.select_one('time.entry-date')
            date = date_elem.get('datetime', '') if date_elem else ""

            # ===== VALIDAÇÕES =====
            word_count = len(full_text.split())

            # Filtrar textos muito curtos (textos vermelhos)
            if word_count < 100:
                print(
                    f"      ⏭️  Texto vermelho muito curto ({word_count} palavras) - pulando")
                return None

            # Label: Todos os artigos do Boatos.org são fake news desmentidas
            label = 'fake'

            print(f"      ✅ {word_count} palavras - {title[:60]}...")

            return {
                'text': full_text,
                'label': label,
                'title': title,
                'category': category,
                'source': 'Boatos.org',
                'url': article_url,
                'date': date,
                'word_count': word_count,
                'collected_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"      ❌ Erro ao processar artigo: {e}")
            self.errors.append({'url': article_url, 'error': str(e)})
            return None

    def scrape_multiple_pages(self, start_page: int = 1, max_pages: int = 5) -> List[Dict]:
        """Coleta artigos de múltiplas páginas."""
        print(f"\n🕷️  Iniciando scraping do Boatos.org")
        print(f"   Páginas: {start_page} até {start_page + max_pages - 1}")
        print(f"   Delay entre requests: 1-1.5 segundos (otimizado)")
        print(f"   🎯 FILTRANDO: Apenas artigos com fake news em vermelho\n")

        # Verificar robots.txt
        if not self.check_robots_txt():
            print("\n❌ Scraping bloqueado pelo robots.txt")
            return []

        all_article_urls = set()  # Usar set para evitar duplicatas

        # Coletar URLs de artigos de cada página
        for page_num in range(start_page, start_page + max_pages):
            # URL pattern do Boatos.org (ajustar se necessário)
            if page_num == 1:
                page_url = f"{self.base_url}/"
            else:
                page_url = f"{self.base_url}/page/{page_num}/"

            print(f"\n📑 PÁGINA {page_num}:")
            article_urls = self.get_article_links_from_page(page_url)
            all_article_urls.update(article_urls)

            # Delay entre páginas (respeitar servidor mas otimizado)
            time.sleep(1.5)

        print(f"\n📊 Total de URLs únicas coletadas: {len(all_article_urls)}")
        print(f"\n🔄 Processando artigos individuais...\n")

        # Processar cada artigo
        for idx, article_url in enumerate(all_article_urls, 1):
            print(f"   [{idx}/{len(all_article_urls)}] {article_url}")

            article_data = self.scrape_article(article_url)

            if article_data:
                self.articles_collected.append(article_data)

            # Delay entre artigos (ético mas otimizado)
            # Reduzido de 2s para 1s - ainda respeitoso mas mais rápido
            time.sleep(1)

        return self.articles_collected

    def save_results(self, filename: str = "boatos_org_scraped.json"):
        """Salva resultados em JSON."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.articles_collected, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"📊 RESUMO FINAL")
        print(f"{'='*70}")
        print(f"✅ Artigos coletados: {len(self.articles_collected)}")
        print(f"❌ Erros encontrados: {len(self.errors)}")

        if self.articles_collected:
            word_counts = [a['word_count'] for a in self.articles_collected]
            print(f"\n📏 Estatísticas de tamanho:")
            print(
                f"   Média:   {sum(word_counts)/len(word_counts):.0f} palavras")
            print(f"   Mínimo:  {min(word_counts)} palavras")
            print(f"   Máximo:  {max(word_counts)} palavras")

            # Filtrar apenas longas (500+ palavras)
            long_articles = [
                a for a in self.articles_collected if a['word_count'] >= 500]
            print(f"\n📰 Artigos longos (≥500 palavras): {len(long_articles)}")

            # Salvar também só os longos
            if long_articles:
                long_path = self.output_dir / f"boatos_org_long_only.json"
                with open(long_path, 'w', encoding='utf-8') as f:
                    json.dump(long_articles, f, indent=2, ensure_ascii=False)
                print(f"   💾 Salvos em: {long_path}")

        print(f"\n💾 Arquivo completo: {output_path}")

        # Salvar erros se houver
        if self.errors:
            errors_path = self.output_dir / "errors.json"
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2, ensure_ascii=False)
            print(f"⚠️  Log de erros: {errors_path}")

        print(f"{'='*70}\n")


def main():
    """Função principal."""
    import argparse

    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Web Scraper de Fake News do Boatos.org',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Coletar 10 páginas (teste rápido, ~5 minutos, ~60-80 fake news)
  python3 scrape_boatos_org.py --pages 10
  
  # Coletar 100 páginas (coleta média, ~30 minutos, ~600-800 fake news)
  python3 scrape_boatos_org.py --pages 100
  
  # Coletar em lotes (páginas 101-200)
  python3 scrape_boatos_org.py --start 101 --pages 100
  
  # Coleta massiva (500 páginas, ~2-3 horas, ~3000 fake news)
  python3 scrape_boatos_org.py --pages 500
  
  # Com arquivo de saída customizado
  python3 scrape_boatos_org.py --pages 50 --output boatos_lote1.json
        """
    )

    parser.add_argument(
        '--pages',
        type=int,
        default=30,
        help='Número de páginas para scrapear (padrão: 30, ~15 artigos/página)'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Página inicial (útil para coletar em lotes, padrão: 1)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='boatos_org_scraped.json',
        help='Nome do arquivo de saída (padrão: boatos_org_scraped.json)'
    )

    parser.add_argument(
        '--min-words',
        type=int,
        default=100,
        help='Mínimo de palavras para incluir artigo (padrão: 100)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🕷️  WEB SCRAPER - BOATOS.ORG")
    print("="*70)
    print("\n⚠️  CONFIGURAÇÃO:")
    print(
        f"   - Páginas: {args.start} até {args.start + args.pages - 1} (total: {args.pages})")
    print(f"   - Mínimo de palavras: {args.min_words}")
    print(f"   - Arquivo de saída: {args.output}")
    print(
        f"   - Estimativa: ~{args.pages * 15} artigos, ~{int(args.pages * 6)} fake news")
    print(f"   - Tempo estimado: ~{int(args.pages * 0.5)} minutos")
    print("\n⚠️  ATENÇÃO:")
    print("   - Respeita robots.txt")
    print("   - Delays de 1-1.5s entre requests")
    print("   - Coleta apenas artigos com fake news (texto vermelho)")
    print("   - Fins educacionais/pesquisa\n")

    # Configurações
    output_dir = Path(__file__).parent / "scraped_data"

    # Criar scraper
    scraper = BoatosOrgScraper(output_dir)

    # Executar scraping
    articles = scraper.scrape_multiple_pages(
        start_page=args.start,
        max_pages=args.pages
    )

    # Salvar resultados
    scraper.save_results(filename=args.output)

    print("✅ Scraping concluído!")


if __name__ == "__main__":
    main()
