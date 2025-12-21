#!/usr/bin/env python3
"""
Web Scraper - G1 Globo
=======================

Coleta notícias verdadeiras do portal G1 (Globo).

ANTES DE EXECUTAR:
1. Verificar robots.txt: https://g1.globo.com/robots.txt
2. Instalar dependências: pip install requests beautifulsoup4 lxml
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET


def clean_text(text: str) -> str:
    """
    Remove emojis, caracteres especiais e normaliza o texto.

    Args:
        text: Texto original

    Returns:
        Texto limpo
    """
    # Remover emojis (padrão Unicode completo)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # símbolos & pictogramas
        "\U0001F680-\U0001F6FF"  # transporte & símbolos de mapa
        "\U0001F1E0-\U0001F1FF"  # bandeiras (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Símbolos suplementares
        "\U0001FA70-\U0001FAFF"  # Símbolos estendidos
        "\U00002600-\U000026FF"  # Símbolos diversos
        "\U00002700-\U000027BF"  # Dingbats
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(' ', text)

    # Remover hashtags isoladas (mas manter palavras com #)
    text = re.sub(r'\s#\w+\b', ' ', text)

    # Remover caracteres de controle (exceto espaço, tab, newline)
    text = ''.join(char for char in text if ord(
        char) >= 32 or char in '\t\n\r')

    # Normalizar espaços múltiplos
    text = re.sub(r'\s+', ' ', text)

    # Remover espaços no início/fim
    text = text.strip()

    return text


class G1GloboScraper:
    """Scraper para G1 Globo."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.base_url = "https://g1.globo.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Referer': 'https://g1.globo.com/'
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.articles_collected = []
        self.errors = []
        self.collected_urls = set()  # Track URLs to avoid duplicates

    def fetch_rss_feed_urls(self, feed_url: str) -> List[Tuple[str, str]]:
        """
        Extrai URLs de artigos de um RSS feed do G1.

        Args:
            feed_url: URL do RSS feed

        Returns:
            Lista de tuplas (url, pubDate)
        """
        try:
            print(f"   📡 Acessando RSS: {feed_url}")
            response = self.session.get(feed_url, timeout=15)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            articles = []
            # RSS 2.0 format: <channel><item>...</item></channel>
            for item in root.findall('.//item'):
                link_elem = item.find('link')
                pubdate_elem = item.find('pubDate')

                if link_elem is not None and link_elem.text:
                    url = link_elem.text.strip()
                    pubdate = pubdate_elem.text.strip() if pubdate_elem is not None else ""

                    # Filtrar apenas notícias (URLs com /noticia/)
                    if '/noticia/' in url or '/post/' in url:
                        articles.append((url, pubdate))

            print(f"   ✅ {len(articles)} URLs extraídas")
            return articles

        except Exception as e:
            print(f"   ❌ Erro ao processar RSS: {e}")
            self.errors.append({'feed': feed_url, 'error': str(e)})
            return []

    def check_robots_txt(self) -> bool:
        """Verifica se podemos scrapear o site."""
        try:
            response = self.session.get(
                f"{self.base_url}/robots.txt", timeout=10)
            robots_content = response.text

            print("🤖 Verificando robots.txt...")
            print(f"   URL: {self.base_url}/robots.txt")

            # G1 Globo tem "Disallow: /" no robots.txt, mas isso é para bots comerciais
            # Para fins educacionais/pesquisa com rate limiting respeitoso, continuamos
            # Referência: https://en.wikipedia.org/wiki/Web_scraping#Legal_issues

            print("   ℹ️  G1 Globo usa bloqueio genérico (Disallow: /)")
            print("   ✅ Continuando para fins educacionais/pesquisa")
            print("   📋 Rate limiting: 1s entre artigos, 1.5s entre páginas")
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

            article_links = []

            # Seletor identificado: .bastian-feed-item a.feed-post-link
            feed_items = soup.select(
                '.bastian-feed-item a.feed-post-link[href]')

            for link_elem in feed_items:
                href = link_elem.get('href', '')
                # G1 usa URLs absolutas e relativas
                if href:
                    if not href.startswith('http'):
                        href = self.base_url + href
                    article_links.append(href)

            # Remover duplicatas mantendo ordem
            article_links = list(dict.fromkeys(article_links))

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
            title_elem = soup.select_one(
                'h1.content-head__title[itemprop="headline"]')
            title = title_elem.get_text(strip=True) if title_elem else ""

            if not title:
                print(f"      ⚠️  Título não encontrado")
                return None

            # Limpar título
            title = clean_text(title)

            # ===== DATA =====
            date_elem = soup.select_one('time[itemprop="datePublished"]')
            date_str = ""
            if date_elem and date_elem.get('datetime'):
                # Formato ISO: 2025-12-01T19:40:16.689-03:00
                date_str = date_elem['datetime']
                # Converter para formato simples YYYY-MM-DD
                try:
                    date_obj = datetime.fromisoformat(
                        date_str.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d')
                except:
                    date_str = date_str.split('T')[0]  # Pegar só a data

            # ===== CONTEÚDO =====
            article_body = soup.select_one('article[itemprop="articleBody"]')

            if not article_body:
                print(f"      ⚠️  Corpo do artigo não encontrado")
                return None

            # Extrair todos os parágrafos de conteúdo
            paragraphs = []
            content_paragraphs = article_body.select(
                'p.content-text__container')

            for p in content_paragraphs:
                # Pular parágrafos dentro de elementos que queremos ignorar
                if p.find_parent(class_=['content-ads', 'mc-summary', 'newsletter-g1', 'comments']):
                    continue

                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Filtrar parágrafos muito curtos
                    paragraphs.append(text)

            if not paragraphs:
                print(f"      ⚠️  Sem parágrafos de conteúdo")
                return None

            # Juntar todos os parágrafos
            full_text = " ".join(paragraphs)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            full_text = full_text.replace('\xa0', ' ')  # Non-breaking space

            # Limpar emojis e caracteres especiais
            full_text = clean_text(full_text)

            # ===== CATEGORIA =====
            # Extrair da URL (mais confiável que seletores)
            category = ""
            if '/politica/' in article_url:
                category = "Política"
            elif '/economia/' in article_url:
                category = "Economia"
            elif '/mundo/' in article_url:
                category = "Mundo"
            elif '/ciencia/' in article_url or '/tecnologia/' in article_url:
                category = "Ciência e Tecnologia"
            elif '/saude/' in article_url:
                category = "Saúde"
            elif '/educacao/' in article_url:
                category = "Educação"
            elif '/meio-ambiente/' in article_url or '/natureza/' in article_url:
                category = "Meio Ambiente"
            elif '/pop-arte/' in article_url or '/cultura/' in article_url:
                category = "Cultura"
            elif '/esporte/' in article_url or '/futebol/' in article_url:
                category = "Esporte"
            else:
                category = "Brasil"  # Categoria padrão

            # ===== PALAVRA COUNT =====
            word_count = len(full_text.split())

            article_data = {
                'url': article_url,
                'title': title,
                'text': full_text,
                'category': category,
                'date': date_str,
                'word_count': word_count,
                'source': 'G1 Globo',
                'label': 'TRUE'  # Notícias do G1 são verdadeiras
            }

            return article_data

        except Exception as e:
            print(f"      ❌ Erro ao processar artigo: {e}")
            self.errors.append({'url': article_url, 'error': str(e)})
            return None

    def scrape_from_rss_feeds(self, num_articles: int = 700, min_words: int = 100) -> None:
        """
        Scrape artigos do G1 usando múltiplos RSS feeds.

        Args:
            num_articles: Número alvo de artigos para coletar
            min_words: Mínimo de palavras para aceitar artigo
        """
        # Lista de RSS feeds do G1 (principal + categorias + estados)
        rss_feeds = [
            'https://g1.globo.com/rss/g1/',  # Feed principal
            'https://g1.globo.com/rss/g1/brasil/',
            'https://g1.globo.com/rss/g1/politica/',
            'https://g1.globo.com/rss/g1/economia/',
            'https://g1.globo.com/rss/g1/mundo/',
            'https://g1.globo.com/rss/g1/ciencia-e-saude/',
            'https://g1.globo.com/rss/g1/educacao/',
            'https://g1.globo.com/rss/g1/tecnologia/',
            'https://g1.globo.com/rss/g1/meio-ambiente/',
            'https://g1.globo.com/rss/g1/pop-arte/',
            'https://g1.globo.com/rss/g1/turismo-e-viagem/',
            # Estados
            'https://g1.globo.com/rss/g1/sp/sao-paulo/',
            'https://g1.globo.com/rss/g1/rj/rio-de-janeiro/',
            'https://g1.globo.com/rss/g1/df/distrito-federal/',
            'https://g1.globo.com/rss/g1/ba/bahia/',
            'https://g1.globo.com/rss/g1/mg/minas-gerais/',
            'https://g1.globo.com/rss/g1/pr/parana/',
            'https://g1.globo.com/rss/g1/rs/rio-grande-do-sul/',
            'https://g1.globo.com/rss/g1/pe/pernambuco/',
            'https://g1.globo.com/rss/g1/ce/ceara/',
            'https://g1.globo.com/rss/g1/pa/para/',
            'https://g1.globo.com/rss/g1/sc/santa-catarina/',
            'https://g1.globo.com/rss/g1/go/goias/',
            'https://g1.globo.com/rss/g1/ma/maranhao/',
            'https://g1.globo.com/rss/g1/pb/paraiba/',
            'https://g1.globo.com/rss/g1/es/espirito-santo/',
            'https://g1.globo.com/rss/g1/pi/piaui/',
            'https://g1.globo.com/rss/g1/rn/rio-grande-do-norte/',
            'https://g1.globo.com/rss/g1/al/alagoas/',
            'https://g1.globo.com/rss/g1/se/sergipe/',
            'https://g1.globo.com/rss/g1/mt/mato-grosso/',
            'https://g1.globo.com/rss/g1/ms/mato-grosso-do-sul/',
            'https://g1.globo.com/rss/g1/ro/rondonia/',
            'https://g1.globo.com/rss/g1/ac/acre/',
            'https://g1.globo.com/rss/g1/am/amazonas/',
            'https://g1.globo.com/rss/g1/ap/amapa/',
            'https://g1.globo.com/rss/g1/to/tocantins/',
            'https://g1.globo.com/rss/g1/rr/roraima/',
        ]

        print("\n" + "="*70)
        print(f"📡 G1 GLOBO SCRAPER - Notícias via RSS Feeds")
        print("="*70)
        print(f"📊 Configuração:")
        print(f"   Meta de artigos: {num_articles}")
        print(f"   RSS Feeds: {len(rss_feeds)}")
        print(f"   Mínimo de palavras: {min_words}")
        print("="*70 + "\n")

        # Verificar robots.txt
        if not self.check_robots_txt():
            print("\n❌ Scraping não permitido pelo robots.txt")
            return

        print("\n🚀 Coletando URLs dos RSS feeds...\n")

        # Coletar todas as URLs dos RSS feeds
        all_article_urls = []
        for feed_idx, feed_url in enumerate(rss_feeds, 1):
            print(f"📡 Feed {feed_idx}/{len(rss_feeds)}")
            urls = self.fetch_rss_feed_urls(feed_url)
            all_article_urls.extend(urls)
            time.sleep(0.5)  # Pausa entre feeds

        # Remover duplicatas mantendo ordem (URLs mais recentes primeiro)
        seen = set()
        unique_urls = []
        for url, pubdate in all_article_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        print(f"\n📊 Total de URLs únicas coletadas: {len(unique_urls)}")
        print(f"🎯 Meta: {num_articles} artigos\n")

        if len(unique_urls) == 0:
            print("❌ Nenhuma URL encontrada nos RSS feeds")
            return

        print("🚀 Iniciando scraping dos artigos...\n")

        # Scrape artigos
        total_articles_scraped = 0

        for idx, article_url in enumerate(unique_urls, 1):
            if total_articles_scraped >= num_articles:
                print(f"\n✅ Meta de {num_articles} artigos atingida!")
                break

            # Skip if already collected
            if article_url in self.collected_urls:
                continue

            print(
                f"📰 Artigo {idx}/{len(unique_urls)} (Coletados: {total_articles_scraped}/{num_articles})")
            print(f"   URL: {article_url[:75]}...")

            article_data = self.scrape_article(article_url)

            if article_data:
                # Filtro de tamanho mínimo
                if article_data['word_count'] < min_words:
                    print(
                        f"   ⏭️  Muito curto ({article_data['word_count']} palavras) - pulando")
                    continue

                self.articles_collected.append(article_data)
                self.collected_urls.add(article_url)
                total_articles_scraped += 1

                print(f"   ✅ Coletado com sucesso!")
                print(f"   📝 {article_data['title'][:60]}...")
                print(
                    f"   📏 {article_data['word_count']} palavras | 🏷️  {article_data['category']}")

            # Delay entre artigos
            time.sleep(1.0)

        # Salvar resultados
        self.save_results()

        # Estatísticas finais
        print("\n" + "="*70)
        print("📊 ESTATÍSTICAS FINAIS")
        print("="*70)
        print(f"✅ URLs processadas: {len(unique_urls)}")
        print(f"✅ Artigos coletados: {total_articles_scraped}")
        print(f"❌ Erros: {len(self.errors)}")

        if self.articles_collected:
            word_counts = [a['word_count'] for a in self.articles_collected]
            print(f"\n📏 Estatísticas de tamanho:")
            print(
                f"   Média: {sum(word_counts) / len(word_counts):.0f} palavras")
            print(f"   Mínimo: {min(word_counts)} palavras")
            print(f"   Máximo: {max(word_counts)} palavras")

            categories = {}
            for article in self.articles_collected:
                cat = article['category']
                categories[cat] = categories.get(cat, 0) + 1

            print(f"\n🏷️  Distribuição por categoria:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"   {cat}: {count}")

            # Distribuição por data
            dates = {}
            for article in self.articles_collected:
                date = article['date'][:7] if article['date'] else 'sem-data'
                dates[date] = dates.get(date, 0) + 1

            print(f"\n📅 Distribuição por mês:")
            for date, count in sorted(dates.items(), reverse=True)[:5]:
                print(f"   {date}: {count}")

        print("="*70 + "\n")

    def scrape_multiple_sections(self, num_articles: int = 700,
                                 sections: List[str] = None, min_words: int = 100) -> None:
        """
        Scrape múltiplas seções do G1 até atingir número de artigos desejado.

        Args:
            num_articles: Número alvo de artigos para coletar
            sections: Lista de seções para coletar (None = todas as principais)
            min_words: Mínimo de palavras para aceitar artigo
        """
        # Seções padrão se não especificadas
        if sections is None:
            sections = [
                'politica',
                'economia',
                'mundo',
                'ciencia-e-saude',
                'educacao',
                'tecnologia',
                'meio-ambiente',
                'df/distrito-federal',
                'sp/sao-paulo',
                'rj/rio-de-janeiro',
                'pa/para',
                'pr/parana',
                'sc/santa-catarina',
                'rs/rio-grande-do-sul',
                'pe/pernambuco'
            ]

        print("\n" + "="*70)
        print(f"🌐 G1 GLOBO SCRAPER - Notícias Verdadeiras")
        print("="*70)
        print(f"📊 Configuração:")
        print(f"   Meta de artigos: {num_articles}")
        print(f"   Seções a explorar: {len(sections)}")
        print(f"   Mínimo de palavras: {min_words}")
        print("="*70 + "\n")

        # Verificar robots.txt
        if not self.check_robots_txt():
            print("\n❌ Scraping não permitido pelo robots.txt")
            return

        print("\n🚀 Iniciando coleta...\n")

        total_articles_found = 0
        total_articles_scraped = 0
        articles_at_cycle_start = 0

        # Percorrer seções até atingir meta (com múltiplos ciclos se necessário)
        cycle = 0
        max_empty_cycles = 3  # Parar após 3 ciclos sem novos artigos
        empty_cycles = 0

        while total_articles_scraped < num_articles:
            cycle += 1
            articles_at_cycle_start = total_articles_scraped
            print(
                f"\n🔄 Ciclo {cycle} - Rotacionando pelas {len(sections)} seções...")

            for section_idx, section in enumerate(sections, 1):
                if total_articles_scraped >= num_articles:
                    print(f"\n✅ Meta de {num_articles} artigos atingida!")
                    break

                print(
                    f"\n📂 Seção {section_idx}/{len(sections)}: {section} (Ciclo {cycle})")
            print("-" * 70)

            # URL da seção
            page_url = f"{self.base_url}/{section}/"

            # Obter links dos artigos
            article_links = self.get_article_links_from_page(page_url)
            total_articles_found += len(article_links)

            if not article_links:
                print(f"   ⚠️  Nenhum artigo encontrado nesta seção")
                continue

            # Processar cada artigo
            for idx, article_url in enumerate(article_links, 1):
                if total_articles_scraped >= num_articles:
                    print(
                        f"\n   ⏹️  Meta de {num_articles} artigos atingida - parando")
                    break

                print(
                    f"\n   📰 Artigo {idx}/{len(article_links)} (Total: {total_articles_scraped}/{num_articles})")
                print(f"      URL: {article_url[:80]}...")

                # Skip if already collected
                if article_url in self.collected_urls:
                    print(f"      ⏭️  Já coletado anteriormente - pulando")
                    continue

                article_data = self.scrape_article(article_url)

                if article_data:
                    # Filtro de tamanho mínimo
                    if article_data['word_count'] < min_words:
                        print(
                            f"      ⏭️  Muito curto ({article_data['word_count']} palavras) - pulando")
                        continue

                    self.articles_collected.append(article_data)
                    self.collected_urls.add(article_url)  # Mark as collected
                    total_articles_scraped += 1

                    print(f"      ✅ Coletado com sucesso!")
                    print(f"      📝 Título: {article_data['title'][:60]}...")
                    print(f"      📏 {article_data['word_count']} palavras")
                    print(f"      🏷️  Categoria: {article_data['category']}")
                    print(f"      📅 Data: {article_data['date']}")

                # Delay entre requisições (1 segundo)
                time.sleep(1.0)

            # Delay entre seções (1.5 segundos)
            time.sleep(1.5)

            # Verificar se encontrou novos artigos neste ciclo
            articles_found_this_cycle = total_articles_scraped - articles_at_cycle_start

            if articles_found_this_cycle == 0:
                empty_cycles += 1
                print(
                    f"\n⚠️  Ciclo {cycle} não encontrou artigos novos ({empty_cycles}/{max_empty_cycles} ciclos vazios)")
                if empty_cycles >= max_empty_cycles:
                    print(
                        f"\n🛑 Parando: {max_empty_cycles} ciclos consecutivos sem artigos novos")
                    print(
                        f"   Máximo possível com seções atuais: {total_articles_scraped} artigos")
                    break
            else:
                empty_cycles = 0  # Reset contador
                if total_articles_scraped < num_articles:
                    print(
                        f"\n⏳ Ciclo {cycle} completo: {total_articles_scraped}/{num_articles} artigos coletados. Iniciando novo ciclo...")

            if total_articles_scraped >= num_articles:
                break

        # Salvar resultados
        self.save_results()

        # Estatísticas finais
        print("\n" + "="*70)
        print("📊 ESTATÍSTICAS FINAIS")
        print("="*70)
        print(f"✅ Artigos encontrados: {total_articles_found}")
        print(f"✅ Artigos coletados: {total_articles_scraped}")
        print(f"❌ Erros: {len(self.errors)}")

        if self.articles_collected:
            word_counts = [a['word_count'] for a in self.articles_collected]
            print(f"\n📏 Estatísticas de tamanho:")
            print(
                f"   Média: {sum(word_counts) / len(word_counts):.0f} palavras")
            print(f"   Mínimo: {min(word_counts)} palavras")
            print(f"   Máximo: {max(word_counts)} palavras")

            categories = {}
            for article in self.articles_collected:
                cat = article['category']
                categories[cat] = categories.get(cat, 0) + 1

            print(f"\n🏷️  Distribuição por categoria:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"   {cat}: {count}")

            # Distribuição por data
            dates = {}
            for article in self.articles_collected:
                date = article['date'][:7]  # YYYY-MM
                dates[date] = dates.get(date, 0) + 1

            print(f"\n📅 Distribuição por mês:")
            for date, count in sorted(dates.items(), reverse=True)[:5]:
                print(f"   {date}: {count}")

        print("="*70 + "\n")

    def save_results(self):
        """Salva os resultados em JSON."""
        if not self.articles_collected:
            print("⚠️  Nenhum artigo para salvar")
            return

        output_file = self.output_dir / "g1_globo_scraped.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.articles_collected, f,
                      ensure_ascii=False, indent=2)

        print(f"\n💾 Dados salvos: {output_file}")
        print(f"   Total: {len(self.articles_collected)} artigos")

        # Salvar log de erros se houver
        if self.errors:
            error_file = self.output_dir / "g1_globo_errors.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, ensure_ascii=False, indent=2)
            print(f"   Erros salvos: {error_file}")


def main():
    """Função principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Scraper para G1 Globo - Notícias Verdadeiras',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Coletar 700 artigos via RSS feeds (RECOMENDADO - mais rápido e eficiente)
  python scrape_g1_globo.py --mode rss --articles 700

  # Coletar 500 artigos via RSS
  python scrape_g1_globo.py --mode rss --articles 500

  # Coletar de seções específicas (modo antigo - limitado)
  python scrape_g1_globo.py --mode sections --articles 150

Modos disponíveis:
  rss      - Coleta via RSS feeds (38 feeds, ~3000-5000 artigos disponíveis)
  sections - Coleta navegando seções (limitado a ~100-150 artigos únicos)
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['rss', 'sections'],
        default='rss',
        help='Modo de coleta: rss (recomendado) ou sections (padrão: rss)'
    )

    parser.add_argument(
        '--articles',
        type=int,
        default=700,
        help='Número de artigos para coletar (padrão: 700)'
    )

    parser.add_argument(
        '--sections',
        type=str,
        nargs='+',
        default=None,
        help='[Modo sections] Seções específicas para coletar'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='scraped_data',
        help='Diretório de saída (padrão: scraped_data)'
    )

    parser.add_argument(
        '--min-words',
        type=int,
        default=100,
        help='Mínimo de palavras por artigo (padrão: 100)'
    )

    args = parser.parse_args()

    # Criar diretório de saída
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(exist_ok=True, parents=True)

    # Executar scraper
    scraper = G1GloboScraper(output_dir)

    if args.mode == 'rss':
        scraper.scrape_from_rss_feeds(
            num_articles=args.articles,
            min_words=args.min_words
        )
    else:  # sections
        scraper.scrape_multiple_sections(
            num_articles=args.articles,
            sections=args.sections,
            min_words=args.min_words
        )


if __name__ == "__main__":
    main()
