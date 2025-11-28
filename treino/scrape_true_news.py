"""
Script para coletar notícias VERDADEIRAS automaticamente de fontes confiáveis.
Foca em notícias de política para balancear o dataset.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime

# Seu NewsAPI que ainda funciona
NEWSAPI_KEY = "772a8f94de344812b28474bf59eab887"


def collect_from_newsapi(query, pages=3):
    """Coleta notícias da NewsAPI (fontes brasileiras confiáveis)."""
    articles = []

    # Fontes confiáveis brasileiras
    sources = "globo,folha-de-s-paulo,google-news-br"

    for page in range(1, pages + 1):
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "pt",
            "sortBy": "relevancy",
            "pageSize": 100,
            "page": page,
            "apiKey": NEWSAPI_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", []):
                    # Pegar título + descrição + conteúdo
                    text = ""
                    if article.get("title"):
                        text += article["title"] + "\n\n"
                    if article.get("description"):
                        text += article["description"] + "\n\n"
                    if article.get("content"):
                        # Remove o "[+X chars]" que a NewsAPI adiciona
                        content = article["content"].split("[+")[0].strip()
                        text += content

                    if len(text) > 200:  # Mínimo de caracteres
                        articles.append({
                            "text": text.strip(),
                            "label": "true",
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                            "url": article.get("url", ""),
                            "collected_at": datetime.now().isoformat()
                        })

                print(
                    f"✅ Página {page}: {len(data.get('articles', []))} artigos coletados")
                time.sleep(1)  # Rate limit
            else:
                print(f"❌ Erro na página {page}: {response.status_code}")
                break
        except Exception as e:
            print(f"❌ Erro: {e}")
            break

    return articles


def collect_from_rss_feeds():
    """Coleta de feeds RSS de veículos confiáveis."""
    feeds = [
        "https://www.poder360.com.br/feed/",
        "https://noticias.uol.com.br/politica/index.xml",
    ]

    articles = []
    for feed_url in feeds:
        try:
            response = requests.get(feed_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')

                for item in items[:50]:  # Pegar até 50 por feed
                    title = item.find('title').text if item.find(
                        'title') else ""
                    description = item.find('description').text if item.find(
                        'description') else ""

                    text = f"{title}\n\n{description}".strip()

                    if len(text) > 200:
                        articles.append({
                            "text": text,
                            "label": "true",
                            "source": feed_url,
                            "collected_at": datetime.now().isoformat()
                        })

                print(f"✅ Feed {feed_url}: {len(items)} artigos")
        except Exception as e:
            print(f"❌ Erro no feed {feed_url}: {e}")

    return articles


def main():
    print("🔍 Coletando notícias VERDADEIRAS...\n")

    all_articles = []

    # Tópicos políticos brasileiros importantes
    queries = [
        "Supremo Tribunal Federal",
        "STF Brasil",
        "Bolsonaro política",
        "Lula governo",
        "Congresso Nacional Brasil",
        "Ministério Público Brasil",
        "Operação Polícia Federal",
    ]

    # Coletar via NewsAPI
    print("📰 Coletando da NewsAPI...")
    for query in queries:
        print(f"\n  Buscando: {query}")
        articles = collect_from_newsapi(query, pages=2)
        all_articles.extend(articles)
        time.sleep(2)

    # Coletar via RSS
    print("\n\n📡 Coletando de feeds RSS...")
    rss_articles = collect_from_rss_feeds()
    all_articles.extend(rss_articles)

    # Remover duplicatas (por texto)
    unique_texts = set()
    unique_articles = []
    for article in all_articles:
        text_key = article["text"][:100]  # Primeiros 100 chars como chave
        if text_key not in unique_texts:
            unique_texts.add(text_key)
            unique_articles.append(article)

    print(f"\n\n✅ Total coletado: {len(all_articles)} artigos")
    print(f"✅ Únicos: {len(unique_articles)} artigos")

    # Salvar
    output_file = "true_news_collected.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_articles, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Salvo em: {output_file}")
    print(f"\n🎯 Próximo passo: Execute 'python merge_dataset.py' para adicionar ao dataset")


if __name__ == "__main__":
    main()
