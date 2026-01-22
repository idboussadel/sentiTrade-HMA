from src.data.collectors import NewsAPICollector
from datetime import datetime, timedelta

# Initialiser le collecteur
collector = NewsAPICollector()

# Tester avec AAPL seulement (5 articles max)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f'Collecte de news pour AAPL...')
print(f'Période: {start_date.date()}  {end_date.date()}')

news = collector.collect(
    tickers=['AAPL'],
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    max_articles=5
)

if not news.empty:
    print(f'\n {len(news)} articles collectés')
    print(f'\nAperçu:')
    print(news[['ticker', 'date', 'title']].head())
else:
    print('  Aucun article collecté (vérifier les limites API)')
