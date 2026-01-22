import yaml
from pathlib import Path

config = {
    'project': {
        'name': 'SentiTrade-HMA',
        'version': '2.0',
        'description': 'Hybrid Market Anticipation'
    },
    'data_sources': {
        'financial': {
            'provider': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2024-12-24',
            'interval': '1d'
        },
        'news': {
            'provider': 'newsapi',
            'start_date': '2023-01-01',
            'end_date': '2024-12-24',
            'max_articles_per_ticker': 100
        }
    },
    'tickers': {
        'source': 'sp500',
        'selection': 'top100',
        'custom_list': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    },
    'preprocessing': {
        'technical_indicators': [
            {'name': 'HMA', 'period': 20},
            {'name': 'RSI', 'period': 14},
            {'name': 'MACD', 'fast': 12, 'slow': 26, 'signal': 9}
        ],
        'train_test_split': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'shuffle': False
        }
    },
    'features': {
        'numerical': ['Open', 'High', 'Low', 'Close', 'Volume', 'HMA', 'RSI', 'MACD'],
        'categorical': ['Ticker'],
        'target': ['Close']
    }
}

Path('configs').mkdir(exist_ok=True)
with open('configs/data_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(' data_config.yaml créé')
