import pandas as pd
from src.data.preprocessors import TechnicalIndicatorCalculator

# Charger les données collectées
data = pd.read_csv('data/raw/financial_data_test.csv')
print(f'Données chargées: {len(data)} lignes')

# Calculer les indicateurs
calculator = TechnicalIndicatorCalculator()
data_with_indicators = calculator.add_all_indicators(data)

print(f' Indicateurs calculés')
print(f'Colonnes: {data_with_indicators.columns.tolist()}')
print('\nAperçu:')
print(data_with_indicators[['Date', 'Ticker', 'Close', 'HMA', 'RSI', 'MACD']].head(10))
