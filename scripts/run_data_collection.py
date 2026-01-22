#!/usr/bin/env python3
"""
Main script to collect and preprocess data for SentiTrade-HMA.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collectors import FinancialDataCollector, NewsAPICollector
from src.data.preprocessors import TechnicalIndicatorCalculator
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.helpers import load_tickers, save_tickers

logger = get_logger(__name__)


def main():
    '''Main data collection pipeline.'''
    logger.info('=' * 80)
    logger.info(' SentiTrade-HMA Data Collection Pipeline')
    logger.info('=' * 80)
    
    # Load configuration
    config = get_config()
    data_config = config.get_data_config()
    
    # Define tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    save_tickers(tickers, config.data_dir / 'sp500_top100_tickers.txt')
    logger.info(f' Using {len(tickers)} tickers')
    
    start_date = data_config['data_sources']['financial']['start_date']
    end_date = data_config['data_sources']['financial']['end_date']
    
    # Step 1: Collect Financial Data
    logger.info('')
    logger.info('=' * 80)
    logger.info('STEP 1: Collecting Financial Data from Yahoo Finance')
    logger.info('=' * 80)
    
    financial_collector = FinancialDataCollector()
    financial_data = financial_collector.collect(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    if not financial_data.empty:
        financial_collector.save(
            financial_data,
            str(config.data_dir / 'raw' / 'financial_data_raw.csv')
        )
        
        # Calculate technical indicators
        logger.info('')
        logger.info(' Calculating technical indicators...')
        tech_calculator = TechnicalIndicatorCalculator()
        financial_with_indicators = tech_calculator.add_all_indicators(financial_data)
        tech_calculator.save(
            financial_with_indicators,
            str(config.data_dir / 'processed' / 'financial_data_with_indicators.csv')
        )
    else:
        logger.error(' No financial data collected. Stopping.')
        return
    
    # Step 2: Collect News Data (optional)
    logger.info('')
    logger.info('=' * 80)
    logger.info('STEP 2: Collecting News from News API')
    logger.info('=' * 80)
    
    try:
        news_collector = NewsAPICollector()
        news_data = news_collector.collect(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_articles=100
        )
        
        if not news_data.empty:
            news_collector.save(
                news_data,
                str(config.data_dir / 'raw' / 'news_articles.csv')
            )
        else:
            logger.warning('  No news data collected (check API key)')
            
    except Exception as e:
        logger.error(f' News collection failed: {str(e)}')
        logger.info(' Continuing without news data...')
    
    # Summary
    logger.info('')
    logger.info('=' * 80)
    logger.info(' DATA COLLECTION COMPLETE')
    logger.info('=' * 80)
    logger.info(f' Raw data: {config.data_dir / \"raw\"}')
    logger.info(f' Processed data: {config.data_dir / \"processed\"}')
    logger.info('')
    logger.info(' Next steps:')
    logger.info('  1. Fine-tune sentiment LLM: python src/training/train_llm.py')
    logger.info('  2. Train TFT model: python src/training/train_tft.py')
    logger.info('  3. Run backtesting: python src/evaluation/backtester.py')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
