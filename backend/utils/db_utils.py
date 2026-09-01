"""Database utilities for general connections and read operations for benchmarks, prices, and news data."""
import sqlite3
import pandas as pd
import os
import logging
from database.dml.benchmarks import SELECT_ALL_BENCHMARKS
from database.dml.benchmark_tickers import SELECT_BENCHMARK_CONSTITUENTS
from database.dml.benchmark_prices import SELECT_BENCHMARK_PRICES
from database.dml.headlines import SELECT_ALL_HEADLINES
from database.dml.ticker_prices import SELECT_HISTORICAL_PRICES

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'data', 'screener_data.db')

def get_db_connection():
    """Returns a SQLite connection, raising FileNotFoundError if the DB doesn't exist."""
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}. Please run update_db.py first.")
        raise FileNotFoundError("Database not found.")
    return sqlite3.connect(DB_PATH)

def get_latest_benchmarks():
    """Reads all benchmarks with their price history and constituent tickers from SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(SELECT_ALL_BENCHMARKS)
        benchmark_rows = cursor.fetchall()
        
        benchmarks = []
        for row in benchmark_rows:
            symbol, name, price, change = row
            
            cursor.execute(SELECT_BENCHMARK_PRICES, (symbol,))
            price_rows = cursor.fetchall()
            
            benchmark = {
                "symbol": symbol,
                "name": name,
                "price": price,
                "change": change,
                "history": [price_row[2] for price_row in price_rows],
                "dates": [price_row[0] for price_row in price_rows],
                "open": [price_row[3] for price_row in price_rows],
                "high": [price_row[4] for price_row in price_rows],
                "low": [price_row[5] for price_row in price_rows],
                "volume": [price_row[6] for price_row in price_rows],
                "constituents": []
            }
            
            cursor.execute(SELECT_BENCHMARK_CONSTITUENTS, (symbol,))
            constituent_rows = cursor.fetchall()
            
            for constituent_row in constituent_rows:
                ticker_symbol, company_name, sector, market_cap, weight = constituent_row
                benchmark["constituents"].append({
                    "symbol": ticker_symbol,
                    "name": company_name,
                    "sector": sector,
                    "marketCap": market_cap,
                    "weight": weight
                })
                
            benchmarks.append(benchmark)
            
        conn.close()
        return benchmarks
    except Exception as e:
        logger.exception(f"Error reading benchmarks from DB: {e}")
        return []

def get_latest_headlines():
    """Reads all cached news headlines from SQLite."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(SELECT_ALL_HEADLINES)
        rows = cursor.fetchall()
        conn.close()
        
        headlines = []
        for row in rows:
            headlines.append({
                "title": row[0],
                "publisher": row[1],
                "link": row[2],
                "summary": row[3],
                "time": row[4]
            })
        return headlines
    except Exception as e:
        logger.exception(f"Error reading headlines from DB: {e}")
        return []

def get_historical_prices_df():
    """Reads ticker price history from SQLite and reshapes it into a MultiIndex DataFrame."""
    try:
        conn = get_db_connection()
        df_stacked = pd.read_sql(SELECT_HISTORICAL_PRICES, conn)
        conn.close()
        
        if df_stacked.empty:
            return pd.DataFrame()
            
        df_stacked = df_stacked.rename(columns={
            'price_date': 'Date',
            'ticker_symbol': 'Ticker',
            'ticker': 'Ticker',
            'close_price': 'Close',
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'volume': 'Volume',
            'dividends': 'Dividends',
            'stock_splits': 'Stock Splits'
        })
        
        df_stacked['Date'] = pd.to_datetime(df_stacked['Date'])
        df_stacked = df_stacked.set_index(['Date', 'Ticker'])
        
        data = df_stacked.unstack(level=1)
        data.columns = data.columns.swaplevel(0, 1)
        return data
    except Exception as e:
        logger.exception(f"Error reading historical prices from DB: {e}")
        return pd.DataFrame()
