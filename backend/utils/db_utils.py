"""Database utilities for general connections and read operations for benchmarks, prices, and news data."""
import pandas as pd
import os
import logging
import urllib.parse
from sqlalchemy import create_engine
from database.dml.benchmarks import SELECT_ALL_BENCHMARKS
from database.dml.benchmark_tickers import SELECT_BENCHMARK_CONSTITUENTS
from database.dml.benchmark_prices import SELECT_BENCHMARK_PRICES
from database.dml.headlines import SELECT_ALL_HEADLINES
from database.dml.ticker_prices import SELECT_HISTORICAL_PRICES

logger = logging.getLogger(__name__)

engine = None

def get_engine():
    global engine
    if engine is None:
        db_user = os.environ.get("DB_USER", "ADMIN")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_dsn = os.environ.get("DB_DSN", "")
        
        if not db_dsn:
            logger.error("DB_DSN environment variable is not set. Cannot connect to Oracle DB.")
            raise ValueError("DB_DSN is not set.")
            
        encoded_dsn = urllib.parse.quote_plus(db_dsn)
        connection_url = f"oracle+oracledb://{db_user}:{db_password}@/?dsn={encoded_dsn}"
        engine = create_engine(connection_url)
    return engine

def get_db_connection():
    """Returns a DBAPI connection from the SQLAlchemy engine."""
    return get_engine().raw_connection()

def get_latest_benchmarks():
    """Reads all benchmarks with their price history and constituent tickers from Oracle DB."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(SELECT_ALL_BENCHMARKS)
        benchmark_rows = cursor.fetchall()
        
        benchmarks = []
        for row in benchmark_rows:
            symbol, name, price, change = row
            
            cursor.execute(SELECT_BENCHMARK_PRICES, [symbol])
            price_rows = cursor.fetchall()
            
            benchmark = {
                "symbol": symbol,
                "name": name,
                "price": price,
                "change": change,
                "history": [price_row[2] for price_row in price_rows],
                "dates": [price_row[0].strftime('%Y-%m-%d') if hasattr(price_row[0], 'strftime') else price_row[0] for price_row in price_rows],
                "open": [price_row[3] for price_row in price_rows],
                "high": [price_row[4] for price_row in price_rows],
                "low": [price_row[5] for price_row in price_rows],
                "volume": [price_row[6] for price_row in price_rows],
                "constituents": []
            }
            
            cursor.execute(SELECT_BENCHMARK_CONSTITUENTS, [symbol])
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
    """Reads all cached news headlines from Oracle DB."""
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
    """Reads ticker price history from Oracle DB and reshapes it into a MultiIndex DataFrame."""
    try:
        eng = get_engine()
        df_stacked = pd.read_sql(SELECT_HISTORICAL_PRICES, eng)
        
        if df_stacked.empty:
            return pd.DataFrame()
            
        # Oracle might return column names in lowercase depending on SQLAlchemy
        df_stacked.columns = [c.lower() for c in df_stacked.columns]
        
        df_stacked = df_stacked.rename(columns={
            'price_date': 'Date',
            'ticker_symbol': 'Ticker',
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
