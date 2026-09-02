"""Database update script that fetches market data and writes it to SQLite."""
import sqlite3
import pandas as pd
import yfinance as yf
import logging
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.services.external_data_service import fetch_benchmarks, fetch_headlines
from backend.database.dml.benchmarks import DELETE_ALL_BENCHMARKS, INSERT_BENCHMARK
from backend.database.dml.tickers import DELETE_ALL_TICKERS, INSERT_TICKER
from backend.database.dml.benchmark_tickers import DELETE_ALL_BENCHMARK_TICKERS, INSERT_BENCHMARK_TICKER
from backend.database.dml.headlines import DELETE_ALL_HEADLINES, INSERT_HEADLINE
from backend.database.dml.benchmark_prices import DELETE_ALL_BENCHMARK_PRICES, INSERT_BENCHMARK_PRICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'screener_data.db')
DDL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ddl')

CHUNK_SIZE = 20
RATE_LIMIT_DELAY = 4

def get_retry_session():
    """Creates a requests session with exponential backoff retries."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session

def _retry_missing_tickers(data, missing, start_str, end_str, session):
    for t in missing:
        try:
            t_data = yf.download([t], start=start_str, end=end_str, group_by="ticker", actions=True, progress=False, threads=False, session=session)
            if not t_data.empty:
                if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[0]:
                    data = data.drop(columns=[t], level=0)
                data = pd.concat([data, t_data], axis=1)
        except Exception as e:
            logger.exception(f"Failed to retry {t}: {e}")
        time.sleep(1)
    return data

def _structure_and_write_data(data, conn):
    logger.info("Structuring data for SQL...")
    data = data.copy()
    df_stacked = data.stack(level=0, future_stack=True).reset_index()
    df_stacked = df_stacked.rename(columns={
        'level_1': 'ticker_symbol',
        'Ticker': 'ticker_symbol',
        'Date': 'price_date',
        'Close': 'close_price',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Volume': 'volume',
        'Dividends': 'dividends',
        'Stock Splits': 'stock_splits'
    })
    df_stacked.columns = [col.lower().replace(' ', '_') for col in df_stacked.columns]
    
    logger.info("Writing historical prices to SQLite...")
    df_stacked.to_sql('ticker_prices', conn, if_exists='replace', index=False)

def _download_ticker_history(tickers, conn):
    """Downloads 1-year price history for all tickers in chunks and writes to SQLite."""
    session = get_retry_session()
    all_data = []
    
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=366)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    total_chunks = (len(tickers) - 1) // CHUNK_SIZE + 1
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        logger.info(f"Downloading chunk {chunk_num}/{total_chunks} ({len(chunk)} tickers)...")
        
        try:
            chunk_data = yf.download(
                chunk, start=start_str, end=end_str,
                group_by="ticker", actions=True, progress=False,
                threads=2, session=session
            )
            
            if not chunk_data.empty:
                if len(chunk) == 1 and not isinstance(chunk_data.columns, pd.MultiIndex):
                    chunk_data.columns = pd.MultiIndex.from_product([[chunk[0]], chunk_data.columns])
                all_data.append(chunk_data)
        except Exception as e:
            logger.exception(f"Error downloading chunk {chunk_num}: {e}")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    if all_data:
        logger.info("Concatenating chunks...")
        data = pd.concat(all_data, axis=1)
        
        logger.info("Validating downloaded data...")
        missing = []
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if t not in data.columns.levels[0] or data[t]['Close'].dropna().empty:
                    missing.append(t)
        
        if missing:
            logger.info(f"Retrying {len(missing)} missing/failed tickers individually...")
            data = _retry_missing_tickers(data, missing, start_str, end_str, session)
        
        _structure_and_write_data(data, conn)

def _insert_benchmark_prices(conn, benchmark):
    dates = benchmark.get("dates", [])
    closes = benchmark.get("history", [])
    opens = benchmark.get("open", [])
    highs = benchmark.get("high", [])
    lows = benchmark.get("low", [])
    volumes = benchmark.get("volume", [])
    
    for idx in range(len(dates)):
        conn.execute(INSERT_BENCHMARK_PRICE, (
            dates[idx],
            benchmark["benchmark_symbol"],
            closes[idx] if idx < len(closes) else 0,
            opens[idx] if idx < len(opens) else 0,
            highs[idx] if idx < len(highs) else 0,
            lows[idx] if idx < len(lows) else 0,
            volumes[idx] if idx < len(volumes) else 0
        ))

def _insert_benchmark_constituents(conn, benchmark):
    for constituent in benchmark["tickers"]:
        conn.execute(INSERT_TICKER, (
            constituent["ticker_symbol"],
            constituent.get("company_name"),
            constituent.get("sector"),
            constituent.get("market_cap")
        ))
        conn.execute(INSERT_BENCHMARK_TICKER, (
            benchmark["benchmark_symbol"],
            constituent["ticker_symbol"],
            constituent.get("weight")
        ))

def _write_benchmarks(benchmarks, conn):
    """Writes normalized benchmark data (indices, constituents, prices) to SQLite."""
    conn.execute(DELETE_ALL_BENCHMARK_TICKERS)
    conn.execute(DELETE_ALL_TICKERS)
    conn.execute(DELETE_ALL_BENCHMARK_PRICES)
    conn.execute(DELETE_ALL_BENCHMARKS)
    
    for benchmark in benchmarks:
        conn.execute(INSERT_BENCHMARK, (
            benchmark["benchmark_symbol"],
            benchmark["benchmark_name"],
            benchmark.get("current_price"),
            benchmark.get("change_pct")
        ))
        
        _insert_benchmark_prices(conn, benchmark)
        _insert_benchmark_constituents(conn, benchmark)

def _write_headlines(conn):
    """Fetches and writes news headlines to SQLite."""
    logger.info("Fetching headlines...")
    news = fetch_headlines()
    conn.execute(DELETE_ALL_HEADLINES)
    for headline in news:
        conn.execute(INSERT_HEADLINE, (
            headline.get("title"),
            headline.get("publisher"),
            headline.get("link"),
            headline.get("summary"),
            headline.get("time")
        ))

def update_database():
    """Main entry point: refreshes all screener data in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    
    logger.info("Initializing database schema...")
    for ddl_file in os.listdir(DDL_DIR):
        if ddl_file.endswith('.sql'):
            with open(os.path.join(DDL_DIR, ddl_file), 'r') as f:
                conn.executescript(f.read())
    
    logger.info("Fetching benchmarks and constituents...")
    benchmarks = fetch_benchmarks()
    
    all_tickers = set()
    for benchmark in benchmarks:
        for constituent in benchmark["tickers"]:
            all_tickers.add(constituent["ticker_symbol"])
    tickers = list(all_tickers)
    
    logger.info(f"Downloading 1y history for {len(tickers)} tickers in chunks...")
    if tickers:
        _download_ticker_history(tickers, conn)
    
    logger.info("Writing normalized benchmarks to SQLite...")
    _write_benchmarks(benchmarks, conn)
    
    _write_headlines(conn)
    
    conn.commit()
    conn.close()
    logger.info("Database update complete!")

if __name__ == "__main__":
    update_database()
