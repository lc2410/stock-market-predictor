import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.db_utils import (
    get_db_connection,
    get_latest_benchmarks,
    get_latest_headlines,
    get_historical_prices_df
)

@patch('utils.db_utils.os.path.exists')
@patch('utils.db_utils.sqlite3.connect')
def test_get_db_connection_success(mock_connect, mock_exists):
    mock_exists.return_value = True
    conn = get_db_connection()
    mock_connect.assert_called_once()
    assert conn == mock_connect.return_value

@patch('utils.db_utils.os.path.exists')
def test_get_db_connection_not_found(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        get_db_connection()

@patch('utils.db_utils.get_db_connection')
def test_get_latest_benchmarks(mock_get_conn):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock return values for cursor.execute
    # 1. SELECT_ALL_BENCHMARKS
    # 2. SELECT_BENCHMARK_PRICES for first benchmark
    # 3. SELECT_BENCHMARK_CONSTITUENTS for first benchmark
    
    def side_effect(query, params=()):
        if 'SELECT * FROM benchmarks' in query:
            mock_cursor.fetchall.return_value = [('SPY', 'SPDR S&P 500', 400.0, 1.5)]
        elif 'benchmark_prices' in query:
            # price_date, benchmark_symbol, close_price, open_price, high_price, low_price, volume
            mock_cursor.fetchall.return_value = [('2023-01-01', 'SPY', 400.0, 395.0, 405.0, 390.0, 1000)]
        elif 'benchmark_tickers' in query:
            # ticker_symbol, company_name, sector, market_cap, weight
            mock_cursor.fetchall.return_value = [('AAPL', 'Apple Inc.', 'Technology', 2000000, 0.05)]
    
    mock_cursor.execute.side_effect = side_effect
    
    benchmarks = get_latest_benchmarks()
    assert len(benchmarks) == 1
    assert benchmarks[0]['symbol'] == 'SPY'
    assert len(benchmarks[0]['history']) == 1
    assert benchmarks[0]['history'][0] == 400.0
    assert len(benchmarks[0]['constituents']) == 1
    assert benchmarks[0]['constituents'][0]['symbol'] == 'AAPL'

@patch('utils.db_utils.get_db_connection')
def test_get_latest_benchmarks_exception(mock_get_conn):
    mock_get_conn.side_effect = Exception("DB Error")
    benchmarks = get_latest_benchmarks()
    assert benchmarks == []

@patch('utils.db_utils.get_db_connection')
def test_get_latest_headlines(mock_get_conn):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.fetchall.return_value = [
        ('Title', 'Publisher', 'http://link', 'Summary', '2023-01-01')
    ]
    
    headlines = get_latest_headlines()
    assert len(headlines) == 1
    assert headlines[0]['title'] == 'Title'
    
@patch('utils.db_utils.get_db_connection')
def test_get_latest_headlines_exception(mock_get_conn):
    mock_get_conn.side_effect = Exception("DB Error")
    headlines = get_latest_headlines()
    assert headlines == []

@patch('utils.db_utils.get_db_connection')
@patch('utils.db_utils.pd.read_sql')
def test_get_historical_prices_df(mock_read_sql, mock_get_conn):
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    
    # Mock empty dataframe
    mock_read_sql.return_value = pd.DataFrame()
    df = get_historical_prices_df()
    assert df.empty
    
    # Mock dataframe with data
    mock_df = pd.DataFrame({
        'price_date': ['2023-01-01', '2023-01-01'],
        'ticker_symbol': ['AAPL', 'MSFT'],
        'close_price': [150, 250],
        'open_price': [145, 245],
        'high_price': [155, 255],
        'low_price': [140, 240],
        'volume': [100, 200],
        'dividends': [0, 0],
        'stock_splits': [0, 0]
    })
    mock_read_sql.return_value = mock_df
    
    df2 = get_historical_prices_df()
    assert not df2.empty
    assert 'AAPL' in df2.columns.levels[0]
    assert 'MSFT' in df2.columns.levels[0]

@patch('utils.db_utils.get_db_connection')
def test_get_historical_prices_df_exception(mock_get_conn):
    mock_get_conn.side_effect = Exception("DB Error")
    df = get_historical_prices_df()
    assert df.empty
