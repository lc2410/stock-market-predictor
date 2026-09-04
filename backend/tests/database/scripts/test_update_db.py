import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

# Ensure backend is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from database.scripts.update_db import (
    _download_ticker_history,
    _write_benchmarks,
    _write_headlines,
    get_retry_session,
    update_database,
)


def test_get_retry_session():
    session = get_retry_session()
    assert session is not None
    assert 'User-Agent' in session.headers
    # Test adapter exists
    assert 'http://' in session.adapters

@patch('database.scripts.update_db.yf.download')
@patch('database.scripts.update_db.time.sleep', MagicMock())
@patch('database.scripts.update_db.get_retry_session', MagicMock())
def test_download_ticker_history(mock_download):
    mock_conn = MagicMock()
    
    # Test with empty chunk data
    mock_download.return_value = pd.DataFrame()
    _download_ticker_history(['AAPL'], mock_conn)
    # Should not write to sql if empty
    mock_conn.execute.assert_not_called()
    
    # Test with valid multiindex chunk data
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    df = pd.DataFrame({
        "Close": [150, 151],
        "Open": [149, 150],
        "High": [155, 156],
        "Low": [145, 146],
        "Volume": [100, 200],
        "Dividends": [0, 0],
        "Stock Splits": [0, 0]
    }, index=dates)
    df.columns = pd.MultiIndex.from_product([['AAPL'], df.columns])
    
    mock_download.return_value = df
    
    # Need to mock to_sql by patching the pandas dataframe since pd.concat returns a new df
    # Actually _download_ticker_history calls df_stacked.to_sql, which uses pandas to_sql method.
    # We can mock pd.DataFrame.to_sql
    with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
        _download_ticker_history(['AAPL'], mock_conn)
        mock_to_sql.assert_called_once()
        args, _kwargs = mock_to_sql.call_args
        assert args[0] == 'ticker_prices'

@patch('database.scripts.update_db.yf.download')
@patch('database.scripts.update_db.time.sleep', MagicMock())
def test_download_ticker_history_exception(mock_download):
    mock_conn = MagicMock()
    mock_download.side_effect = Exception("Download failed")
    # Should handle gracefully and not crash
    _download_ticker_history(['AAPL'], mock_conn)

def test_write_benchmarks():
    mock_conn = MagicMock()
    
    benchmarks = [{
        "benchmark_symbol": "SPY",
        "benchmark_name": "SPDR S&P 500",
        "current_price": 400.0,
        "change_pct": 1.5,
        "dates": ["2023-01-01"],
        "history": [400.0],
        "open": [395.0],
        "high": [405.0],
        "low": [390.0],
        "volume": [1000],
        "tickers": [{
            "ticker_symbol": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "market_cap": 2000000,
            "weight": 0.05
        }]
    }]
    
    _write_benchmarks(benchmarks, mock_conn)
    # Check it called execute for deletes
    assert mock_conn.execute.call_count > 4 # 4 deletes + inserts
    
@patch('database.scripts.update_db.fetch_headlines')
def test_write_headlines(mock_fetch):
    mock_conn = MagicMock()
    mock_fetch.return_value = [{
        "title": "Test Title",
        "publisher": "Test Pub",
        "link": "http://link",
        "summary": "Sum",
        "time": "2023-01-01"
    }]
    
    _write_headlines(mock_conn)
    assert mock_conn.execute.call_count == 2 # Delete + Insert

@patch('database.scripts.update_db.get_engine')
@patch('database.scripts.update_db.os.listdir')
@patch('database.scripts.update_db.open', new_callable=MagicMock)
@patch('database.scripts.update_db.fetch_benchmarks')
@patch('database.scripts.update_db._download_ticker_history')
@patch('database.scripts.update_db._write_benchmarks')
@patch('database.scripts.update_db._write_headlines')
def test_update_database(mock_write_head, mock_write_bench, mock_down, mock_fetch_bench, mock_open, mock_listdir, mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_listdir.return_value = ['table1.sql']
    
    # Setup mock file read
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "CREATE TABLE dummy;"
    mock_open.return_value = mock_file
    
    mock_fetch_bench.return_value = [{
        "benchmark_symbol": "SPY",
        "tickers": [{"ticker_symbol": "AAPL"}]
    }]
    
    update_database()
    
    mock_connect.assert_called_once()
    
    # Verify raw_connection was used and commit was called
    raw_conn_mock = mock_conn.raw_connection.return_value.__enter__.return_value
    cursor_mock = raw_conn_mock.cursor.return_value
    assert cursor_mock.execute.call_count >= 1
    
    mock_down.assert_called_once()
    mock_write_bench.assert_called_once()
    mock_write_head.assert_called_once()
    assert raw_conn_mock.commit.call_count >= 1

@patch('database.scripts.update_db.yf.download')
@patch('database.scripts.update_db.time.sleep', MagicMock())
def test_download_ticker_history_missing(mock_download):
    mock_conn = MagicMock()
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    
    # First call to download returns df with missing AAPL data (NaNs)
    df1 = pd.DataFrame({
        "Close": [float('nan'), float('nan')],
        "Open": [149, 150],
        "High": [155, 156],
        "Low": [145, 146],
        "Volume": [100, 200],
        "Dividends": [0, 0],
        "Stock Splits": [0, 0]
    }, index=dates)
    df1.columns = pd.MultiIndex.from_product([['AAPL'], df1.columns])

    # Second call (retry) returns valid data
    df2 = pd.DataFrame({
        "Close": [150, 151],
        "Open": [149, 150],
        "High": [155, 156],
        "Low": [145, 146],
        "Volume": [100, 200],
        "Dividends": [0, 0],
        "Stock Splits": [0, 0]
    }, index=dates)
    df2.columns = pd.MultiIndex.from_product([['AAPL'], df2.columns])

    mock_download.side_effect = [df1, df2]

    with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
        _download_ticker_history(['AAPL'], mock_conn)
        mock_to_sql.assert_called_once()
        assert mock_download.call_count == 2

@patch('database.scripts.update_db.yf.download')
@patch('database.scripts.update_db.time.sleep', MagicMock())
def test_download_ticker_history_missing_retry_exception(mock_download):
    mock_conn = MagicMock()
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    
    df1 = pd.DataFrame({
        "Close": [float('nan'), float('nan')],
        "Open": [149, 150],
        "High": [155, 156],
        "Low": [145, 146],
        "Volume": [100, 200],
        "Dividends": [0, 0],
        "Stock Splits": [0, 0]
    }, index=dates)
    df1.columns = pd.MultiIndex.from_product([['AAPL'], df1.columns])

    # Retry fails with an exception
    mock_download.side_effect = [df1, Exception("Retry failed")]

    with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
        _download_ticker_history(['AAPL'], mock_conn)
        mock_to_sql.assert_called_once()
        assert mock_download.call_count == 2

@patch('database.scripts.update_db.yf.download')
@patch('database.scripts.update_db.time.sleep', MagicMock())
def test_download_ticker_history_flat_index(mock_download):
    mock_conn = MagicMock()
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    
    # First call returns df without multiindex
    df1 = pd.DataFrame({
        "Close": [150, 151],
        "Open": [149, 150],
        "High": [155, 156],
        "Low": [145, 146],
        "Volume": [100, 200],
        "Dividends": [0, 0],
        "Stock Splits": [0, 0]
    }, index=dates)

    mock_download.return_value = df1

    with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
        _download_ticker_history(['AAPL'], mock_conn)
        mock_to_sql.assert_called_once()

def test_write_benchmarks_missing_fields():
    mock_conn = MagicMock()
    
    benchmarks = [{
        "benchmark_symbol": "SPY",
        "benchmark_name": "SPDR S&P 500",
        "tickers": [{
            "ticker_symbol": "AAPL",
        }]
    }]
    
    _write_benchmarks(benchmarks, mock_conn)
    assert mock_conn.execute.call_count > 4

@patch('database.scripts.update_db.get_engine', MagicMock())
@patch('database.scripts.update_db.os.listdir', MagicMock(return_value=[]))
@patch('database.scripts.update_db.fetch_benchmarks')
@patch('database.scripts.update_db._download_ticker_history', MagicMock())
@patch('database.scripts.update_db._write_benchmarks', MagicMock())
@patch('database.scripts.update_db._write_headlines', MagicMock())
def test_update_database_empty_tickers(mock_fetch_bench):
    mock_fetch_bench.return_value = []
    
    with patch('database.scripts.update_db._download_ticker_history') as mock_down:
        update_database()
        mock_down.assert_not_called()
