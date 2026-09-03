from unittest.mock import MagicMock, patch

import pandas as pd

from services.external_data_service import (
    fetch_benchmark_tickers,
    fetch_benchmarks,
    fetch_headlines,
)


@patch('services.external_data_service.requests.get')
def test_fetch_benchmark_tickers(mock_get):
    mock_response = MagicMock()
    # Mock wikipedia table html
    html = """
    <html>
        <body>
            <table>
                <tr><th>Symbol</th><th>Sector</th></tr>
                <tr><td>AAPL</td><td>Technology</td></tr>
                <tr><td>MSFT</td><td>Technology</td></tr>
            </table>
        </body>
    </html>
    """
    mock_response.text = html
    mock_get.return_value = mock_response
    
    tickers = fetch_benchmark_tickers()
    assert "S&P 500" in tickers
    assert len(tickers["S&P 500"]) == 2
    assert tickers["S&P 500"][0]["ticker_symbol"] == "AAPL"

@patch('services.external_data_service.fetch_benchmark_tickers')
@patch('services.external_data_service.yf.download')
@patch('services.external_data_service.requests.Session')
@patch('services.external_data_service.yf.Ticker')
def test_fetch_benchmarks(mock_ticker, mock_session, mock_download, mock_fetch_tickers):
    # Mock the fetched tickers
    mock_fetch_tickers.return_value = {
        "Dow 30": [{"ticker_symbol": "AAPL", "sector": "Tech"}],
        "Nasdaq 100": [],
        "S&P 500": [],
        "Russell 1000": []
    }
    
    # Mock yfinance download for benchmarks
    dates = pd.date_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "Close": [100]*10,
        "Open": [100]*10,
        "High": [100]*10,
        "Low": [100]*10,
        "Volume": [100]*10
    }, index=dates)
    df.columns = pd.MultiIndex.from_product([['^DJI'], df.columns])
    mock_download.return_value = df
    
    # Mock requests session
    mock_session_instance = MagicMock()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        'quoteResponse': {
            'result': [
                {'symbol': 'AAPL', 'shortName': 'Apple', 'marketCap': 10000, 'regularMarketPrice': 150}
            ]
        }
    }
    mock_session_instance.get.return_value = mock_res
    mock_session.return_value = mock_session_instance
    
    # Mock yfinance fast_info for fallback
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.fast_info = {'market_cap': 10000}
    mock_ticker.return_value = mock_ticker_instance
    
    res = fetch_benchmarks()
    assert len(res) == 1
    assert res[0]["benchmark_symbol"] == "^DJI"
    assert res[0]["tickers"][0]["ticker_symbol"] == "AAPL"

@patch('services.external_data_service.yf.Search')
def test_fetch_headlines(mock_search):
    mock_instance = MagicMock()
    mock_instance.news = [
        {
            "title": "Test News",
            "publisher": "Test Provider",
            "link": "http://test.com",
            "providerPublishTime": 1672574400,
            "summary": "Summary"
        },
        {} # missing fields
    ]
    mock_search.return_value = mock_instance
    
    headlines = fetch_headlines()
    assert len(headlines) == 2
    assert headlines[0]["title"] == "Test News"
    assert headlines[0]["link"] == "http://test.com"
