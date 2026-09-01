from unittest.mock import patch

import pandas as pd

from services.screener_service import (
    get_screener_dashboard_data,
    process_custom_scans_by_benchmark,
)


def test_process_custom_scans_by_benchmark():
    # Construct a MultiIndex dataframe like yfinance returns for multiple tickers
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    
    # AAPL
    aapl_df = pd.DataFrame({
        "Close": [150 + i for i in range(20)],
        "High": [151 + i for i in range(20)],
        "Low": [149 + i for i in range(20)],
        "Volume": [1000] * 20,
        "Dividends": [0] * 20
    }, index=dates)
    aapl_df.loc[dates[-1], "Dividends"] = 1.0 # Add a dividend
    
    # MSFT
    msft_df = pd.DataFrame({
        "Close": [250 - i for i in range(20)],
        "High": [251 - i for i in range(20)],
        "Low": [249 - i for i in range(20)],
        "Volume": [2000] * 20,
    }, index=dates)
    
    # Create multiindex
    aapl_df.columns = pd.MultiIndex.from_product([['AAPL'], aapl_df.columns])
    msft_df.columns = pd.MultiIndex.from_product([['MSFT'], msft_df.columns])
    
    combined_data = pd.concat([aapl_df, msft_df], axis=1)
    
    benchmarks_list = [
        {
            "name": "Tech Benchmark",
            "constituents": [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "MSFT", "name": "Microsoft Corp."}
            ]
        }
    ]
    
    movers, scans = process_custom_scans_by_benchmark(combined_data, benchmarks_list)
    
    assert "Tech Benchmark" in movers
    assert "Tech Benchmark" in scans
    
    tech_movers = movers["Tech Benchmark"]
    assert len(tech_movers["day_gainers"]) == 1
    assert tech_movers["day_gainers"][0]["symbol"] == "AAPL" # Since AAPL goes up
    
    tech_scans = scans["Tech Benchmark"]
    assert len(tech_scans["new_high"]) >= 0
    assert len(tech_scans["biggest_dividends"]) == 1
    assert tech_scans["biggest_dividends"][0]["symbol"] == "AAPL"
    assert tech_scans["biggest_dividends"][0]["last_dividend_date"] == dates[-1].strftime('%Y-%m-%d')

@patch('services.screener_service.get_latest_benchmarks')
@patch('services.screener_service.get_latest_headlines')
@patch('services.screener_service.get_historical_prices_df')
def test_get_screener_dashboard_data(mock_prices, mock_headlines, mock_benchmarks):
    mock_benchmarks.return_value = [{"name": "Test Benchmark", "constituents": []}]
    mock_headlines.return_value = [{"title": "News"}]
    
    # Empty data
    mock_prices.return_value = pd.DataFrame()
    
    res = get_screener_dashboard_data()
    assert res["benchmarks"][0]["name"] == "Test Benchmark"
    assert res["headlines"][0]["title"] == "News"
    assert res["market_movers"] == {}
    
    # With data
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "Close": [150 + i for i in range(20)],
        "High": [151 + i for i in range(20)],
        "Low": [149 + i for i in range(20)],
        "Volume": [1000] * 20,
    }, index=dates)
    df.columns = pd.MultiIndex.from_product([['AAPL'], df.columns])
    mock_prices.return_value = df
    
    res2 = get_screener_dashboard_data()
    assert res2["custom_scans"] is not None
