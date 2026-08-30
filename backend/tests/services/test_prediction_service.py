import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
from services.prediction_service import (
    sanitize_for_json,
    build_frontend_payload,
    fetch_company_fundamentals,
    resolve_search_query,
    run_prediction_pipeline
)

def test_sanitize_for_json():
    obj = {
        "a": float('inf'),
        "b": float('-inf'),
        "c": float('nan'),
        "d": [float('inf')],
        "e": {"f": float('nan')},
        "g": 10.5
    }
    sanitized = sanitize_for_json(obj)
    assert sanitized["a"] is None
    assert sanitized["b"] is None
    assert sanitized["c"] is None
    assert sanitized["d"][0] is None
    assert sanitized["e"]["f"] is None
    assert sanitized["g"] == 10.5

def test_build_frontend_payload():
    ticker = "AAPL"
    info = {"longName": "Apple Inc."}
    raw_ml_data = {
        "anchor_date": pd.Timestamp("2023-01-01"),
        "next_dividend_date": pd.NaT,
        "today_close": 150.0,
        "price_forecasts": {},
        "chart_future_dates": [],
        "chart_future_prices": [],
        "chart_future_upper": [],
        "chart_future_lower": [],
        "train_fit_dates": [],
        "train_fit_prices": [],
        "div_forecasts": {},
        "div_future_dates": [],
        "div_future_amounts": [],
        "div_future_upper": [],
        "div_future_lower": [],
        "train_fit_div_dates": [],
        "train_fit_div_amounts": [],
    }
    nlp_data = {
        "grade": "A",
        "sentiment": "Bullish",
        "reasoning": {}
    }
    
    payload = build_frontend_payload(ticker, raw_ml_data, {}, nlp_data, info, False)
    assert payload["Company_Name"] == "Apple Inc."
    assert payload["Next_Dividend_Date"] == "N/A"
    assert payload["Stock_Grade"] == "A"

@patch('services.prediction_service.yf.Ticker')
def test_fetch_company_fundamentals(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.info = {"quoteType": "ETF"}
    
    # Mock funds data
    mock_funds = MagicMock()
    
    mock_holdings = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "name": ["Apple", "Microsoft"],
        "weight": [0.05, 0.04]
    })
    mock_holdings.index = ["AAPL", "MSFT"]
    
    mock_sectors = {"technology": 0.5, "financialServices": 0.2}
    
    mock_funds.top_holdings = mock_holdings
    mock_funds.sector_weightings = mock_sectors
    
    mock_instance.funds_data = mock_funds
    mock_ticker.return_value = mock_instance
    
    info, is_fund, is_crypto, holdings, sectors = fetch_company_fundamentals("SPY")
    assert is_fund is True
    assert is_crypto is False
    assert len(holdings) == 2
    assert len(sectors) == 2

@patch('services.prediction_service.requests.get')
def test_resolve_search_query(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'quotes': [{'symbol': 'AAPL'}]
    }
    mock_get.return_value = mock_response
    assert resolve_search_query("Apple") == "AAPL"

@patch('services.prediction_service.fetch_company_fundamentals')
@patch('services.prediction_service.fetch_data')
@patch('services.prediction_service.run_price_prediction')
@patch('services.prediction_service.run_dividend_prediction')
@patch('services.prediction_service.analyze_news_sentiment')
@patch('services.prediction_service.calculate_asset_grade')
def test_run_prediction_pipeline(mock_grade, mock_sentiment, mock_div, mock_price, mock_data, mock_fundamentals):
    mock_fundamentals.return_value = ({"longName": "Apple Inc."}, False, False, [], [])
    
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame({"Close": [150]*100, "Volume": [1000]*100}, index=dates)
    mock_data.return_value = (df, None)
    
    mock_price.return_value = {
        "p_anchors": {1: 150},
        "p_lower": {1: 140},
        "p_upper": {1: 160},
        "price_forecasts": {},
        "train_fit_dates": [],
        "train_fit_prices": [],
        "has_enough_price_data": True
    }
    
    mock_div.return_value = {
        "d_anchors": {1: 1},
        "d_lower": {1: 0.9},
        "d_upper": {1: 1.1},
        "div_forecasts": {},
        "train_fit_div_dates": [],
        "train_fit_div_amounts": [],
        "next_dividend_date": pd.NaT,
        "has_enough_div_data": False,
        "avg_days_between": 90
    }
    
    mock_sentiment.return_value = (0.5, {})
    mock_grade.return_value = ("A", "Bullish", {})
    
    pipeline = run_prediction_pipeline("AAPL")
    results = list(pipeline)
    
    assert len(results) == 6
    assert results[-1]["status"] == "complete"
    assert results[-1]["result"]["Company_Name"] == "Apple Inc."
    
    # Test error case (no data)
    mock_data.return_value = (None, None)
    pipeline_err = run_prediction_pipeline("INVALID")
    results_err = list(pipeline_err)
    assert len(results_err) == 2
    assert results_err[-1]["status"] == "error"
