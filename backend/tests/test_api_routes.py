import pytest
from pytest import approx
import pandas as pd
import requests
import numpy as np
from backend.apis.routes import sanitize_for_json
from unittest.mock import patch, MagicMock, PropertyMock
from app import app

@pytest.fixture
def client():
    """Sets up a Flask test client for the duration of the tests."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# UI Endpoint Tests
def test_home_page(client):
    """Verifies the UI is served correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Stock & Dividend Forecaster" in response.data


# Search Endpoint Tests
@patch('backend.apis.routes.requests.get')
def test_search_endpoint_success(mock_get, client):
    """Verifies the search endpoint successfully allows all asset types."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "quotes": [
            {"symbol": "AAPL", "shortname": "Apple Inc.", "quoteType": "EQUITY"},
            {"symbol": "VOO", "shortname": "Vanguard S&P 500 ETF", "quoteType": "ETF"},
            {"symbol": "BTC-USD", "shortname": "Bitcoin", "quoteType": "CRYPTOCURRENCY"}
        ]
    }
    mock_get.return_value = mock_response

    response = client.get('/search/AAPL')
    assert response.status_code == 200
    data = response.get_json()
    
    assert len(data) == 3
    assert data[0]["symbol"] == "AAPL"
    assert data[1]["symbol"] == "VOO"
    assert data[2]["symbol"] == "BTC-USD"

@patch('backend.apis.routes.requests.get')
def test_search_endpoint_empty_results(mock_get, client):
    """Verifies the search endpoint handles missing 'quotes' keys gracefully."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"other_key": "value"} # Intentionally missing 'quotes'
    mock_get.return_value = mock_response

    response = client.get('/search/BLAH')
    assert response.status_code == 200
    assert response.get_json() == []

@patch('backend.apis.routes.requests.get')
def test_search_endpoint_exception(mock_get, client):
    """Verifies the search endpoint gracefully handles network timeouts."""
    mock_get.side_effect = requests.exceptions.RequestException("Network timeout")

    response = client.get('/search/ERROR')
    assert response.status_code == 200
    assert response.get_json() == []


# Predict Endpoint Tests
@patch('backend.apis.routes.run_real_time_model')
@patch('backend.apis.routes.get_chart_data')
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_success(mock_ticker, mock_get_chart_data, mock_run_real_time_model, client):
    """Verifies the API safely processes successful standard equity predictions."""
    mock_ml_data = {
        "anchor_date": pd.Timestamp("2026-06-12"),
        "today_close": 150.0,
        "price_direction": 1,
        "price_conf": 0.855,
        "forecasted_close": 152.0,
        "next_dividend_date": pd.NaT,
        "div_direction": None,
        "div_conf": 0.0,
        "forecasted_div": "N/A",
        "extended_forecasts": {},
        "chart_future_dates": [], "chart_future_prices": [], "chart_future_upper": [], "chart_future_lower": [],
        "train_fit_dates": [], "train_fit_prices": [],
        "div_extended_forecasts": {}, "div_future_dates": [], "div_future_amounts": [], "div_future_upper": [], "div_future_lower": [],
        "train_fit_div_dates": [], "train_fit_div_amounts": []
    }
    mock_run_real_time_model.return_value = mock_ml_data
    mock_get_chart_data.return_value = {"dates": ["2026-06-12"], "prices": [148.0], "dividend_dates": [], "dividend_amounts": []}

    mock_instance = MagicMock()
    mock_instance.info = {"longName": "Apple Inc.", "quoteType": "EQUITY", "recommendationKey": "buy"}
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.fetch_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.5, {"positive": ["Good"]})
        
        response = client.get('/predict/AAPL')
        assert response.status_code == 200
        
        json_data = response.get_json()
        assert json_data['Ticker'] == 'AAPL'
        assert 'Chart_History' in json_data
        assert json_data['Chart_History']['prices'][0] == approx(148.0)
        assert json_data['Stock_Grade'] in ["A+", "A", "B", "C", "D", "F"]

@patch('backend.apis.routes.run_real_time_model')
@patch('backend.apis.routes.get_chart_data')
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_etf_success(mock_ticker, mock_get_chart_data, mock_run_real_time_model, client):
    """Verifies the API accurately parses ETF Holdings, Sectors, and formats Downward/Dividend logic."""
    mock_ml_data = {
        "anchor_date": pd.Timestamp("2026-06-12"),
        "today_close": 150.0,
        "price_direction": 0,
        "price_conf": 0.60,
        "forecasted_close": 148.0,
        "next_dividend_date": pd.Timestamp("2026-07-01"),
        "div_direction": 1,
        "div_conf": 0.90,
        "forecasted_div": 1.5,
        "extended_forecasts": {},
        "chart_future_dates": [], "chart_future_prices": [], "chart_future_upper": [], "chart_future_lower": [],
        "train_fit_dates": [], "train_fit_prices": [],
        "div_extended_forecasts": {}, "div_future_dates": [], "div_future_amounts": [], "div_future_upper": [], "div_future_lower": [],
        "train_fit_div_dates": [], "train_fit_div_amounts": []
    }
    mock_run_real_time_model.return_value = mock_ml_data
    mock_get_chart_data.return_value = {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}

    mock_instance = MagicMock()
    mock_instance.info = {"quoteType": "ETF"}
    
    # Safely mock the yfinance funds_data payload
    mock_funds_data = MagicMock()
    holdings_df = pd.DataFrame({"weight": [0.07], "name": ["Apple Inc."]}, index=["AAPL"])
    mock_funds_data.top_holdings = holdings_df
    mock_funds_data.sector_weightings = {"technology": 0.40, "realestate": 0.10}
    mock_instance.funds_data = mock_funds_data
    
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.fetch_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.5, {"positive": ["Good"]})
        
        response = client.get('/predict/VOO')
        assert response.status_code == 200
        json_data = response.get_json()
        
        # Verify formatting triggers 
        assert json_data['Price_Predicted'] == "Down"
        assert json_data['Div_Predicted'] == "Up"
        assert json_data['Next_Dividend_Date'] == "2026-07-01"

        # Verify ETF Sector & Holdings logic parsed successfully
        reasoning = json_data.get('AI_Reasoning', {})
        assert 'etf_holdings' in reasoning
        assert len(reasoning['etf_holdings']) > 0
        assert reasoning['etf_holdings'][0]['symbol'] == 'AAPL'
        assert 'etf_sectors' in reasoning
        assert len(reasoning['etf_sectors']) > 0

@patch('backend.apis.routes.run_real_time_model')
@patch('backend.apis.routes.get_chart_data')
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_info_exception(mock_ticker, mock_get_chart_data, mock_run_real_time_model, client):
    """Verifies the API doesn't crash if Yahoo Finance's info dictionary fails to load."""
    mock_run_real_time_model.return_value = {
        "anchor_date": pd.Timestamp("2026-06-12"), "today_close": 150.0, "price_direction": 1, "price_conf": 0.8, "forecasted_close": 152.0, "next_dividend_date": pd.NaT, "div_direction": 0, "div_conf": 0.0, "forecasted_div": "N/A", "extended_forecasts": {}, "chart_future_dates": [], "chart_future_prices": [], "chart_future_upper": [], "chart_future_lower": [], "train_fit_dates": [], "train_fit_prices": [], "div_extended_forecasts": {}, "div_future_dates": [], "div_future_amounts": [], "div_future_upper": [], "div_future_lower": [], "train_fit_div_dates": [], "train_fit_div_amounts": []
    }
    mock_get_chart_data.return_value = {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}

    mock_instance = MagicMock()
    # Simulate property access throwing an error
    type(mock_instance).info = PropertyMock(side_effect=Exception("API limit reached"))
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.fetch_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.0, {"neutral": "No news"})
        response = client.get('/predict/AAPL')
        
        # It should survive the Exception and default to an empty info dict
        assert response.status_code == 200
        assert response.get_json()['Ticker'] == 'AAPL'

@patch('backend.apis.routes.run_real_time_model')
@patch('backend.apis.routes.get_chart_data')
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_etf_parsing_exception(mock_ticker, mock_get_chart_data, mock_run_real_time_model, client):
    """Verifies that an error inside the ETF holdings parser doesn't crash the pipeline."""
    mock_run_real_time_model.return_value = {
        "anchor_date": pd.Timestamp("2026-06-12"), "today_close": 150.0, "price_direction": 1, "price_conf": 0.8, "forecasted_close": 152.0, "next_dividend_date": pd.NaT, "div_direction": -1, "div_conf": 0.0, "forecasted_div": "N/A", "extended_forecasts": {}, "chart_future_dates": [], "chart_future_prices": [], "chart_future_upper": [], "chart_future_lower": [], "train_fit_dates": [], "train_fit_prices": [], "div_extended_forecasts": {}, "div_future_dates": [], "div_future_amounts": [], "div_future_upper": [], "div_future_lower": [], "train_fit_div_dates": [], "train_fit_div_amounts": []
    }
    mock_get_chart_data.return_value = {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}

    mock_instance = MagicMock()
    mock_instance.info = {"quoteType": "MUTUALFUND"}
    
    # Simulating a crash when trying to access funds_data
    type(mock_instance).funds_data = PropertyMock(side_effect=Exception("Corrupt fund data"))
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.fetch_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.0, {"neutral": "No news"})
        response = client.get('/predict/FXAIX')
        
        assert response.status_code == 200
        assert response.get_json()['Ticker'] == 'FXAIX'

@patch('backend.apis.routes.run_real_time_model')
def test_predict_endpoint_not_found(mock_run_real_time_model, client):
    """Verifies the API handles invalid tickers/insufficient data returning a 404."""
    mock_run_real_time_model.return_value = None

    response = client.get('/predict/INVALID')
    assert response.status_code == 404
    assert "error" in response.get_json()

@patch('backend.apis.routes.run_real_time_model')
def test_predict_endpoint_internal_error(mock_run_real_time_model, client):
    """Verifies the API catches catastrophic pipeline failures returning a 500."""
    mock_run_real_time_model.side_effect = Exception("Catastrophic ML failure")

    response = client.get('/predict/CRASH')
    assert response.status_code == 500
    assert "internal server error" in response.get_json()["error"].lower()


# Sanitizer Tests
def test_sanitize_clean_data():
    clean_data = {
        "Ticker": "AAPL",
        "Forecasted_Close": 150.50,
        "Chart_History": [148.0, 149.0, 150.0],
        "Is_Valid": True
    }
    result = sanitize_for_json(clean_data)
    assert result == clean_data

def test_sanitize_invalid_floats():
    dirty_data = [
        float('nan'), float('inf'), float('-inf'), 
        np.nan, np.inf, -np.inf
    ]
    result = sanitize_for_json(dirty_data)
    assert all(item is None for item in result)

def test_sanitize_nested_structures():
    nested_dirty_data = {
        "Ticker": "TSLA",
        "Metrics": {
            "PE_Ratio": float('nan'),
            "Moving_Averages": [200.5, float('inf'), 198.2]
        },
        "Flags": [float('-inf'), {"bad_val": np.nan}]
    }
    
    expected_clean_data = {
        "Ticker": "TSLA",
        "Metrics": {
            "PE_Ratio": None,
            "Moving_Averages": [200.5, None, 198.2]
        },
        "Flags": [None, {"bad_val": None}]
    }
    
    result = sanitize_for_json(nested_dirty_data)
    assert result == expected_clean_data

def test_sanitize_pandas_types():
    pd_data = {
        "Missing_Date": pd.NaT, 
        "Missing_Value": pd.NA
    }
    result = sanitize_for_json(pd_data)
    
    assert result["Missing_Date"] is None
    assert result["Missing_Value"] is None