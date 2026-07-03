import pytest
from pytest import approx
import pandas as pd
import requests
import numpy as np
import json
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

@patch('backend.apis.routes.requests.get')
def test_search_endpoint_empty_results(mock_get, client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"other_key": "value"}
    mock_get.return_value = mock_response

    response = client.get('/search/BLAH')
    assert response.status_code == 200
    assert response.get_json() == []

@patch('backend.apis.routes.requests.get')
def test_search_endpoint_exception(mock_get, client):
    mock_get.side_effect = requests.exceptions.RequestException("Network timeout")
    response = client.get('/search/ERROR')
    assert response.status_code == 200
    assert response.get_json() == []

# Helper fixtures for mocking predictions
import functools

def patch_predictions(func):
    @patch('backend.apis.routes.fetch_data')
    @patch('backend.apis.routes.run_price_prediction')
    @patch('backend.apis.routes.run_dividend_prediction')
    @patch('backend.apis.routes.generate_future_chart_data')
    @patch('backend.apis.routes.get_chart_data')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, is_etf=False, is_error=False, not_found=False):
    if not_found:
        mock_fetch.return_value = (None, None)
        return
        
    if is_error:
        mock_fetch.side_effect = Exception("Catastrophic ML failure")
        return
        
    mock_fetch.return_value = (pd.DataFrame({"Close": [150.0]}, index=[pd.Timestamp("2026-06-12")]), None)
    
    mock_price.return_value = {
        "price_forecasts": {"Next_Day": {"Direction": "Down" if is_etf else "Up", "Direction_Confidence": 60.0 if is_etf else 85.5, "Amount": 148.0 if is_etf else 152.0, "Amount_Confidence": 50.0 if is_etf else 90.0}},
        "p_anchors": {}, "p_lower": {}, "p_upper": {},
        "train_fit_dates": [], "train_fit_prices": [],
        "has_enough_price_data": True
    }
    
    mock_div.return_value = {
        "div_forecasts": {"Next_Payout": {"Direction": "Up", "Direction_Confidence": 90.0, "Amount": 1.5, "Amount_Confidence": 80.0}} if is_etf else {},
        "d_anchors": {}, "d_lower": {}, "d_upper": {},
        "train_fit_div_dates": [], "train_fit_div_amounts": [],
        "next_dividend_date": pd.Timestamp("2026-07-01") if is_etf else pd.NaT,
        "avg_days_between": 90,
        "has_enough_div_data": is_etf
    }
    
    mock_generate_chart.return_value = ([], [], [], [])
    mock_get_chart.return_value = {"dates": ["2026-06-12"], "prices": [148.0], "dividend_dates": [], "dividend_amounts": []}


# Predict Endpoint Tests
@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_success(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart)
    mock_instance = MagicMock()
    mock_instance.info = {"longName": "Apple Inc.", "quoteType": "EQUITY", "recommendationKey": "buy"}
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.analyze_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.5, {"positive": ["Good"]})
        
        response = client.get('/predict/AAPL')
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['Ticker'] == 'AAPL'
        assert json_data['Chart_History']['prices'][0] == approx(148.0)

@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_etf_success(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, is_etf=True)
    mock_instance = MagicMock()
    mock_instance.info = {"quoteType": "ETF"}
    
    mock_funds_data = MagicMock()
    mock_funds_data.top_holdings = pd.DataFrame({"weight": [0.07], "name": ["Apple Inc."]}, index=["AAPL"])
    mock_funds_data.sector_weightings = {"technology": 0.40, "realestate": 0.10}
    mock_instance.funds_data = mock_funds_data
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.analyze_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.5, {"positive": ["Good"]})
        response = client.get('/predict/VOO')
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['Price_Forecasts']['Next_Day']['Direction'] == "Down"
        assert json_data['Div_Forecasts']['Next_Payout']['Direction'] == "Up"

@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_info_exception(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart)
    mock_instance = MagicMock()
    type(mock_instance).info = PropertyMock(side_effect=Exception("API limit reached"))
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.analyze_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.0, {"neutral": "No news"})
        response = client.get('/predict/AAPL')
        assert response.status_code == 200
        assert response.get_json()['Ticker'] == 'AAPL'

@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_endpoint_etf_parsing_exception(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart)
    mock_instance = MagicMock()
    mock_instance.info = {"quoteType": "MUTUALFUND"}
    type(mock_instance).funds_data = PropertyMock(side_effect=Exception("Corrupt fund data"))
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.analyze_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.0, {"neutral": "No news"})
        response = client.get('/predict/FXAIX')
        assert response.status_code == 200
        assert response.get_json()['Ticker'] == 'FXAIX'

@patch_predictions
def test_predict_endpoint_not_found(mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, not_found=True)
    response = client.get('/predict/INVALID')
    assert response.status_code == 404

@patch_predictions
def test_predict_endpoint_internal_error(mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, is_error=True)
    response = client.get('/predict/CRASH')
    assert response.status_code == 500


# Sanitizer Tests
def test_sanitize_clean_data():
    clean_data = {"Ticker": "AAPL", "Forecasted_Close": 150.50, "Is_Valid": True}
    assert sanitize_for_json(clean_data) == clean_data

def test_sanitize_invalid_floats():
    dirty_data = [float('nan'), float('inf'), float('-inf'), np.nan, np.inf, -np.inf]
    result = sanitize_for_json(dirty_data)
    assert all(item is None for item in result)

def test_sanitize_pandas_types():
    pd_data = {
        "Missing_Date": pd.NaT, 
        "Missing_Value": pd.NA
    }
    result = sanitize_for_json(pd_data)
    
    assert result["Missing_Date"] is None
    assert result["Missing_Value"] is None

def test_sanitize_single_nat():
    assert sanitize_for_json(pd.NaT) is None

from backend.apis.routes import build_frontend_payload
def test_build_frontend_payload_info_exception():
    raw_ml_data = {
        "anchor_date": pd.Timestamp("2026-06-12"),
        "next_dividend_date": pd.NaT,
        "today_close": 100.0,
        "price_forecasts": {},
        "div_forecasts": {},
        "chart_future_dates": [],
        "chart_future_prices": [],
        "chart_future_upper": [],
        "chart_future_lower": [],
        "train_fit_dates": [],
        "train_fit_prices": [],
        "div_future_dates": [],
        "div_future_amounts": [],
        "div_future_upper": [],
        "div_future_lower": [],
        "train_fit_div_dates": [],
        "train_fit_div_amounts": []
    }
    nlp_data = {"grade": "A", "sentiment": 100, "reasoning": ""}
    # Passing a string instead of dict to force an exception when it calls .get()
    payload = build_frontend_payload("AAPL", raw_ml_data, {}, nlp_data, "invalid_info_type")
    assert payload["Company_Name"] == "AAPL"

def test_build_frontend_payload_crypto():
    raw_ml_data = {
        "anchor_date": pd.Timestamp("2026-06-12"),
        "next_dividend_date": pd.NaT,
        "today_close": 100.0,
        "price_forecasts": {},
        "div_forecasts": {},
        "chart_future_dates": [],
        "chart_future_prices": [],
        "chart_future_upper": [],
        "chart_future_lower": [],
        "train_fit_dates": [],
        "train_fit_prices": [],
        "div_future_dates": [],
        "div_future_amounts": [],
        "div_future_upper": [],
        "div_future_lower": [],
        "train_fit_div_dates": [],
        "train_fit_div_amounts": []
    }
    nlp_data = {"grade": "A", "sentiment": 100, "reasoning": ""}
    payload = build_frontend_payload("BTC-USD", raw_ml_data, {}, nlp_data, {}, is_crypto=True)
    assert payload["Ticker"] == "BTC-USD"

# Stream Endpoint Tests
@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_stream_endpoint_success(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart)
    mock_instance = MagicMock()
    mock_instance.info = {"longName": "Apple Inc.", "quoteType": "EQUITY", "recommendationKey": "buy"}
    mock_ticker.return_value = mock_instance
    
    with patch('backend.apis.routes.analyze_news_sentiment') as mock_sentiment:
        mock_sentiment.return_value = (0.5, {"positive": ["Good"]})
        
        response = client.get('/predict_stream/AAPL')
        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'
        
        text = response.get_data(as_text=True)
        events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
        
        assert len(events) > 0
        final_event = events[-1]
        assert final_event['status'] == 'complete'
        assert final_event['result']['Ticker'] == 'AAPL'

@patch_predictions
def test_predict_stream_endpoint_not_found(mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, not_found=True)
    response = client.get('/predict_stream/INVALID')
    text = response.get_data(as_text=True)
    events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
    
    final_event = events[-1]
    assert final_event['status'] == 'error'
    assert "Invalid ticker" in final_event['error']

@patch_predictions
def test_predict_stream_endpoint_internal_error(mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart, is_error=True)
    response = client.get('/predict_stream/CRASH')
    text = response.get_data(as_text=True)
    events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
    
    final_event = events[-1]
    assert final_event['status'] == 'error'
    assert "internal server error" in final_event['error']

@patch_predictions
@patch('backend.apis.routes.yf.Ticker')
def test_predict_stream_endpoint_info_exception(mock_ticker, mock_get_chart, mock_generate_chart, mock_div, mock_price, mock_fetch, client):
    setup_mocks(mock_fetch, mock_price, mock_div, mock_generate_chart, mock_get_chart)
    mock_fetch.side_effect = Exception("Crash")
    
    mock_instance = MagicMock()
    type(mock_instance).info = PropertyMock(side_effect=Exception("API limit reached"))
    mock_ticker.return_value = mock_instance
    
    response = client.get('/predict_stream/INFOEXCEPT')
    text = response.get_data(as_text=True)
    events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
    assert events[-1]['status'] == 'error'

def test_predict_stream_cache(client):
    from backend.apis.routes import forecast_cache
    forecast_cache['CACHED'] = {"Ticker": "CACHED", "fake": "data"}
    
    response = client.get('/predict_stream/CACHED')
    text = response.get_data(as_text=True)
    events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
    
    assert len(events) == 1
    assert events[0]['status'] == 'complete'
    assert events[0]['result']['Ticker'] == 'CACHED'