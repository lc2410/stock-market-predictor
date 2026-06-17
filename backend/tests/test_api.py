import pytest
from pytest import approx
import pandas as pd
import requests
import numpy as np
from backend.apis.routes import sanitize_for_json
from unittest.mock import patch, MagicMock
from app import app

@pytest.fixture
def client():
    """Sets up a Flask test client for the duration of the tests."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# UI endpoint tests
def test_home_page(client):
    """Verifies the UI is served correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Stock & Dividend Forecaster" in response.data


# search endpoint tests
@patch('backend.apis.routes.requests.get')
def test_search_endpoint_success(mock_get, client):
    """Verifies the search endpoint successfully filters only Equity and ETFs."""
    # Mocking the Yahoo Finance response to include mixed asset types
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
    
    # Should only return the EQUITY and ETF, ignoring the Crypto
    assert len(data) == 2
    assert data[0]["symbol"] == "AAPL"
    assert data[1]["symbol"] == "VOO"

@patch('backend.apis.routes.requests.get')
def test_search_endpoint_exception(mock_get, client):
    """Verifies the search endpoint gracefully handles network timeouts."""
    # Simulating a network failure when reaching out to Yahoo Finance
    mock_get.side_effect = requests.exceptions.RequestException("Network timeout")

    response = client.get('/search/ERROR')
    
    # Should safely return an empty list rather than crashing
    assert response.status_code == 200
    assert response.get_json() == []

# predict endpoint tests
@patch('backend.apis.routes.run_real_time_model')
@patch('backend.apis.routes.get_chart_data')
def test_predict_endpoint_success(mock_get_chart_data, mock_run_real_time_model, client):
    """Verifies the API safely processes successful predictions."""
    mock_df = pd.DataFrame([{
        "Ticker": "AAPL", "Company_Name": "Apple Inc.", "Next_Trading_Day": "2026-06-15",
        "Price_Predicted": "Up", "Price_Confidence (%)": 85.5, "Forecasted_Close": 150.0,
        "Next_Dividend_Date": "N/A", "Div_Predicted": "N/A", "Div_Confidence (%)": "N/A",
        "Forecasted_Dividend": "N/A", "Forecasted_Yield (%)": "N/A", "Extended_Forecasts": {}, 
        "Chart_Future_Dates": [], "Chart_Future_Prices": [], "Chart_Future_Upper": [], 
        "Chart_Future_Lower": [], "Train_Fit_Dates": [], "Train_Fit_Prices": [], 
        "Div_Extended_Forecasts": {}, "Div_Future_Dates": [], "Div_Future_Amounts": [], 
        "Div_Future_Upper": [], "Div_Future_Lower": []
    }])
    mock_run_real_time_model.return_value = mock_df
    mock_get_chart_data.return_value = {"dates": ["2026-06-12"], "prices": [148.0]}

    response = client.get('/predict/AAPL')
    assert response.status_code == 200
    
    json_data = response.get_json()
    assert json_data['Ticker'] == 'AAPL'
    assert 'Chart_History' in json_data
    assert json_data['Chart_History']['prices'][0] == approx(148.0)

@patch('backend.apis.routes.run_real_time_model')
def test_predict_endpoint_not_found(mock_run_real_time_model, client):
    """Verifies the API handles invalid tickers/insufficient data returning a 404."""
    # Simulating a failure to fetch data resulting in None
    mock_run_real_time_model.return_value = None

    response = client.get('/predict/INVALID')
    assert response.status_code == 404
    assert "error" in response.get_json()

@patch('backend.apis.routes.run_real_time_model')
def test_predict_endpoint_internal_error(mock_run_real_time_model, client):
    """Verifies the API catches catastrophic pipeline failures returning a 500."""
    # Simulating a severe math or memory failure in the ML pipeline
    mock_run_real_time_model.side_effect = Exception("Catastrophic ML failure")

    response = client.get('/predict/CRASH')
    assert response.status_code == 500
    assert "internal server error" in response.get_json()["error"].lower()

# sanitizer tests
def test_sanitize_clean_data():
    """Verifies that standard data types (strings, ints, valid floats, booleans) pass through untouched."""
    clean_data = {
        "Ticker": "AAPL",
        "Forecasted_Close": 150.50,
        "Chart_History": [148.0, 149.0, 150.0],
        "Is_Valid": True
    }
    result = sanitize_for_json(clean_data)
    assert result == clean_data

def test_sanitize_invalid_floats():
    """Verifies that Python and Numpy versions of NaN, Infinity, and -Infinity are correctly converted to None."""
    dirty_data = [
        float('nan'), float('inf'), float('-inf'), 
        np.nan, np.inf, -np.inf
    ]
    result = sanitize_for_json(dirty_data)
    
    # Every single item in the array should have been neutralized to `None`
    assert all(item is None for item in result)

def test_sanitize_nested_structures():
    """Verifies the sanitizer successfully recurses through deeply nested dictionaries and lists."""
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
    """Verifies that Pandas-specific missing types (like NaT for missing dates) are handled correctly."""
    pd_data = {
        "Missing_Date": pd.NaT, 
        "Missing_Value": pd.NA
    }
    result = sanitize_for_json(pd_data)
    
    assert result["Missing_Date"] is None
    assert result["Missing_Value"] is None