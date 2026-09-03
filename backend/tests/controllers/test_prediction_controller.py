"""Tests for the prediction controller endpoints and service utilities."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app import app
from services.prediction_service import build_frontend_payload, sanitize_for_json


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('controllers.prediction_controller.run_prediction_pipeline')
def test_predict_endpoint_success(mock_pipeline, client):
    """Tests that a valid ticker returns a 200 response with prediction data."""
    mock_pipeline.return_value = [
        {"status": "processing", "step": "Gathering", "progress": 15},
        {"status": "complete", "result": {"Ticker": "AAPL", "Price": 150.0}}
    ]
    response = client.get('/api/predict/AAPL')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['Ticker'] == 'AAPL'
    assert json_data['Price'] == 150.0

@patch('controllers.prediction_controller.run_prediction_pipeline')
def test_predict_endpoint_not_found(mock_pipeline, client):
    """Tests that an invalid ticker returns a 404 error."""
    mock_pipeline.return_value = [
        {"status": "error", "error": "Invalid ticker"}
    ]
    response = client.get('/api/predict/INVALID')
    assert response.status_code == 404

@patch('controllers.prediction_controller.run_prediction_pipeline')
def test_predict_endpoint_internal_error(mock_pipeline, client):
    """Tests that an unhandled exception returns a 500 error."""
    mock_pipeline.side_effect = Exception("Crash")
    response = client.get('/api/predict/CRASH')
    assert response.status_code == 500

@patch('controllers.prediction_controller.resolve_search_query')
@patch('controllers.prediction_controller.run_prediction_pipeline')
def test_predict_stream_endpoint_success(mock_pipeline, mock_resolve, client):
    """Tests that the SSE streaming endpoint yields progress events and a final result."""
    mock_resolve.return_value = "AAPL"
    mock_pipeline.return_value = [
        {"status": "processing", "step": "Gathering", "progress": 15},
        {"status": "complete", "result": {"Ticker": "AAPL"}}
    ]
    response = client.get('/api/predict_stream/AAPL')
    assert response.status_code == 200
    assert response.mimetype == 'text/event-stream'
    
    text = response.get_data(as_text=True)
    events = [json.loads(line.replace('data: ', '')) for line in text.split('\n\n') if line.startswith('data: ')]
    
    assert len(events) == 2
    assert events[-1]['status'] == 'complete'
    assert events[-1]['result']['Ticker'] == 'AAPL'

def test_sanitize_clean_data():
    """Tests that clean data passes through sanitization unchanged."""
    clean_data = {"Ticker": "AAPL", "Forecasted_Close": 150.50, "Is_Valid": True}
    assert sanitize_for_json(clean_data) == clean_data

def test_sanitize_invalid_floats():
    """Tests that NaN and Infinity values are replaced with None."""
    dirty_data = [float('nan'), float('inf'), float('-inf'), np.nan, np.inf, -np.inf]
    result = sanitize_for_json(dirty_data)
    assert all(item is None for item in result)

def test_sanitize_pandas_types():
    """Tests that pandas NaT and NA values are replaced with None."""
    pd_data = {
        "Missing_Date": pd.NaT, 
        "Missing_Value": pd.NA
    }
    result = sanitize_for_json(pd_data)
    
    assert result["Missing_Date"] is None
    assert result["Missing_Value"] is None

def test_sanitize_single_nat():
    """Tests that a standalone NaT value is replaced with None."""
    assert sanitize_for_json(pd.NaT) is None

def test_build_frontend_payload_info_exception():
    """Tests that the payload builder gracefully handles invalid info types."""
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
    payload = build_frontend_payload("AAPL", raw_ml_data, {}, nlp_data, "invalid_info_type")
    assert payload["Company_Name"] == "AAPL"
