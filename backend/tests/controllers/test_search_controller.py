"""Tests for the search controller endpoint."""
import pytest
import requests
from unittest.mock import patch, MagicMock
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('controllers.search_controller.requests.get')
def test_search_endpoint_success(mock_get, client):
    """Tests that a valid query returns matching stock symbols."""
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

@patch('controllers.search_controller.requests.get')
def test_search_endpoint_empty_results(mock_get, client):
    """Tests that a query with no matches returns an empty list."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"other_key": "value"}
    mock_get.return_value = mock_response

    response = client.get('/search/BLAH')
    assert response.status_code == 200
    assert response.get_json() == []

@patch('controllers.search_controller.requests.get')
def test_search_endpoint_exception(mock_get, client):
    """Tests that a network error returns an empty list gracefully."""
    mock_get.side_effect = requests.exceptions.RequestException("Network timeout")
    response = client.get('/search/ERROR')
    assert response.status_code == 200
    assert response.get_json() == []
