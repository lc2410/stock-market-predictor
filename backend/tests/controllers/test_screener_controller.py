"""Tests for the screener controller endpoint."""
from unittest.mock import patch

import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('controllers.screener_controller.get_screener_dashboard_data')
def test_screener_endpoint_success(mock_screener, client):
    """Tests that the screener returns benchmark and movers data."""
    mock_screener.return_value = {
        "benchmarks": [{"name": "DOW", "price": 40000}],
        "market_movers": {},
        "custom_scans": {},
        "headlines": []
    }
    response = client.get('/api/screener')
    assert response.status_code == 200
    data = response.get_json()
    assert "benchmarks" in data
    assert data["benchmarks"][0]["name"] == "DOW"

@patch('controllers.screener_controller.get_screener_dashboard_data')
def test_screener_endpoint_error(mock_screener, client):
    """Tests that a service error returns a 500 response."""
    mock_screener.side_effect = Exception("DB Error")
    response = client.get('/api/screener')
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
