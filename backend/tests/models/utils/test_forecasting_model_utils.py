import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from models.utils.forecasting_model_utils import (
    fetch_data,
    get_chart_data,
    extract_quantiles_metrics
)
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def test_extract_quantiles_metrics():
    X = pd.DataFrame(np.random.rand(100, 2), columns=["A", "B"])
    y_reg = np.linspace(-0.1, 0.1, 100)
    y_class = (y_reg > 0).astype(int)
    
    clf = HistGradientBoostingClassifier(random_state=1)
    clf.fit(X, y_class)
    
    reg_median = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, random_state=1)
    reg_median.fit(X, y_reg)
    
    reg_lower = HistGradientBoostingRegressor(loss='quantile', quantile=0.1, random_state=1)
    reg_lower.fit(X, y_reg)
    
    reg_upper = HistGradientBoostingRegressor(loss='quantile', quantile=0.9, random_state=1)
    reg_upper.fit(X, y_reg)
    
    test_row = X.iloc[[0]].copy()
    metrics = extract_quantiles_metrics(clf, reg_median, reg_lower, reg_upper, test_row, ["A", "B"], 100.0)
    
    assert "Direction" in metrics
    assert "Amount" in metrics
    assert 0 <= metrics["Direction_Confidence"] <= 100
    assert "Amount_Lower" in metrics
    assert "Amount_Upper" in metrics

def test_get_chart_data_success():
    mock_hist = pd.DataFrame({
        "Close": [150.0, 155.0],
        "Dividends": [0.0, 0.5]
    }, index=pd.date_range("2023-01-01", periods=2))
    
    data = get_chart_data(mock_hist)
    assert len(data["dates"]) == 2
    assert len(data["dividend_dates"]) == 1

def test_get_chart_data_edge_cases():
    # 1. Empty data
    assert get_chart_data(None) == {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}
    assert get_chart_data(pd.DataFrame()) == {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}
    
    # 2. show_all_prices = True
    df = pd.DataFrame({"Close": [10.0]}, index=pd.date_range("2020-01-01", periods=1))
    res = get_chart_data(df, show_all_prices=True)
    assert len(res["prices"]) == 1
    
    # 3. No dividends column
    assert get_chart_data(df)["dividend_dates"] == []
    
    # 4. Dividends column but all zero
    df["Dividends"] = 0.0
    assert get_chart_data(df)["dividend_dates"] == []

@patch("models.utils.forecasting_model_utils.yf.Ticker")
def test_fetch_data_empty(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance
    
    price, div = fetch_data("EMPTY", 1260)
    assert price is None
    assert div is None

@patch("models.utils.forecasting_model_utils.yf.Ticker")
def test_fetch_data_insufficient(mock_ticker):
    mock_instance = MagicMock()
    # Return a tiny dataset that triggers the "break early" logic
    mock_instance.history.return_value = pd.DataFrame({"Close": [100.0, 101.0], "Dividends": [0.0, 0.0]}, index=pd.date_range("2020-01-01", periods=2))
    mock_ticker.return_value = mock_instance
    
    price, div = fetch_data("TINY", 1260)
    # Will hit the len(data) < expected_days break, but len(data) == 2 so it returns the data for downstream handling
    assert price is not None
    assert len(price) == 2

@patch("models.utils.forecasting_model_utils.yf.Ticker")
def test_fetch_data_one_row(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame({"Close": [100.0], "Dividends": [0.0]}, index=pd.date_range("2020-01-01", periods=1))
    mock_ticker.return_value = mock_instance
    
    price, div = fetch_data("ONE", 1260)
    # Will hit len(data) < 2
    assert price is None

@patch("models.utils.forecasting_model_utils.yf.Ticker")
def test_fetch_data_full_loop_and_dividends_slice(mock_ticker):
    mock_instance = MagicMock()
    
    # 1st call (5y): has enough days (1200), but only 5 dividends
    df_5y = pd.DataFrame({"Close": np.ones(1200), "Dividends": 0.0}, index=pd.date_range("2018-01-01", periods=1200))
    df_5y.loc[df_5y.index[0:5], "Dividends"] = 1.0
    
    # 2nd call (10y): has 2500 days, and 30 dividends
    df_10y = pd.DataFrame({"Close": np.ones(2500), "Dividends": 0.0}, index=pd.date_range("2013-01-01", periods=2500))
    df_10y.loc[df_10y.index[-30:], "Dividends"] = 1.0
    
    mock_instance.history.side_effect = [df_5y, df_10y]
    mock_ticker.return_value = mock_instance
    
    price, div = fetch_data("LOOP", 1260)
    assert price is not None
    assert div is not None
    # Verifies price was sliced to min_required_days
    assert len(price) == 1260 + 252
    # Verifies dividends were sliced down from 2500
    assert len(div) < 2500

from models.utils.forecasting_model_utils import generate_future_chart_data

def test_generate_future_chart_data_empty():
    d, p, u, l = generate_future_chart_data({1: 100.0}, {1: 90.0}, {1: 110.0}, pd.Timestamp("2023-01-01"), is_crypto=False)
    assert d == []
    assert p == []

def test_generate_future_chart_data_equity():
    anchors = {1: 100.0, 5: 110.0, 21: 120.0}
    anchors_lower = {1: 90.0, 5: 100.0, 21: 110.0}
    anchors_upper = {1: 110.0, 5: 120.0, 21: 130.0}
    d, p, u, l = generate_future_chart_data(anchors, anchors_lower, anchors_upper, pd.Timestamp("2023-01-01"), is_crypto=False)
    assert len(d) == 252
    assert p[0] == 100.0
    # Interpolation test
    assert p[2] > 100.0 and p[2] < 110.0
    assert p[-1] == 120.0 # flat line after last anchor

def test_generate_future_chart_data_crypto():
    anchors = {1: 100.0, 7: 110.0}
    d, p, u, l = generate_future_chart_data(anchors, anchors, anchors, pd.Timestamp("2023-01-01"), is_crypto=True)
    assert len(d) == 365
    assert p[0] == 100.0

def test_generate_future_chart_data_dividend():
    anchors = {1: 1.0, 2: 1.1}
    d, p, u, l = generate_future_chart_data(anchors, anchors, anchors, pd.Timestamp("2023-01-01"), is_crypto=False, is_div=True)
    assert len(d) == 5
    assert p[0] == 1.0

