import pytest
from pytest import approx
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from backend.models.forecast_model import (
    _engineer_price_features, 
    _train_multi_horizon_price,
    _engineer_div_features,
    _train_multi_horizon_div,
    _fetch_data,
    run_real_time_model,
    get_chart_data,
    extract_quantiles_metrics
)
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

@pytest.fixture
def dummy_stock_data():
    """Generates 1,300 days of fake stock data for model testing."""
    dates = pd.date_range(start="2018-01-01", periods=1300, freq="B")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "Close": np.linspace(100, 200, 1300) + np.random.normal(0, 2, 1300),
        "Volume": np.random.randint(100000, 500000, 1300),
        "High": np.linspace(105, 205, 1300),
        "Low": np.linspace(95, 195, 1300),
        "Dividends": 0.0
    }, index=dates)
    
    div_indices = np.random.choice(1300, 30, replace=False)
    df.loc[df.index[div_indices], "Dividends"] = np.random.uniform(0.1, 0.5, 30)
        
    return df

def test_engineer_price_features_creates_correct_columns(dummy_stock_data):
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    assert len(price_data) > 0
    assert "RSI_14" in predictors
    assert "MACD_Hist" in predictors

def test_train_multi_horizon_price(dummy_stock_data):
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    # Use a small window to speed up test
    res, anchors, a_lower, a_upper, t_dates, t_prices = _train_multi_horizon_price(price_data, predictors, is_crypto=False, price_window=500)
    
    assert "Next_Day" in res
    assert "Next_Year" in res
    assert "Direction" in res["Next_Day"]
    assert 0 <= res["Next_Day"]["Direction_Confidence"] <= 100
    assert res["Next_Day"]["Amount"] > 0
    assert "Amount_Lower" in res["Next_Day"]
    assert "Amount_Upper" in res["Next_Day"]
    
    assert len(t_dates) == len(t_prices)

def test_engineer_div_features_success(dummy_stock_data):
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, next_div_date = _engineer_div_features(dummy_stock_data, anchor_date)
    assert divs is not None
    assert "Price_Return_252" in div_predictors

def test_engineer_div_features_no_dividends(dummy_stock_data):
    df = dummy_stock_data.copy()
    df["Dividends"] = 0.0
    anchor_date = df.index[-1]
    divs, div_predictors, next_div_date = _engineer_div_features(df, anchor_date)
    assert divs is None
    assert pd.isna(next_div_date)

def test_train_multi_horizon_div(dummy_stock_data):
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, next_div_date = _engineer_div_features(dummy_stock_data, anchor_date)
    
    res, anchors, a_lower, a_upper, t_dates, t_amounts = _train_multi_horizon_div(divs, div_predictors, div_window=25)
    
    assert "Next_Payout" in res
    assert "Payout_4" in res
    assert res["Next_Payout"]["Amount"] > 0
    assert "Amount_Lower" in res["Next_Payout"]
    assert "Amount_Upper" in res["Next_Payout"]

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

def test_run_real_time_model_integration(dummy_stock_data):
    with patch("backend.models.forecast_model._fetch_data", return_value=(dummy_stock_data, dummy_stock_data, 300)):
        res = run_real_time_model("TEST", price_window=300, div_window=25)
        assert res is not None
        assert "price_forecasts" in res
        assert "div_forecasts" in res
        assert "Next_Day" in res["price_forecasts"]
        assert len(res["chart_future_dates"]) == 252 # US business days

def test_run_real_time_model_no_data():
    with patch("backend.models.forecast_model._fetch_data", return_value=(None, None, None)):
        res = run_real_time_model("TEST")
        assert res is None

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

@patch("backend.models.forecast_model.yf.Ticker")
def test_fetch_data_empty(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance
    
    price, div, window = _fetch_data("EMPTY")
    assert price is None
    assert div is None

@patch("backend.models.forecast_model.yf.Ticker")
def test_fetch_data_insufficient(mock_ticker):
    mock_instance = MagicMock()
    # Return a tiny dataset that triggers the "break early" logic
    mock_instance.history.return_value = pd.DataFrame({"Close": [100.0, 101.0], "Dividends": [0.0, 0.0]}, index=pd.date_range("2020-01-01", periods=2))
    mock_ticker.return_value = mock_instance
    
    price, div, window = _fetch_data("TINY")
    # Will hit the len(data) < expected_days break, but len(data) == 2 so it returns the data for downstream handling
    assert price is not None
    assert len(price) == 2

@patch("backend.models.forecast_model.yf.Ticker")
def test_fetch_data_one_row(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame({"Close": [100.0], "Dividends": [0.0]}, index=pd.date_range("2020-01-01", periods=1))
    mock_ticker.return_value = mock_instance
    
    price, div, window = _fetch_data("ONE")
    # Will hit len(data) < 2
    assert price is None

@patch("backend.models.forecast_model.yf.Ticker")
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
    
    price, div, window = _fetch_data("LOOP")
    assert price is not None
    assert div is not None
    # Verifies price was sliced to min_required_days
    assert len(price) == 1260 + 252
    # Verifies dividends were sliced down from 2500
    assert len(div) < 2500

def test_train_multi_horizon_price_insufficient_data(dummy_stock_data):
    # Pass a tiny dataset to trigger ML fallback
    price_data, predictors = _engineer_price_features(dummy_stock_data.iloc[-10:])
    res, anchors, a_lower, a_upper, t_dates, t_prices = _train_multi_horizon_price(price_data, predictors, is_crypto=True, price_window=5)
    assert res == {}
    assert len(t_dates) == 0

def test_train_multi_horizon_div_insufficient_data(dummy_stock_data):
    # Pass a tiny dataset to trigger ML fallback
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, _ = _engineer_div_features(dummy_stock_data.iloc[-10:], anchor_date)
    # divs could be None if no dividends, so manually create one
    tiny_divs = pd.DataFrame({"Dividends": [1.0, 1.1]}, index=pd.date_range("2020-01-01", periods=2))
    tiny_divs["Price_Return_252"] = 0.05
    tiny_divs["Price_Volatility_252"] = 0.1
    tiny_divs["Div_Growth_1"] = 0.1
    tiny_divs["Yield_On_Cost"] = 0.05
    
    res, anchors, a_lower, a_upper, t_dates, t_amounts = _train_multi_horizon_div(tiny_divs, ["Price_Return_252"], div_window=2)
    assert res == {}
    assert len(t_dates) == 0

def test_run_real_time_model_crypto_path(dummy_stock_data):
    with patch("backend.models.forecast_model._fetch_data", return_value=(dummy_stock_data, dummy_stock_data, 500)):
        res = run_real_time_model("BTC-USD", price_window=500, div_window=25, is_crypto=True)
        assert res is not None
        # Crypto model returns 365 days of future dates instead of 252 business days
        assert len(res["chart_future_dates"]) == 365
