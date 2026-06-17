import pytest
from pytest import approx
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from backend.models.forecast_model import (
    _engineer_price_features, 
    _train_price_classifier,
    _train_price_regressor,
    _forecast_price_long_term,
    _engineer_div_features,
    _train_div_classifier,
    _train_div_regressor,
    _forecast_div_long_term,
    _fetch_data,
    run_real_time_model,
    get_chart_data
)

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
    
    for i in range(100, 1200, 90):
        df.iloc[i, df.columns.get_loc("Dividends")] = 0.5 + (i/10000.0)
        
    return df


# price model tests
def test_engineer_price_features_creates_correct_columns(dummy_stock_data):
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    assert len(price_data) > 0
    assert "RSI_14" in predictors
    assert "Price_Target" in price_data.columns

def test_train_price_classifier(dummy_stock_data):
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    train = price_data.iloc[-500:-1]  
    test = price_data.iloc[-1:].copy()       
    direction, confidence = _train_price_classifier(train, test, predictors, "TEST")
    assert direction in [0, 1]
    assert 0.0 <= confidence <= 1.0

def test_train_price_regressor_up(dummy_stock_data):
    """Test regressor forcing an upward trajectory."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    train = price_data.iloc[-500:-1]  
    test = price_data.iloc[-1:].copy()
    today_close = float(test["Close"].values[0])
    
    forecast, fit_prices = _train_price_regressor(
        train, test, predictors, direction=1, today_close=today_close, 
        ticker="TEST", target_dates=price_data.index[-500:-1]
    )
    assert forecast >= round(today_close * 1.001, 2)

def test_train_price_regressor_down(dummy_stock_data):
    """Test regressor forcing a downward trajectory."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    train = price_data.iloc[-500:-1]  
    test = price_data.iloc[-1:].copy()
    today_close = float(test["Close"].values[0])
    
    forecast, fit_prices = _train_price_regressor(
        train, test, predictors, direction=0, today_close=today_close, 
        ticker="TEST", target_dates=price_data.index[-500:-1]
    )
    assert forecast <= round(today_close * 0.999, 2)

def test_forecast_price_long_term_full_data(dummy_stock_data):
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    dates, prices, upper, lower, ext = _forecast_price_long_term(
        price_data, price_data.iloc[-1:].copy(), predictors, 150.0, 152.0, 500, price_data.index[-1]
    )
    assert len(dates) == 252
    assert "1_Year" in ext

def test_forecast_price_long_term_short_data(dummy_stock_data):
    """Test long term forecast fallback when there are fewer than 100 days of data."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    short_price_data = price_data.iloc[-50:] 
    dates, prices, upper, lower, ext = _forecast_price_long_term(
        short_price_data, short_price_data.iloc[-1:].copy(), predictors, 150.0, 152.0, 50, short_price_data.index[-1]
    )
    assert ext["1_Year"]["Price"] == approx(150.0)


# dividend model tests
def test_engineer_div_features_success(dummy_stock_data):
    divs, div_predictors, next_date, today_div = _engineer_div_features(dummy_stock_data, dummy_stock_data.index[-1])
    assert divs is not None
    assert today_div > 0

def test_engineer_div_features_no_dividends(dummy_stock_data):
    """Test dividend extraction when a company pays no dividends."""
    dummy_stock_data["Dividends"] = 0.0
    divs, div_predictors, next_date, today_div = _engineer_div_features(dummy_stock_data, dummy_stock_data.index[-1])
    assert divs is None
    assert today_div == approx(0.0)

def test_engineer_div_features_zero_days():
    """Test edge case where dividend dates result in avg_days_between <= 0."""
    df = pd.DataFrame({"Dividends": [1.0]*15, "Close": [100.0]*15}, index=[pd.Timestamp("2020-01-01")]*15)
    divs, preds, n_date, t_div = _engineer_div_features(df, pd.Timestamp("2020-01-02"))
    assert n_date == pd.Timestamp("2020-01-01") + pd.Timedelta(days=90)

def test_train_div_classifier_same_class():
    """Test classifier fallback when the company's dividend never changes."""
    df = pd.DataFrame({"Div_Target": [1, 1, 1], "feat": [1, 2, 3]})
    pred, conf = _train_div_classifier(df, df, ["feat"])
    assert pred == 1
    assert conf == approx(0.5)

def test_train_div_classifier_value_error():
    """Test Isotonic Calibration ValueError fallback on small splits."""
    train = pd.DataFrame({"Div_Target": [1, 1, 1, 1, 0], "feat": [1, 2, 3, 4, 5]})
    test = pd.DataFrame({"feat": [3]})
    pred, conf = _train_div_classifier(train, test, ["feat"])
    assert pred in [0, 1]

def test_train_div_regressor_down():
    """Test dividend regressor forcing a downward trajectory."""
    train = pd.DataFrame({"Dividends": [1.0, 1.0, 1.0, 1.0], "Next_Dividend": [1.0, 1.1, 1.2, 0.9], "feat": [1, 2, 3, 4]})
    test = pd.DataFrame({"feat": [4]})
    forecast, fit_amounts = _train_div_regressor(train, test, ["feat"], div_pred=0, today_div=1.0)
    assert forecast <= 1.0 * 0.999

def test_forecast_div_long_term_short():
    """Test long term dividend projection fallback when data is sparse."""
    divs = pd.DataFrame({"Dividends": [1.0]*5, "feat": [1,2,3,4,5]})
    test = pd.DataFrame({"feat": [1]})
    d, a, u, l, ext = _forecast_div_long_term(divs, ["feat", "Dividends"], test, 1.0, 1.1, pd.Timestamp("2020-01-01"), 90, 4)
    assert ext["4_Payouts"]["Amount"] == approx(1.0)

# data retrieval & chart data tests
@patch('backend.models.forecast_model.yf.Ticker')
def test_get_chart_data_success(mock_ticker):
    """Test successful data retrieval for chart rendering."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.date_range("2023-01-01", periods=2))
    mock_instance.dividends = pd.Series([1.0], index=pd.date_range("2023-01-01", periods=1))
    mock_ticker.return_value = mock_instance
    
    res = get_chart_data("AAPL", predicted_price=105.0)
    assert len(res["dates"]) == 3 
    assert res["prices"][-1] == approx(105.0)
    assert len(res["dividend_dates"]) == 1

@patch('backend.models.forecast_model.yf.Ticker')
def test_get_chart_data_empty(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_instance.dividends = pd.Series(dtype=float)
    mock_ticker.return_value = mock_instance
    res = get_chart_data("FAKE")
    assert res["dates"] == []

@patch('backend.models.forecast_model.yf.Ticker')
def test_fetch_data_empty(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance
    assert _fetch_data("EMPTY") is None

@patch('backend.models.forecast_model.yf.Ticker')
def test_fetch_data_insufficient(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame({"Close": [1]*100}, index=pd.date_range("2020-01-01", periods=100))
    mock_ticker.return_value = mock_instance
    assert _fetch_data("SHORT") is None

@patch('backend.models.forecast_model.yf.Ticker')
def test_fetch_data_success(mock_ticker):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame({"Close": [1]*1300}, index=pd.date_range("2010-01-01", periods=1300))
    mock_ticker.return_value = mock_instance
    assert len(_fetch_data("GOOD")) == 1300

# real-time model integration tests
@patch('backend.models.forecast_model._fetch_data')
def test_run_real_time_model_integration(mock_fetch_data, dummy_stock_data):
    """End-to-End Test: Simulates a perfect pipeline run."""
    mock_fetch_data.return_value = dummy_stock_data
    result_df = run_real_time_model("TEST", price_window=300, div_window=8)
    assert result_df is not None
    assert "Forecasted_Close" in result_df.columns
    assert result_df["Forecasted_Dividend"].iloc[0] != "N/A"

@patch('backend.models.forecast_model._fetch_data')
def test_run_real_time_model_no_data(mock_fetch_data):
    """End-to-End Test: Simulates a pipeline run with bad data."""
    mock_fetch_data.return_value = None
    assert run_real_time_model("TEST") is None

@patch('backend.models.forecast_model.yf.Ticker')
@patch('backend.models.forecast_model._fetch_data')
def test_run_real_time_model_no_divs_and_info_err(mock_fetch, mock_ticker, dummy_stock_data):
    """End-to-End Test: Simulates a non-dividend stock and a crashed Yahoo Finance Info API."""
    dummy_stock_data["Dividends"] = 0.0
    mock_fetch.return_value = dummy_stock_data
    
    mock_instance = MagicMock()
    type(mock_instance).info = PropertyMock(side_effect=Exception("API Error"))
    mock_ticker.return_value = mock_instance
    
    res = run_real_time_model("TEST", price_window=300)
    assert res["Forecasted_Dividend"].iloc[0] == "N/A"
    assert res["Company_Name"].iloc[0] == "TEST"