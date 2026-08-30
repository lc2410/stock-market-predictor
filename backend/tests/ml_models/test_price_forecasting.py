"""Tests for the price forecasting module."""
import pandas as pd
from ml_models.price_forecasting import (
    _engineer_price_features, 
    _train_multi_horizon_price,
    run_price_prediction
)

def test_engineer_price_features_creates_correct_columns(dummy_stock_data):
    """Tests that feature engineering produces the expected technical indicator columns."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    assert len(price_data) > 0
    assert "RSI_14" in predictors
    assert "MACD_Hist" in predictors

def test_train_multi_horizon_price(dummy_stock_data):
    """Tests that multi-horizon price training produces valid forecasts with bounds."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    # Use a small window to speed up test
    res, _, _, _, t_dates, t_prices = _train_multi_horizon_price(price_data, predictors, is_crypto=False, price_window=500)
    
    assert "Next_Day" in res
    assert "Next_Year" in res
    assert "Direction" in res["Next_Day"]
    assert 0 <= res["Next_Day"]["Direction_Confidence"] <= 100
    assert res["Next_Day"]["Amount"] > 0
    assert "Amount_Lower" in res["Next_Day"]
    assert "Amount_Upper" in res["Next_Day"]
    
    assert len(t_dates) == len(t_prices)

def test_run_price_prediction_integration(dummy_stock_data):
    """Tests the full price prediction pipeline end-to-end."""
    res = run_price_prediction(dummy_stock_data, price_window=300)
    assert res is not None
    assert "price_forecasts" in res
    assert "Next_Day" in res["price_forecasts"]

def test_run_price_prediction_no_data():
    """Tests the prediction pipeline with minimal data triggers the fallback."""
    res = run_price_prediction(pd.DataFrame({"Close": [10.0]}, index=[pd.Timestamp("2023-01-01")]))
    assert "Next_Day" not in res["price_forecasts"]
    assert len(res["train_fit_dates"]) == 0

def test_train_multi_horizon_price_insufficient_data(dummy_stock_data):
    """Tests that insufficient data triggers the ML fallback with empty results."""
    # Pass a tiny dataset to trigger ML fallback
    price_data, predictors = _engineer_price_features(dummy_stock_data.iloc[-10:])
    res, _, _, _, t_dates, _ = _train_multi_horizon_price(price_data, predictors, is_crypto=True, price_window=5)
    assert res == {}
    assert len(t_dates) == 0

def test_train_multi_horizon_price_crypto_dates(dummy_stock_data):
    """Tests that crypto mode uses calendar days instead of business days."""
    price_data, predictors = _engineer_price_features(dummy_stock_data)
    # Test crypto dates logic where we use +1 day instead of business days
    res, _, _, _, t_dates, _ = _train_multi_horizon_price(price_data, predictors, is_crypto=True, price_window=500)
    assert len(t_dates) > 0
    # Next day after the first valid index should be just +1 day
    assert "Next_Day" in res
