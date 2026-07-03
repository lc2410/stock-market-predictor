import pandas as pd
from backend.models.dividend_forecasting import (
    _engineer_div_features,
    _train_multi_horizon_div,
    run_dividend_prediction
)

def test_engineer_div_features_success(dummy_stock_data):
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, _ = _engineer_div_features(dummy_stock_data, anchor_date)
    assert divs is not None
    assert "Price_Return_252" in div_predictors

def test_engineer_div_features_no_dividends(dummy_stock_data):
    df = dummy_stock_data.copy()
    df["Dividends"] = 0.0
    anchor_date = df.index[-1]
    divs, _, next_div_date = _engineer_div_features(df, anchor_date)
    assert divs is None
    assert pd.isna(next_div_date)

def test_train_multi_horizon_div(dummy_stock_data):
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, next_div_date = _engineer_div_features(dummy_stock_data, anchor_date)
    
    res, _, _, _, _, _ = _train_multi_horizon_div(divs, div_predictors, div_window=25)
    
    assert "Next_Payout" in res
    assert "Payout_4" in res
    assert res["Next_Payout"]["Amount"] > 0
    assert "Amount_Lower" in res["Next_Payout"]
    assert "Amount_Upper" in res["Next_Payout"]

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
    
    res, _, _, _, t_dates, _ = _train_multi_horizon_div(tiny_divs, ["Price_Return_252"], div_window=2)
    assert res == {}
    assert len(t_dates) == 0

def test_run_dividend_prediction_integration(dummy_stock_data):
    anchor_date = dummy_stock_data.index[-1]
    res = run_dividend_prediction(dummy_stock_data, anchor_date, div_window=25)
    assert res["has_enough_div_data"] == True
    assert "Next_Payout" in res["div_forecasts"]
    assert res["avg_days_between"] > 0

def test_run_dividend_prediction_no_data():
    empty_df = pd.DataFrame(columns=["Close", "Dividends"])
    res = run_dividend_prediction(empty_df, pd.Timestamp("2023-01-01"))
    assert res["has_enough_div_data"] == False
    assert res["avg_days_between"] == 90
    assert len(res["train_fit_div_dates"]) == 0

def test_engineer_div_features_same_day_dividends():
    # Force average days to 0 to test fallback to 90
    df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0],
        "Dividends": [0.0, 1.0, 1.0] # 2 dividends on same day? No, index is dates.
    }, index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")])
    divs, _, next_div_date = _engineer_div_features(df, pd.Timestamp("2020-01-02"), div_window=2)
    assert divs is not None
    # since dates are identical, diff is 0 days, avg_days_between falls back to 90
    assert next_div_date == pd.Timestamp("2020-01-02") + pd.Timedelta(days=90)

def test_train_multi_horizon_div_future_dates(dummy_stock_data):
    # Pass a tiny div_window and mock the fit loop so we index out of bounds on the historical data
    # This triggers the `else` condition on line 84: test_fit_dates.append(...) using avg_days
    anchor_date = dummy_stock_data.index[-1]
    divs, div_predictors, _ = _engineer_div_features(dummy_stock_data, anchor_date)
    # Give it exactly 10 rows so it passes the ML length check but runs out of future dates quickly
    res, _, _, _, t_dates, _ = _train_multi_horizon_div(divs.iloc[-15:], div_predictors, div_window=15)
    assert len(t_dates) > 0

