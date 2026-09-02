from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.service_utils import (
    calculate_52_week_high_low,
    calculate_52_week_return,
    calculate_average_volume,
    calculate_change_pct,
    calculate_rsi,
    calculate_ttm_dividend_yield,
    calculate_volatility,
    fetch_data,
    generate_future_chart_data,
    get_chart_data,
)


def test_calculate_change_pct():
    assert calculate_change_pct(110, 100) == 10.0
    assert calculate_change_pct(90, 100) == -10.0
    assert calculate_change_pct(100, 0) == 0.0

def test_calculate_volatility():
    assert calculate_volatility(110, 100) == 10.0
    assert calculate_volatility(100, 0) == 0.0

def test_calculate_52_week_high_low():
    df = pd.DataFrame({'Close': [100 + i for i in range(300)]})
    high, low = calculate_52_week_high_low(df)
    assert high == 399
    assert low == 148

def test_calculate_52_week_return():
    # Less than 2 days
    df_short = pd.DataFrame({'Close': [100]})
    assert calculate_52_week_return(df_short) is None

    # Less than 252 days
    df_medium = pd.DataFrame({'Close': [100, 110, 120]})
    assert calculate_52_week_return(df_medium) == pytest.approx(0.2)

    # More than 252 days
    df_long = pd.DataFrame({'Close': [100] * 252 + [110]})
    # pct_change(252).iloc[-1] => (110 - 100) / 100 = 0.1
    assert abs(calculate_52_week_return(df_long) - 0.1) < 1e-6

def test_calculate_average_volume():
    # Window 30
    df_long = pd.DataFrame({'Volume': [10] * 50})
    assert calculate_average_volume(df_long, window=30) == 10.0

    # Short
    df_short = pd.DataFrame({'Volume': [10, 20]})
    assert calculate_average_volume(df_short, window=30) == 15.0

def test_calculate_ttm_dividend_yield():
    dates = pd.date_range(end=datetime.now(), periods=400, freq='D')
    df = pd.DataFrame(index=dates, data={'Dividends': [0.1] * 400})
    
    # 365 days of dividends
    yield_val = calculate_ttm_dividend_yield(df, 100)
    # Sum of 366 rows of 0.1 because >= one_year_ago
    # approx 36.6 / 100 = 0.366
    assert yield_val > 0.35

    # No dividend col
    df_nodiv = pd.DataFrame(index=dates, data={'Close': [10] * 400})
    assert calculate_ttm_dividend_yield(df_nodiv, 100) == 0.0

    # Price zero
    assert calculate_ttm_dividend_yield(df, 0) == 0.0

def test_calculate_rsi():
    df = pd.DataFrame({'Close': [10, 12, 15, 14, 16, 18, 20, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30]})
    rsi = calculate_rsi(df, window=14)
    assert not rsi.empty
    assert rsi.iloc[-1] > 0

def test_get_chart_data():
    dates = pd.date_range(end='2023-01-01', periods=10, freq='D')
    price_df = pd.DataFrame(index=dates, data={'Close': [10]*10})
    div_df = pd.DataFrame(index=dates, data={'Dividends': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]})
    
    res = get_chart_data(price_df, div_df, is_crypto=False, show_all_prices=True, show_all_divs=True)
    assert len(res["dates"]) == 10
    assert len(res["prices"]) == 10
    assert len(res["dividend_dates"]) == 1
    assert len(res["dividend_amounts"]) == 1

    # Empty
    res_empty = get_chart_data(pd.DataFrame())
    assert res_empty["dates"] == []

@patch('utils.service_utils.yf.Ticker')
def test_fetch_data(mock_ticker):
    mock_instance = MagicMock()
    
    dates = pd.date_range(end='2023-01-01', periods=2000, freq='D')
    hist_df = pd.DataFrame(index=dates, data={'Close': [10]*2000, 'Dividends': [0]*2000})
    hist_df.loc[dates[-100], 'Dividends'] = 1 # One dividend
    
    mock_instance.history.return_value = hist_df
    mock_ticker.return_value = mock_instance
    
    p_df, d_df = fetch_data("AAPL", 252, is_crypto=False)
    assert p_df is not None
    assert d_df is not None

def test_generate_future_chart_data():
    anchor_date = datetime(2023, 1, 1)
    
    anchors = {1: 100, 5: 110}
    lower = {1: 90, 5: 100}
    upper = {1: 110, 5: 120}
    
    # Not enough anchors
    res = generate_future_chart_data({1: 100}, lower, upper, anchor_date, False, False)
    assert res == ([], [], [], [])
    
    res = generate_future_chart_data(anchors, lower, upper, anchor_date, True, False)
    dates, prices = res[:2]
    assert len(dates) == 365
    assert len(prices) == 365
    assert prices[0] == 100.0
    assert prices[4] == 110.0
