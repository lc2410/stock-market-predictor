"""Service utilities for UI display, screener dashboards, and prediction orchestration (data fetching and charting)."""

import pandas as pd
import yfinance as yf
import numpy as np
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

logging.getLogger('yfinance').setLevel(logging.ERROR)

# Screener Utility Functions
def calculate_change_pct(current_price, prev_price):
    """Calculates the percentage change between two prices."""
    if prev_price <= 0:
        return 0.0
    return ((current_price - prev_price) / prev_price) * 100

def calculate_volatility(high, low):
    """Calculates intraday volatility as a percentage of the low price."""
    if low <= 0:
        return 0.0
    return ((high - low) / low) * 100

def calculate_52_week_high_low(data, col="Close"):
    """Calculates the 52-week (252 trading days) high and low."""
    high = data[col].rolling(252, min_periods=1).max().iloc[-1]
    low = data[col].rolling(252, min_periods=1).min().iloc[-1]
    return high, low

def calculate_52_week_return(data, col="Close"):
    """Calculates the 52-week price return."""
    if len(data) < 252:
        if len(data) < 2:
            return None
        return data[col].pct_change(len(data) - 1).iloc[-1]
    return data[col].pct_change(252).iloc[-1]

def calculate_average_volume(data, window=30, col="Volume"):
    """Calculates the rolling average volume."""
    if len(data) >= window:
        return float(data[col].iloc[-window:].mean())
    return float(data[col].mean())

def calculate_ttm_dividend_yield(data, current_price):
    """Calculates Trailing 12-Month Dividend Yield."""
    if "Dividends" not in data.columns or current_price <= 0:
        return 0.0
    
    if isinstance(data.index, pd.DatetimeIndex):
        one_year_ago = data.index.max() - pd.Timedelta(days=365)
        recent_dividends = data["Dividends"][data.index >= one_year_ago]
        dividend_payout = recent_dividends.sum()
        
        if dividend_payout > 0:
            return float(dividend_payout / current_price)
    return 0.0

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index using SMA smoothing."""
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)


# Prediction Service Data & Chart Functions
def get_chart_data(price_data, div_data=None, is_crypto=False, show_all_prices=False, show_all_divs=False):
    """Retrieves the recent historical price and dividend data needed to draw the frontend charts."""
    if price_data is None or price_data.empty:
        return {"dates": [], "prices": [], "dividend_dates": [], "dividend_amounts": []}
        
    if show_all_prices:
        hist = price_data
    else:
        # Slice to past 1 year of trading data for the chart UI
        days_in_year = 365 if is_crypto else 252
        hist = price_data.iloc[-days_in_year:]
    
    dates = hist.index.strftime('%Y-%m-%d').tolist()
    prices = [round(float(p), 2) for p in hist['Close'].tolist()]

    div_source = div_data if div_data is not None else price_data

    # Extract historical dividends
    if 'Dividends' in div_source.columns:
        dividends = div_source[div_source['Dividends'] > 0]['Dividends']
        if not dividends.empty:
            if not show_all_divs:
                # Slice to past 5 dividend payouts for the chart UI
                dividends = dividends.iloc[-5:]
            dividend_dates = dividends.index.strftime('%Y-%m-%d').tolist()
            dividend_amounts = [round(float(d), 2) for d in dividends.tolist()]
        else:
            dividend_dates = []
            dividend_amounts = []
    else:
        dividend_dates = []
        dividend_amounts = []

    return {
        "dates": dates,
        "prices": prices,
        "dividend_dates": dividend_dates,
        "dividend_amounts": dividend_amounts
    }

def _process_fetched_data(data):
    if data.empty:
        return None, None
        
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep='last')]
    data = data.dropna(subset=['Close'])
    
    dividends = data[data["Dividends"] > 0]
    return data, dividends

def fetch_data(ticker, target_window, is_crypto=False):
    """
    Fetches historical stock data from Yahoo Finance.
    Adaptively fetches data incrementally (5 to 30 years) to balance API latency and data completeness.
    """
    stock_ticker = yf.Ticker(ticker)
    
    buffer_days = 365 if is_crypto else 252
    min_required_days = target_window + buffer_days
    
    years_to_fetch = 5
    data = None
    dividends = None
    
    while years_to_fetch <= 30:
        end_date = datetime.now() + timedelta(days=1)
        days_to_fetch = (years_to_fetch * 365) + (years_to_fetch // 4) + 1 # Add leap days
        start_date = end_date - timedelta(days=days_to_fetch)
        raw_data = stock_ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        data, dividends = _process_fetched_data(raw_data)
        if data is None:
            return None, None
        
        has_enough_price = len(data) >= min_required_days
        
        has_enough_divs = True
        expected_days = years_to_fetch * (365 if is_crypto else 252) * 0.90
        # Check if we need more history to capture the 25 payout minimum
        if 0 < len(dividends) < 25 and len(data) >= expected_days:
            has_enough_divs = False
        
        if (has_enough_price and has_enough_divs) or len(data) < expected_days:
            break
            
        years_to_fetch += 5

    if data is None or len(data) < 2:
        return None, None
        
    # Isolate recent data for the price model to minimize computation
    price_data_slice = data.iloc[-min_required_days:].copy() if len(data) >= min_required_days else data.copy()
        
    # Strip non-price metrics from the price dataset
    price_data_slice = price_data_slice.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
    
    # Isolate enough data to capture 25 payouts plus a 1-year trailing price buffer
    if len(dividends) > 25:
        earliest_div_date = dividends.index[-25]
        cutoff_date = earliest_div_date - pd.Timedelta(days=365)
        div_data_slice = data.loc[cutoff_date:].copy()
    else:
        div_data_slice = data.copy()
        
    return price_data_slice, div_data_slice

def generate_future_chart_data(horizon_anchors, anchors_lower, anchors_upper, anchor_date, is_crypto, is_div=False, avg_days_between=90):
    """
    Interpolates linearly between forecasted horizon anchor points (e.g., 1 day, 1 week, 1 month, 1 year) 
    to generate continuous line data for rendering charts on the frontend.
    """
    if len(horizon_anchors) <= 1:
        return [], [], [], []
        
    if is_div:
        all_future_dates = [anchor_date + pd.Timedelta(days=avg_days_between * i) for i in range(1, 6)]
        keys = [1, 2, 3, 4, 5]
    else:
        if is_crypto:
            all_future_dates = pd.date_range(start=anchor_date + pd.Timedelta(days=1), periods=365, freq='D')
        else:
            us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            all_future_dates = pd.date_range(start=anchor_date + us_bday, periods=252, freq=us_bday)
        keys = list(range(1, len(all_future_dates) + 1))

    pts_median = sorted(horizon_anchors.items())
    pts_lower = sorted(anchors_lower.items())
    pts_upper = sorted(anchors_upper.items())

    def interp_amount(step, anchors):
        for i in range(len(anchors) - 1):
            step_start, price_start = anchors[i]
            step_end, price_end = anchors[i + 1]
            if step_start <= step <= step_end:
                frac = (step - step_start) / (step_end - step_start)
                return float(np.exp(np.log(price_start) + frac * (np.log(price_end) - np.log(price_start))))
        return anchors[-1][1]

    dates, prices, upper, lower = [], [], [], []
    for i, step in enumerate(keys):
        amount_t = round(interp_amount(step, pts_median), 2)
        lower_bound = round(interp_amount(step, pts_lower), 2)
        upper_bound = round(interp_amount(step, pts_upper), 2)
        
        dates.append(all_future_dates[i].strftime('%Y-%m-%d'))
        prices.append(amount_t)
        upper.append(upper_bound)
        lower.append(lower_bound)

    return dates, prices, upper, lower
