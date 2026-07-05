"""
forecasting_model_utils.py
--------------------------
Shared mathematical utilities and helper functions for the machine learning pipelines.
This module handles dynamic data fetching from Yahoo Finance, date math for market closures
(weekends/holidays), and structing the historical/projected datasets for Chart.js.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging

logging.getLogger('yfinance').setLevel(logging.ERROR)

def get_us_bday():
    """Returns a calendar object used to skip weekends and US market holidays."""
    return CustomBusinessDay(calendar=USFederalHolidayCalendar())

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
        data = stock_ticker.history(period=f"{years_to_fetch}y")
        
        if data.empty:
            return None, None
            
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        data = data[~data.index.duplicated(keep='last')]
        data = data.dropna(subset=['Close'])
        
        dividends = data[data["Dividends"] > 0]
        
        has_enough_price = len(data) >= min_required_days
        
        has_enough_divs = True
        # Check if we need more history to capture the 25 payout minimum
        if len(dividends) > 0 and len(dividends) < 25:
            expected_days = years_to_fetch * (365 if is_crypto else 252) * 0.90
            if len(data) >= expected_days:
                has_enough_divs = False
        
        if has_enough_price and has_enough_divs:
            break
            
        # Break early if we've hit the asset's IPO date
        expected_days = years_to_fetch * (365 if is_crypto else 252) * 0.90
        if len(data) < expected_days:
            break
            
        years_to_fetch += 5

    if data is None or len(data) < 2:
        return None, None
        
    # Isolate recent data for the price model to minimize computation
    if len(data) >= min_required_days:
        price_data_slice = data.iloc[-min_required_days:].copy()
    else:
        price_data_slice = data.copy()
        
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

def extract_quantiles_metrics(clf, reg_median, reg_lower, reg_upper, test_row, predictors, today_val):
    """
    Extracts direction, confidence, and bounds from the models.
    Regressor defines the amount and direction; classifier defines direction confidence.
    """
    mean_log_return = float(reg_median.predict(test_row[predictors])[0])
    lower_log_return = float(reg_lower.predict(test_row[predictors])[0])
    upper_log_return = float(reg_upper.predict(test_row[predictors])[0])
    
    # Natively align direction with regressor outcome
    direction = 1 if mean_log_return > 0 else 0
    
    # Extract classifier confidence
    proba = clf.predict_proba(test_row[predictors])[0]
    if len(clf.classes_) == 2 and clf.classes_[1] == 1:
        prob_up = float(proba[1])
    else:
        prob_up = 1.0 if clf.classes_[0] == 1 else 0.0
        
    dir_conf_final = prob_up if direction == 1 else (1.0 - prob_up)
    
    # Process Bounds
    margin = abs((upper_log_return - lower_log_return) / 2.0)
    final_upper_log = mean_log_return + margin
    final_lower_log = mean_log_return - margin
        
    forecasted_amount = float(today_val * np.exp(mean_log_return))
    amount_lower = float(today_val * np.exp(final_lower_log))
    amount_upper = float(today_val * np.exp(final_upper_log))
    
    # Enforce minimum visual margin (max of 2% or $0.01) to prevent flat bounds in UI
    min_margin = max(forecasted_amount * 0.02, 0.01)
    
    if (amount_upper - forecasted_amount) < min_margin:
        amount_upper = forecasted_amount + min_margin
        amount_lower = max(0.0, forecasted_amount - min_margin) # Prevent negative values
    
    return {
        "Direction": "Up" if direction == 1 else "Down",
        "Direction_Confidence": round(dir_conf_final * 100, 2),
        "Amount": round(forecasted_amount, 2),
        "Amount_Lower": round(amount_lower, 2),
        "Amount_Upper": round(amount_upper, 2)
    }

def fit_models(clf_base, reg_median, reg_lower, reg_upper, valid_all, predictors, col_class, col_reg):
    """Helper to fit classifiers and quantile regressors."""
    class_counts = valid_all[col_class].value_counts()
    min_class_count = class_counts.min() if len(class_counts) > 1 else 0
    
    if min_class_count >= 2:
        cv_folds = min(3, min_class_count)
        clf = CalibratedClassifierCV(estimator=clf_base, method='isotonic', cv=cv_folds)
    else:
        clf = clf_base
        
    clf.fit(valid_all[predictors], valid_all[col_class])
    reg_median.fit(valid_all[predictors], valid_all[col_reg])
    reg_lower.fit(valid_all[predictors], valid_all[col_reg])
    reg_upper.fit(valid_all[predictors], valid_all[col_reg])
    return clf

def init_models(lr, md, msl, iters):
    """Helper to instantiate base classifier and quantile regressors."""
    clf_base = HistGradientBoostingClassifier(
        learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=iters,
        class_weight="balanced", random_state=1
    )
    reg_median = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.5, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=iters, random_state=1
    )
    reg_lower = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.1, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=iters, random_state=1
    )
    reg_upper = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.9, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=iters, random_state=1
    )
    return clf_base, reg_median, reg_lower, reg_upper

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
            us_bday = get_us_bday()
            all_future_dates = pd.date_range(start=anchor_date + us_bday, periods=252, freq=us_bday)
        keys = list(range(1, len(all_future_dates) + 1))

    pts_median = sorted(horizon_anchors.items())
    pts_lower = sorted(anchors_lower.items())
    pts_upper = sorted(anchors_upper.items())

    def interp_amount(t, anchors):
        for i in range(len(anchors) - 1):
            t0, p0 = anchors[i]
            t1, p1 = anchors[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                return float(np.exp(np.log(p0) + frac * (np.log(p1) - np.log(p0))))
        return anchors[-1][1]

    dates, prices, upper, lower = [], [], [], []
    for i, t in enumerate(keys):
        amount_t = round(interp_amount(t, pts_median), 2)
        lower_bound = round(interp_amount(t, pts_lower), 2)
        upper_bound = round(interp_amount(t, pts_upper), 2)
        
        dates.append(all_future_dates[i].strftime('%Y-%m-%d'))
        prices.append(amount_t)
        upper.append(upper_bound)
        lower.append(lower_bound)

    return dates, prices, upper, lower
