import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging

logging.getLogger('yfinance').setLevel(logging.ERROR)

def _get_us_bday():
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

def _fetch_data(ticker, is_crypto=False):
    """
    Fetches historical stock data from Yahoo Finance.
    Adaptively fetches data incrementally (5 to 30 years) to balance API latency and data completeness.
    """
    stock_ticker = yf.Ticker(ticker)
    
    target_window = 1825 if is_crypto else 1260
    buffer_days = 365 if is_crypto else 252
    min_required_days = target_window + buffer_days
    
    years_to_fetch = 5
    data = None
    dividends = None
    
    while years_to_fetch <= 30:
        data = stock_ticker.history(period=f"{years_to_fetch}y")
        
        if data.empty:
            return None, None, None
            
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
        return None, None, None
        
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
        
    return price_data_slice, div_data_slice, target_window

def _engineer_price_features(data):
    """Computes technical indicators for the ML price model predictors."""
    price_data = data[["Close", "Volume"]].copy()
    
    # Log Returns
    price_data["Log_Return"] = np.log(price_data["Close"] / price_data["Close"].shift(1))
    
    # Lagged Returns
    price_data["Return_Lag_1"] = price_data["Log_Return"].shift(1)
    price_data["Return_Lag_2"] = price_data["Log_Return"].shift(2)
    price_data["Return_Lag_3"] = price_data["Log_Return"].shift(3)

    # RSI (5-day and 14-day)
    delta = price_data["Close"].diff()
    
    gain_5 = delta.where(delta > 0, 0).ewm(span=5, adjust=False).mean()
    loss_5 = (-delta.where(delta < 0, 0)).ewm(span=5, adjust=False).mean()
    price_data["RSI_5"] = (100 - (100 / (1 + gain_5 / loss_5))).fillna(100)
    
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    price_data["RSI_14"] = (100 - (100 / (1 + gain / loss))).fillna(100)

    # MACD Histogram
    ema_12 = price_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_data["Close"].ewm(span=26, adjust=False).mean()
    price_data["MACD_Hist"] = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    roll_mean = price_data["Close"].rolling(20).mean()
    roll_std = price_data["Close"].rolling(20).std()
    price_data["BB_Width"] = (4 * roll_std) / (roll_mean + 1e-9)
    price_data["BB_Pos"] = (price_data["Close"] - (roll_mean - 2 * roll_std)) / (4 * roll_std + 1e-9)

    # Volume Ratio
    rolling_vol = price_data["Volume"].rolling(10).mean()
    price_data["Vol_Ratio_10"] = np.where(rolling_vol > 0, price_data["Volume"] / rolling_vol, 1.0)

    # SMA Ratios
    price_data["SMA_Ratio_50"] = price_data["Close"] / price_data["Close"].rolling(50).mean()
    price_data["SMA_Ratio_200"] = price_data["Close"] / price_data["Close"].rolling(200).mean()
    
    # Historical Volatility
    price_data["Hist_Vol_20"] = price_data["Log_Return"].rolling(20).std()
    
    # Rate of Change (ROC)
    price_data["ROC_10"] = price_data["Close"].pct_change(10)
    price_data["ROC_21"] = price_data["Close"].pct_change(21)
    
    # Drawdown
    roll_max_50 = price_data["Close"].rolling(50, min_periods=1).max()
    price_data["Drawdown_50"] = (price_data["Close"] - roll_max_50) / roll_max_50
    roll_max_200 = price_data["Close"].rolling(200, min_periods=1).max()
    price_data["Drawdown_200"] = (price_data["Close"] - roll_max_200) / roll_max_200

    predictors = [
        "Log_Return", "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
        "RSI_5", "RSI_14", "MACD_Hist", 
        "BB_Width", "BB_Pos", "Vol_Ratio_10",
        "SMA_Ratio_50", "SMA_Ratio_200",
        "Hist_Vol_20", "ROC_10", "ROC_21", "Drawdown_50", "Drawdown_200"
    ]

    return price_data, predictors

def _engineer_div_features(data, anchor_date, div_window=25):
    """
    Extracts dividend payouts and calculates predictors (rolling averages, growth).
    Returns None if fewer than `div_window` payouts exist to trigger UI fallback.
    """
    # 1-year trailing price return
    data["Price_Return_252"] = data["Close"].pct_change(252)
    divs = data[["Dividends", "Price_Return_252"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()

    if len(divs) < div_window:
        return None, None, pd.NaT

    last_div_date = divs.index[-1]
    avg_days_between = divs.index.to_series().diff().mean().days
    
    if pd.isna(avg_days_between) or avg_days_between <= 0:
        avg_days_between = 90

    projected_date = last_div_date + pd.Timedelta(days=avg_days_between)
    while projected_date <= anchor_date:
        projected_date += pd.Timedelta(days=avg_days_between)
    next_dividend_date = projected_date

    # Period-over-period growth
    divs["Div_Growth_1"] = divs["Dividends"].pct_change(1)
    # Rolling 4-payout average
    divs["Rolling_Avg_4"] = divs["Dividends"].rolling(4).mean()

    div_predictors = ["Price_Return_252", "Div_Growth_1", "Rolling_Avg_4"]

    return divs, div_predictors, next_dividend_date

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

def _train_multi_horizon_price(price_data, predictors, is_crypto, price_window):
    """
    Trains regressors and classifiers across multiple horizons to forecast future prices.
    Uses quantile regression to construct the expected range bounds.
    """
    horizons = [1, 7, 30, 90, 180, 270, 365] if is_crypto else [1, 5, 21, 63, 126, 189, 252]
    labels = ["Next_Day", "Next_Week", "Next_Month", "Next_3_Months", "Next_6_Months", "Next_9_Months", "Next_Year"]
    
    # Trim the dataset to the requested training window length (plus any needed buffers)
    price_data = price_data.iloc[-(price_window + 1):].copy()
    
    # Extract the last row as the anchor point to forecast from
    test_row = price_data.iloc[-1:].copy()
    today_close = float(test_row["Close"].values[0])
    
    results = {}
    test_fit_dates = []
    test_fit_prices = []
    horizon_anchors = {0: today_close}
    horizon_anchors_lower = {0: today_close}
    horizon_anchors_upper = {0: today_close}
    
    for h_days, label in zip(horizons, labels):
        col_reg = f"Target_{h_days}"
        col_class = f"Class_{h_days}"
        
        price_data[col_reg] = np.log(price_data["Close"].shift(-h_days) / price_data["Close"])
        price_data[col_class] = (price_data[col_reg] > 0).astype(int)
        
        is_short_term = h_days <= 30
        
        # Restrict tree depth and learning rate for short-term horizons to prevent overfitting
        lr = 0.02 if is_short_term else 0.05
        md = 5 if is_short_term else 10
        msl = 20 if is_short_term else 10
        
        valid_all = price_data.iloc[:-1].dropna(subset=predictors + [col_reg, col_class])
        clf_base = HistGradientBoostingClassifier(
            learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=200,
            class_weight="balanced", random_state=1
        )
        
        reg_median = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.5, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=200, random_state=1
        )
        reg_lower = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.1, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=200, random_state=1
        )
        reg_upper = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.9, learning_rate=lr, max_depth=md, min_samples_leaf=msl, max_iter=200, random_state=1
        )
        
        if len(valid_all) > 15:
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
            
            if h_days == 1:
                test_preds = reg_median.predict(valid_all[predictors])
                test_fit_prices = [round(float(x), 2) for x in (valid_all["Close"].values * np.exp(test_preds))]
                if is_crypto:
                    shifted_idx = valid_all.index + pd.Timedelta(days=1)
                else:
                    us_bday = _get_us_bday()
                    shifted_idx = pd.DatetimeIndex([d + us_bday for d in valid_all.index])
                test_fit_dates = shifted_idx.strftime('%Y-%m-%d').tolist()
            
            metrics = extract_quantiles_metrics(clf, reg_median, reg_lower, reg_upper, test_row, predictors, today_close)
        else:
            # Insufficient data. Return empty results to trigger UI fallback.
            results = {}
            horizon_anchors = {0: today_close}
            horizon_anchors_lower = {0: today_close}
            horizon_anchors_upper = {0: today_close}
            test_fit_dates = []
            test_fit_prices = []
            break
            
        results[label] = metrics
        horizon_anchors[h_days] = metrics["Amount"]
        horizon_anchors_lower[h_days] = metrics["Amount_Lower"]
        horizon_anchors_upper[h_days] = metrics["Amount_Upper"]

    return results, horizon_anchors, horizon_anchors_lower, horizon_anchors_upper, test_fit_dates, test_fit_prices

def _train_multi_horizon_div(divs, div_predictors, div_window):
    """
    Trains regressors and classifiers to forecast future dividend payouts.
    Uses quantile regression to construct expected range bounds.
    """
    labels = ["Next_Payout", "Payout_2", "Payout_3", "Payout_4", "Payout_5"]
    
    # Trim the dataset to the requested training window length
    effective_div_window = min(div_window, len(divs) - 1)
    divs = divs.iloc[-(effective_div_window + 1):].copy()
    
    # Extract the last row as the anchor point to forecast from
    test_row = divs.iloc[-1:].copy()
    
    today_div = float(test_row["Dividends"].values[0])
    results = {}
    test_fit_dates = []
    test_fit_amounts = []
    horizon_anchors = {0: today_div}
    horizon_anchors_lower = {0: today_div}
    horizon_anchors_upper = {0: today_div}
    
    avg_days = divs.index.to_series().diff().mean().days
    if pd.isna(avg_days): avg_days = 90
    
    for h_payouts, label in enumerate(labels, start=1):
        col_reg = f"Target_{h_payouts}"
        col_class = f"Class_{h_payouts}"
        
        divs[col_reg] = np.log(divs["Dividends"].shift(-h_payouts) / divs["Dividends"])
        divs[col_class] = (divs[col_reg] > 0).astype(int)
        
        valid_all = divs.iloc[:-1].dropna(subset=div_predictors + [col_reg, col_class])
        clf_base = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=6, min_samples_leaf=3, max_iter=150,
            class_weight="balanced", random_state=1
        )
        
        reg_median = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.5, learning_rate=0.05, max_depth=6, min_samples_leaf=3, max_iter=150, random_state=1
        )
        reg_lower = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.1, learning_rate=0.05, max_depth=6, min_samples_leaf=3, max_iter=150, random_state=1
        )
        reg_upper = HistGradientBoostingRegressor(
            loss='quantile', quantile=0.9, learning_rate=0.05, max_depth=6, min_samples_leaf=3, max_iter=150, random_state=1
        )
        
        if len(valid_all) >= 10:
            class_counts = valid_all[col_class].value_counts()
            min_class_count = class_counts.min() if len(class_counts) > 1 else 0
            
            if min_class_count >= 2:
                cv_folds = min(3, min_class_count)
                clf = CalibratedClassifierCV(estimator=clf_base, method='isotonic', cv=cv_folds)
            else:
                clf = clf_base
                
            clf.fit(valid_all[div_predictors], valid_all[col_class])
            reg_median.fit(valid_all[div_predictors], valid_all[col_reg])
            reg_lower.fit(valid_all[div_predictors], valid_all[col_reg])
            reg_upper.fit(valid_all[div_predictors], valid_all[col_reg])
            
            if h_payouts == 1 and not valid_all.empty:
                test_preds = reg_median.predict(valid_all[div_predictors])
                test_fit_amounts = [round(float(x), 2) for x in (valid_all["Dividends"].values * np.exp(test_preds))]
                
                idx_positions = [divs.index.get_loc(idx) + 1 for idx in valid_all.index]
                for p in idx_positions:
                    if p < len(divs):
                        test_fit_dates.append(divs.index[p].strftime('%Y-%m-%d'))
                    else:
                        test_fit_dates.append((divs.index[-1] + pd.Timedelta(days=avg_days)).strftime('%Y-%m-%d'))
                        
            metrics = extract_quantiles_metrics(clf, reg_median, reg_lower, reg_upper, test_row, div_predictors, today_div)
        else:
            # Insufficient data. Return empty results to trigger UI fallback.
            results = {}
            horizon_anchors = {0: today_div}
            horizon_anchors_lower = {0: today_div}
            horizon_anchors_upper = {0: today_div}
            test_fit_dates = []
            test_fit_amounts = []
            break
            
        results[label] = metrics
        horizon_anchors[h_payouts] = metrics["Amount"]
        horizon_anchors_lower[h_payouts] = metrics["Amount_Lower"]
        horizon_anchors_upper[h_payouts] = metrics["Amount_Upper"]

    return results, horizon_anchors, horizon_anchors_lower, horizon_anchors_upper, test_fit_dates, test_fit_amounts

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
            us_bday = _get_us_bday()
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

def run_real_time_model(ticker, price_window=1260, div_window=25, is_crypto=False):
    """
    Main orchestration function. Fetches data, engineers features, trains price 
    and dividend forecasting models, generates chart interpolations, and 
    packages the final payload for the frontend API response.
    """
    # Retrieve the historical stock data using period="max" and strictly isolate necessary slices
    price_data_raw, div_data_raw, price_window = _fetch_data(ticker, is_crypto)
    if price_data_raw is None:
        return None

    anchor_date = price_data_raw.index[-1]
    
    # Check if we have enough historical data to run the Price ML pipeline
    has_enough_price_data = len(price_data_raw) >= (price_window + (365 if is_crypto else 252))
    
    if has_enough_price_data:
        # Generate predictive features based on technical indicators for the price model
        price_data, predictors = _engineer_price_features(price_data_raw)
        
        # Train the multi-horizon price forecasting models and generate predicted intervals
        price_forecasts, p_anchors, p_lower, p_upper, train_fit_dates, train_fit_prices = _train_multi_horizon_price(
            price_data, predictors, is_crypto, price_window
        )
    else:
        price_forecasts = {}
        last_price = float(price_data_raw.iloc[-1]["Close"])
        p_anchors = {0: last_price}
        p_lower = {0: last_price}
        p_upper = {0: last_price}
        train_fit_dates, train_fit_prices = [], []
    
    # Create the coordinate map for the unified frontend chart data
    chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower = generate_future_chart_data(
        p_anchors, p_lower, p_upper, anchor_date, is_crypto, is_div=False
    )

    # Process and isolate the historical dividend data and associated predictors from the custom dividend slice
    divs, div_predictors, next_dividend_date = _engineer_div_features(div_data_raw, anchor_date, div_window)
    
    div_forecasts = {}
    div_future_dates, div_future_amounts = [], []
    div_future_upper, div_future_lower = [], []
    train_fit_div_dates, train_fit_div_amounts = [], []

    if divs is not None:
        # Determine the average time interval between payouts to project future dates
        avg_days_between = divs.index.to_series().diff().mean().days
        if pd.isna(avg_days_between) or avg_days_between <= 0:
            avg_days_between = 90 
            
        # Train the multi-horizon dividend forecasting models and generate predicted intervals
        div_forecasts, d_anchors, d_lower, d_upper, train_fit_div_dates, train_fit_div_amounts = _train_multi_horizon_div(
            divs, div_predictors, div_window
        )
        
        # Create the coordinate map for the frontend dividend chart
        div_future_dates, div_future_amounts, div_future_upper, div_future_lower = generate_future_chart_data(
            d_anchors, d_lower, d_upper, anchor_date, is_crypto, is_div=True, avg_days_between=avg_days_between
        )
        
        train_fit_div_dates = train_fit_div_dates[-5:]
        train_fit_div_amounts = train_fit_div_amounts[-5:]

    # Trim price chart training points to past 1 year for rendering efficiency
    days_in_year = 365 if is_crypto else 252
    train_fit_dates = train_fit_dates[-days_in_year:]
    train_fit_prices = train_fit_prices[-days_in_year:]

    # Retrieve the formatted historical chart data for immediate visualization using the price data
    chart_history = get_chart_data(
        price_data=price_data_raw, 
        div_data=div_data_raw,
        is_crypto=is_crypto, 
        show_all_prices=not has_enough_price_data, 
        show_all_divs=(divs is None)
    )

    # Package and return all data structures to be serialized by the API
    return {
        "anchor_date": anchor_date,
        "today_close": float(price_data_raw["Close"].iloc[-1]),
        "next_dividend_date": next_dividend_date,
        
        "price_forecasts": price_forecasts,
        "chart_future_dates": chart_future_dates,
        "chart_future_prices": chart_future_prices,
        "chart_future_upper": chart_future_upper,
        "chart_future_lower": chart_future_lower,
        "train_fit_dates": train_fit_dates,
        "train_fit_prices": train_fit_prices,
        
        "div_forecasts": div_forecasts,
        "div_future_dates": div_future_dates,
        "div_future_amounts": div_future_amounts,
        "div_future_upper": div_future_upper,
        "div_future_lower": div_future_lower,
        "train_fit_div_dates": train_fit_div_dates,
        "train_fit_div_amounts": train_fit_div_amounts,
        
        "chart_history": chart_history
    }