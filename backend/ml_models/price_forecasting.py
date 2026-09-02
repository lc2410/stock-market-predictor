"""Price forecasting module using gradient-boosted quantile regression."""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from utils.ml_model_utils import extract_quantiles_metrics, init_models, fit_models, calculate_log_returns

def _calculate_rsi(data, window=14):
    """Calculates Relative Strength Index."""
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=window, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)

def _calculate_macd_hist(data, fast=12, slow=26, signal=9):
    """Calculates MACD Histogram."""
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def _calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculates Bollinger Bands width and position."""
    roll_mean = data["Close"].rolling(window).mean()
    roll_std = data["Close"].rolling(window).std()
    bb_width = (2 * num_std * roll_std) / (roll_mean + 1e-9)
    bb_pos = (data["Close"] - (roll_mean - num_std * roll_std)) / (2 * num_std * roll_std + 1e-9)
    return bb_width, bb_pos

def _calculate_drawdown(data, window):
    """Calculates Drawdown from rolling maximum."""
    roll_max = data["Close"].rolling(window, min_periods=1).max()
    return (data["Close"] - roll_max) / roll_max

def _engineer_price_features(data):
    """Computes technical indicators for the ML price model predictors."""
    price_data = data[["Close", "Volume"]].copy()
    
    price_data["Log_Return"] = calculate_log_returns(price_data)
    
    price_data["Return_Lag_1"] = price_data["Log_Return"].shift(1)
    price_data["Return_Lag_2"] = price_data["Log_Return"].shift(2)
    price_data["Return_Lag_3"] = price_data["Log_Return"].shift(3)

    price_data["RSI_5"] = _calculate_rsi(price_data, window=5)
    price_data["RSI_14"] = _calculate_rsi(price_data, window=14)

    price_data["MACD_Hist"] = _calculate_macd_hist(price_data)

    price_data["BB_Width"], price_data["BB_Pos"] = _calculate_bollinger_bands(price_data, window=20, num_std=2)

    rolling_vol = price_data["Volume"].rolling(10).mean()
    price_data["Vol_Ratio_10"] = np.where(rolling_vol > 0, price_data["Volume"] / rolling_vol, 1.0)

    price_data["SMA_Ratio_50"] = price_data["Close"] / price_data["Close"].rolling(50).mean()
    price_data["SMA_Ratio_200"] = price_data["Close"] / price_data["Close"].rolling(200).mean()
    
    price_data["Hist_Vol_20"] = price_data["Log_Return"].rolling(20).std()
    
    # Rate of Change (ROC)
    price_data["ROC_10"] = price_data["Close"].pct_change(10)
    price_data["ROC_21"] = price_data["Close"].pct_change(21)
    
    price_data["Drawdown_50"] = _calculate_drawdown(price_data, window=50)
    price_data["Drawdown_200"] = _calculate_drawdown(price_data, window=200)

    predictors = [
        "Log_Return", "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
        "RSI_5", "RSI_14", "MACD_Hist", 
        "BB_Width", "BB_Pos", "Vol_Ratio_10",
        "SMA_Ratio_50", "SMA_Ratio_200",
        "Hist_Vol_20", "ROC_10", "ROC_21", "Drawdown_50", "Drawdown_200"
    ]

    return price_data, predictors

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
        
        price_data[col_reg] = calculate_log_returns(price_data, forward=True, periods=h_days)
        price_data[col_class] = (price_data[col_reg] > 0).astype(int)
        
        is_short_term = h_days <= 30
        
        # Restrict tree depth and learning rate for short-term horizons to prevent overfitting
        learning_rate = 0.02 if is_short_term else 0.05
        max_depth = 5 if is_short_term else 10
        min_samples_leaf = 20 if is_short_term else 10
        
        valid_all = price_data.iloc[:-1].dropna(subset=predictors + [col_reg, col_class])
        clf_base, reg_median, reg_lower, reg_upper = init_models(learning_rate, max_depth, min_samples_leaf, 200)
        
        if len(valid_all) > 15:
            clf = fit_models(clf_base, reg_median, reg_lower, reg_upper, valid_all, predictors, col_class, col_reg)
            
            if h_days == 1:
                test_preds = reg_median.predict(valid_all[predictors])
                test_fit_prices = [round(float(x), 2) for x in (valid_all["Close"].values * np.exp(test_preds))]
                if is_crypto:
                    shifted_idx = valid_all.index + pd.Timedelta(days=1)
                else:
                    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
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

def run_price_prediction(price_data_raw, is_crypto=False, price_window=1260):
    """
    Orchestrates the price forecasting model using raw price data.
    """
    has_enough_price_data = len(price_data_raw) >= (price_window + (365 if is_crypto else 252))
    
    if has_enough_price_data:
        price_data, predictors = _engineer_price_features(price_data_raw)
        
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
        
    return {
        "price_forecasts": price_forecasts,
        "p_anchors": p_anchors,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "train_fit_dates": train_fit_dates,
        "train_fit_prices": train_fit_prices,
        "has_enough_price_data": has_enough_price_data
    }
