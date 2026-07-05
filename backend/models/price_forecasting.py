import pandas as pd
import numpy as np
from models.utils.forecasting_model_utils import get_us_bday, extract_quantiles_metrics, init_models, fit_models

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
        clf_base, reg_median, reg_lower, reg_upper = init_models(lr, md, msl, 200)
        
        if len(valid_all) > 15:
            clf = fit_models(clf_base, reg_median, reg_lower, reg_upper, valid_all, predictors, col_class, col_reg)
            
            if h_days == 1:
                test_preds = reg_median.predict(valid_all[predictors])
                test_fit_prices = [round(float(x), 2) for x in (valid_all["Close"].values * np.exp(test_preds))]
                if is_crypto:
                    shifted_idx = valid_all.index + pd.Timedelta(days=1)
                else:
                    us_bday = get_us_bday()
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
