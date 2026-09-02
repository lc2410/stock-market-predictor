"""Dividend forecasting module using gradient-boosted quantile regression."""

import pandas as pd
import numpy as np
from utils.ml_model_utils import extract_quantiles_metrics, init_models, fit_models, calculate_log_returns

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
    if pd.isna(avg_days):
        avg_days = 90
    
    for h_payouts, label in enumerate(labels, start=1):
        col_reg = f"Target_{h_payouts}"
        col_class = f"Class_{h_payouts}"
        
        divs[col_reg] = calculate_log_returns(divs, col="Dividends", forward=True, periods=h_payouts)
        divs[col_class] = (divs[col_reg] > 0).astype(int)
        
        valid_all = divs.iloc[:-1].dropna(subset=div_predictors + [col_reg, col_class])
        clf_base, reg_median, reg_lower, reg_upper = init_models(0.05, 6, 3, 150)
        
        if len(valid_all) >= 10:
            clf = fit_models(clf_base, reg_median, reg_lower, reg_upper, valid_all, div_predictors, col_class, col_reg)
            
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

def run_dividend_prediction(div_data_raw, anchor_date, div_window=25):
    """
    Orchestrates the dividend forecasting model using raw dividend data.
    """
    divs, div_predictors, next_dividend_date = _engineer_div_features(div_data_raw, anchor_date, div_window)
    
    div_forecasts = {}
    d_anchors = {}
    d_lower = {}
    d_upper = {}
    train_fit_div_dates = []
    train_fit_div_amounts = []
    avg_days_between = 90
    
    if divs is not None:
        avg_days_between = divs.index.to_series().diff().mean().days
        if pd.isna(avg_days_between) or avg_days_between <= 0:
            avg_days_between = 90 
            
        div_forecasts, d_anchors, d_lower, d_upper, train_fit_div_dates, train_fit_div_amounts = _train_multi_horizon_div(
            divs, div_predictors, div_window
        )
        
    return {
        "div_forecasts": div_forecasts,
        "d_anchors": d_anchors,
        "d_lower": d_lower,
        "d_upper": d_upper,
        "train_fit_div_dates": train_fit_div_dates,
        "train_fit_div_amounts": train_fit_div_amounts,
        "next_dividend_date": next_dividend_date,
        "avg_days_between": avg_days_between,
        "has_enough_div_data": divs is not None
    }
