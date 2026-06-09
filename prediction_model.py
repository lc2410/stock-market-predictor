import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging

logging.getLogger('yfinance').setLevel(logging.ERROR)

def _get_us_bday():
    return CustomBusinessDay(calendar=USFederalHolidayCalendar())

def get_chart_data(ticker_symbol, predicted_price=None):
    """Fetches 1 year of price history and the last 12 dividend payouts for chart rendering."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")

        if not hist.empty:
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep='last')]

        dates = hist.index.strftime('%Y-%m-%d').tolist() if not hist.empty else []
        prices = hist['Close'].tolist() if not hist.empty else []

        if predicted_price is not None and not hist.empty:
            anchor_date = hist.index[-1]
            dates.append((anchor_date + _get_us_bday()).strftime('%Y-%m-%d'))
            prices.append(float(predicted_price))

        divs = ticker.dividends.tail(12)
        if not divs.empty:
            divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
            divs = divs[~divs.index.duplicated(keep='last')]

        return {
            "dates": dates,
            "prices": prices,
            "dividend_dates": divs.index.strftime('%Y-%m-%d').tolist() if not divs.empty else [],
            "dividend_amounts": divs.values.tolist() if not divs.empty else [],
        }
    except Exception as e:
        print(f"Error fetching chart data: {e}")
        return None

def _fetch_data(ticker):
    """Grab max available history and enforce a minimum row count for training."""
    stock_ticker = yf.Ticker(ticker)
    data = stock_ticker.history(period="max")

    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'.")
        return None

    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep='last')]

    data = data.loc["2010-01-01":].copy()

    if len(data) < 1261:
        print(f"Error: Not enough historical data for {ticker}.")
        return None

    return data

def _engineer_features(data):
    """Builds momentum, volatility, RSI, MACD, Bollinger Band, ATR, and volume ratio features for the price model."""
    price_data = data[["Close", "Volume", "High", "Low"]].copy()
    price_data["Tomorrow"] = price_data["Close"].shift(-1)
    price_data["Price_Target"] = (price_data["Tomorrow"] > price_data["Close"]).astype(int)
    price_data["Return"] = price_data["Close"].pct_change()

    predictors = []

    for h in [2, 5, 10, 20, 63, 126, 252, 504, 1260]:
        price_data[f"Vol_{h}"] = price_data["Return"].rolling(h).std()
        price_data[f"Mom_{h}"] = price_data["Return"].rolling(h).sum()
        price_data[f"Close_Ratio_{h}"] = price_data["Close"] / price_data["Close"].rolling(h).mean()
        predictors += [f"Vol_{h}", f"Mom_{h}", f"Close_Ratio_{h}"]

    for lag in range(1, 6):
        price_data[f"Return_Lag_{lag}"] = price_data["Return"].shift(lag)
        predictors.append(f"Return_Lag_{lag}")

    delta = price_data["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    price_data["RSI_14"] = (100 - (100 / (1 + gain / loss))).fillna(100)
    predictors.append("RSI_14")

    ema_12 = price_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_data["Close"].ewm(span=26, adjust=False).mean()
    price_data["MACD_Line"] = ema_12 - ema_26
    price_data["MACD_Signal"] = price_data["MACD_Line"].ewm(span=9, adjust=False).mean()
    price_data["MACD_Hist"] = price_data["MACD_Line"] - price_data["MACD_Signal"]
    predictors += ["MACD_Line", "MACD_Signal", "MACD_Hist"]

    for bb_w in [20, 50]:
        roll_mean = price_data["Close"].rolling(bb_w).mean()
        roll_std = price_data["Close"].rolling(bb_w).std()
        upper = roll_mean + 2 * roll_std
        lower = roll_mean - 2 * roll_std
        price_data[f"BB_Pos_{bb_w}"] = (price_data["Close"] - lower) / (upper - lower + 1e-9)
        price_data[f"BB_Width_{bb_w}"] = (upper - lower) / (roll_mean + 1e-9)
        predictors += [f"BB_Pos_{bb_w}", f"BB_Width_{bb_w}"]

    hl = price_data["High"] - price_data["Low"]
    hc = (price_data["High"] - price_data["Close"].shift()).abs()
    lc = (price_data["Low"] - price_data["Close"].shift()).abs()
    price_data["ATR_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean() / price_data["Close"]
    predictors.append("ATR_14")

    price_data["Vol_Ratio_10"] = price_data["Volume"] / price_data["Volume"].rolling(10).mean()
    price_data["Vol_Ratio_20"] = price_data["Volume"] / price_data["Volume"].rolling(20).mean()
    predictors += ["Vol_Ratio_10", "Vol_Ratio_20"]

    price_data["HL_Range"] = (price_data["High"] - price_data["Low"]) / price_data["Close"]
    predictors.append("HL_Range")

    price_data = price_data.dropna(subset=predictors)

    return price_data, predictors

def _prune_features(train, predictors, ticker):
    """Drop the bottom 25% of features to reduce noise before calibration."""
    quick_rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1, n_jobs=-1)
    quick_rf.fit(train[predictors], train["Price_Target"])
    importances = pd.Series(quick_rf.feature_importances_, index=predictors)
    pruned = importances[importances >= importances.quantile(0.25)].index.tolist()
    print(f"[{ticker}] Features after pruning: {len(pruned)}")
    return pruned

def _train_price_classifier(train, test, predictors, ticker):
    """Tune RF via RandomizedSearchCV and apply isotonic calibration for real confidence percentages."""
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=1, n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, 12, None],
         "min_samples_leaf": [5, 10, 20, 30], "max_features": ["sqrt", "log2", 0.5]},
        n_iter=8, cv=tscv, scoring="roc_auc", random_state=1, n_jobs=1,
    )
    search.fit(train[predictors], train["Price_Target"])
    best_rf = search.best_estimator_
    print(f"[{ticker}] Best classifier params: {search.best_params_}")

    calibrated_clf = CalibratedClassifierCV(best_rf, method="isotonic", cv=TimeSeriesSplit(n_splits=5), n_jobs=1)
    calibrated_clf.fit(train[predictors], train["Price_Target"])

    direction = int(calibrated_clf.predict(test[predictors])[0])
    prob_up = float(calibrated_clf.predict_proba(test[predictors])[0, 1])

    confidence = prob_up if direction == 1 else 1.0 - prob_up
    return direction, confidence

def _train_price_regressor(train, test, predictors, direction, today_close, ticker, target_dates):
    """Predict the magnitude of the move and align the dollar amount to the classifier's direction."""
    tscv = TimeSeriesSplit(n_splits=3)
    search_reg = RandomizedSearchCV(
        RandomForestRegressor(random_state=1, n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None], "min_samples_leaf": [5, 10, 20]},
        n_iter=8, cv=tscv, scoring="neg_mean_absolute_error", random_state=1, n_jobs=1,
    )
    search_reg.fit(train[predictors], train["Tomorrow"])
    price_reg = search_reg.best_estimator_
    print(f"[{ticker}] Best regressor params: {search_reg.best_params_}")

    forecasted_close = float(price_reg.predict(test[predictors])[0])

    train_fitted = price_reg.predict(train[predictors])
    train_fit_dates = target_dates.strftime("%Y-%m-%d").tolist()
    train_fit_prices = [round(float(p), 2) for p in train_fitted]

    if direction == 1:
        forecasted_close = max(forecasted_close, today_close * 1.001)
    else:
        forecasted_close = min(forecasted_close, today_close * 0.999)

    return round(forecasted_close, 2), train_fit_dates, train_fit_prices

def _forecast_long_term(price_data, test_row, predictors, today_close, forecast_close, price_window, anchor_date):
    """Projects prices out 252 trading days using log-interpolation between trained horizon anchors (next day, 1 week, 1 month, 1 year)."""
    daily_vol = price_data["Return"].std()
    us_bday = _get_us_bday()
    all_future_dates = pd.date_range(
        start=anchor_date + us_bday, periods=252, freq=us_bday
    )

    horizon_prices = {0: today_close, 1: forecast_close}
    for h_label, h_days in [("1_Week", 5), ("1_Month", 21), ("1_Year", 252)]:
        lt_data = price_data[["Close"] + predictors].copy()
        lt_data["Future_Log_Return"] = np.log(lt_data["Close"].shift(-h_days) / lt_data["Close"])
        lt_data = lt_data.dropna()

        if len(lt_data) > 100:
            lt_reg = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=1, n_jobs=-1
            )
            lt_reg.fit(lt_data.iloc[-price_window:][predictors], lt_data.iloc[-price_window:]["Future_Log_Return"])
            clip_bound = 0.90 if h_days == 252 else 0.60
            pred_log_return = np.clip(float(lt_reg.predict(test_row[predictors])[0]), -clip_bound, clip_bound)
            horizon_prices[h_days] = float(round(today_close * np.exp(pred_log_return), 2))
        else:
            horizon_prices[h_days] = today_close

    anchors = sorted(horizon_prices.items())

    def interp_price(t):
        # Log-linear interpolation between anchor points keeps price changes multiplicative rather than additive
        for i in range(len(anchors) - 1):
            t0, p0 = anchors[i]
            t1, p1 = anchors[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                return float(np.exp(np.log(p0) + frac * (np.log(p1) - np.log(p0))))
        return anchors[-1][1]

    chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower = [], [], [], []
    for t, future_date in enumerate(all_future_dates, start=1):
        price_t = round(interp_price(t), 2)
        moe_t = 1.96 * daily_vol * np.sqrt(t)
        chart_future_dates.append(future_date.strftime("%Y-%m-%d"))
        chart_future_prices.append(price_t)
        chart_future_upper.append(float(round(price_t * (1 + moe_t), 2)))
        chart_future_lower.append(float(round(price_t * (1 - moe_t), 2)))

    extended_forecasts = {
        label: {
            "Date": chart_future_dates[idx],
            "Price": chart_future_prices[idx],
            "Upper": chart_future_upper[idx],
            "Lower": chart_future_lower[idx],
        }
        for label, idx in [("1_Week", 4), ("1_Month", 20), ("1_Year", 251)]
    }

    return chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower, extended_forecasts

def _engineer_div_features(data, anchor_date):
    """Extracts dividend history, projects the next payout date, and builds growth/rolling features for the dividend models."""
    divs = data[["Dividends"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()

    if len(divs) < 10:
        return None, None, pd.NaT, 0.0

    last_div_date = divs.index[-1]
    avg_days_between = divs.index.to_series().diff().mean().days
    
    if pd.isna(avg_days_between) or avg_days_between <= 0:
        avg_days_between = 90  # Reasonable quarterly fallback for sparse dividend histories

    projected_date = last_div_date + pd.Timedelta(days=avg_days_between)
    while projected_date <= anchor_date:
        projected_date += pd.Timedelta(days=avg_days_between)
    next_dividend_date = projected_date

    divs["Next_Dividend"] = divs["Dividends"].shift(-1)
    divs["Div_Target"] = (divs["Next_Dividend"] > divs["Dividends"]).astype(int)
    divs["Div_Growth_1"] = divs["Dividends"].pct_change(1)
    divs["Div_Growth_2"] = divs["Dividends"].pct_change(2)
    
    for w in [4, 6, 8]:
        divs[f"Rolling_Avg_{w}"] = divs["Dividends"].rolling(w).mean()
        divs[f"Rolling_Std_{w}"] = divs["Dividends"].rolling(w).std()

    div_predictors = [
        "Dividends", "Div_Growth_1", "Div_Growth_2",
        "Rolling_Avg_4", "Rolling_Std_4", "Rolling_Avg_6",
        "Rolling_Std_6", "Rolling_Avg_8", "Rolling_Std_8",
    ]
    
    today_div = float(divs["Dividends"].iloc[-1])
    divs = divs.dropna(subset=div_predictors)

    return divs, div_predictors, next_dividend_date, today_div

def _train_div_classifier(train_div, test_div, div_predictors):
    # Can't train a classifier when every row has the same label — just return neutral confidence
    if train_div["Div_Target"].nunique() <= 1:
        div_pred = int(train_div["Div_Target"].iloc[0])
        return div_pred, 0.5

    try:
        cal_div = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1),
            method="isotonic", cv=TimeSeriesSplit(n_splits=3), n_jobs=1,
        )
        cal_div.fit(train_div[div_predictors], train_div["Div_Target"])
        div_pred = int(cal_div.predict(test_div[div_predictors])[0])
        div_conf_up = float(cal_div.predict_proba(test_div[div_predictors])[0, 1])
    except ValueError:
        # Isotonic calibration needs at least 2 classes per fold; fall back to uncalibrated RF if splits are too small
        div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
        div_pred = int(div_clf.predict(test_div[div_predictors])[0])
        div_conf_up = (
            float(div_clf.predict_proba(test_div[div_predictors])[0, 1])
            if len(div_clf.classes_) == 2
            else 0.5
        )

    return div_pred, div_conf_up

def _train_div_regressor(train_div, test_div, div_predictors, div_pred, today_div):
    """Trains a basic RF regressor on dividend history and nudges the output to stay consistent with the classifier's direction."""
    div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
    div_reg.fit(train_div[div_predictors], train_div["Next_Dividend"])
    forecasted_div = float(div_reg.predict(test_div[div_predictors])[0])

    forecasted_div = (
        max(forecasted_div, today_div * 1.001) if div_pred == 1
        else min(forecasted_div, today_div * 0.999)
    )
    return round(forecasted_div, 2)

def _forecast_div_long_term(divs, div_predictors, test_div, today_div, forecasted_div, next_dividend_date, avg_days_between, div_window):
    """Forecasts the next 4 dividend cycles, training a separate regressor per cycle and widening the CI with sqrt(cycle)."""
    payout_vol = divs["Dividends"].pct_change().std()

    div_future_dates  = [next_dividend_date.strftime("%Y-%m-%d")]
    div_future_amounts = [round(forecasted_div, 2)] 
    moe_1 = 1.96 * payout_vol * np.sqrt(1)
    div_future_upper  = [round(forecasted_div * (1 + moe_1), 2)] 
    div_future_lower  = [round(max(forecasted_div * (1 - moe_1), 0.0), 2)] 

    for cycle in range(2, 5):
        target_col = f"Div_Ahead_{cycle}"
        lt_divs = divs[div_predictors].copy()
        lt_divs[target_col] = lt_divs["Dividends"].shift(-cycle)
        lt_divs = lt_divs.dropna()

        if len(lt_divs) > 8:
            lt_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
            lt_reg.fit(lt_divs.iloc[-div_window:][div_predictors], lt_divs.iloc[-div_window:][target_col])
            predicted_amount = round(float(lt_reg.predict(test_div[div_predictors])[0]), 2) 
        else:
            predicted_amount = round(today_div, 2) 

        moe = 1.96 * payout_vol * np.sqrt(cycle)
        projected_date = next_dividend_date + pd.Timedelta(days=avg_days_between * (cycle - 1))

        div_future_dates.append(projected_date.strftime("%Y-%m-%d"))
        div_future_amounts.append(predicted_amount)
        div_future_upper.append(round(predicted_amount * (1 + moe), 2)) 
        div_future_lower.append(round(max(predicted_amount * (1 - moe), 0.0), 2)) 

    div_extended_forecasts = {
        label: {
            "Date": div_future_dates[idx],
            "Amount": div_future_amounts[idx],
            "Upper": div_future_upper[idx],
            "Lower": div_future_lower[idx],
        }
        for label, idx in [("2_Payouts", 1), ("3_Payouts", 2), ("4_Payouts", 3)]
    }

    return div_future_dates, div_future_amounts, div_future_upper, div_future_lower, div_extended_forecasts

def run_real_time_model(ticker, price_window=1000, div_window=20):
    """Main pipeline orchestrator: ties the classifiers, regressors, and history fetchers together."""
    data = _fetch_data(ticker)
    if data is None:
        return None

    anchor_date = data.index[-1]
    next_trading_day_str = (anchor_date + _get_us_bday()).strftime('%Y-%m-%d')

    price_data, predictors = _engineer_features(data)
    train_price = price_data.iloc[-(price_window + 1):-1]
    test_price = price_data.iloc[-1:].copy()
    today_close = float(test_price["Close"].values[0])
    
    predictors = _prune_features(train_price, predictors, ticker)
    direction, price_conf = _train_price_classifier(train_price, test_price, predictors, ticker)

    target_dates = price_data.index[-price_window:]

    forecasted_close, train_fit_dates, train_fit_prices = _train_price_regressor(
        train_price, test_price, predictors, direction, today_close, ticker, target_dates
    )

    chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower, extended_forecasts = (
        _forecast_long_term(price_data, test_price, predictors, today_close, forecasted_close, price_window, anchor_date)
    )

    divs, div_predictors, next_dividend_date, today_div = _engineer_div_features(data, anchor_date)
    
    forecasted_div = "N/A"
    div_direction_text = "N/A"
    final_div_conf = "N/A"
    div_future_dates, div_future_amounts = [], []
    div_future_upper, div_future_lower = [], []
    div_extended_forecasts = {}

    if divs is not None:
        avg_days_between = divs.index.to_series().diff().mean().days
        if pd.isna(avg_days_between) or avg_days_between <= 0:
            avg_days_between = 90  # Reasonable quarterly fallback for sparse dividend histories
            
        train_div = divs.iloc[-(div_window + 1):-1]
        test_div = divs.iloc[-1:].copy()

        div_pred, div_conf_up = _train_div_classifier(train_div, test_div, div_predictors)
        forecasted_div = _train_div_regressor(train_div, test_div, div_predictors, div_pred, today_div)

        if div_pred == 1:
            div_direction_text = "Up"
            final_div_conf = round(div_conf_up * 100, 2)
        else:
            div_direction_text = "Down"
            final_div_conf = round((1.0 - div_conf_up) * 100, 2)

        div_future_dates, div_future_amounts, div_future_upper, div_future_lower, div_extended_forecasts = (
            _forecast_div_long_term(divs, div_predictors, test_div, today_div, forecasted_div, next_dividend_date, avg_days_between, div_window)
        )

    forecasted_yield = (
        round((forecasted_div / today_close) * 100, 2)
        if today_close > 0 and forecasted_div != "N/A"
        else "N/A"
    )

    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    return pd.DataFrame({
        "Next_Trading_Day":        [next_trading_day_str],
        "Ticker":                  [ticker],
        "Company_Name":            [company_name],
        "Price_Predicted":         ["Up" if direction == 1 else "Down"],
        "Price_Confidence (%)":    [round(price_conf * 100, 2)],
        "Forecasted_Close":        [forecasted_close],
        "Next_Dividend_Date":      [next_dividend_date.strftime("%Y-%m-%d") if pd.notna(next_dividend_date) else "N/A"],
        "Div_Predicted":           [div_direction_text],
        "Div_Confidence (%)":      [final_div_conf],
        "Forecasted_Dividend":     [forecasted_div],
        "Forecasted_Yield (%)":    [forecasted_yield],
        "Extended_Forecasts":      [extended_forecasts],
        "Chart_Future_Dates":      [chart_future_dates],
        "Chart_Future_Prices":     [chart_future_prices],
        "Chart_Future_Upper":      [chart_future_upper],
        "Chart_Future_Lower":      [chart_future_lower],
        "Train_Fit_Dates":         [train_fit_dates],
        "Train_Fit_Prices":        [train_fit_prices],
        "Div_Extended_Forecasts":  [div_extended_forecasts],
        "Div_Future_Dates":        [div_future_dates],
        "Div_Future_Amounts":      [div_future_amounts],
        "Div_Future_Upper":        [div_future_upper],
        "Div_Future_Lower":        [div_future_lower],
    })