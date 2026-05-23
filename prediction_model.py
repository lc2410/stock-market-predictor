import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import date
import logging

logging.getLogger('yfinance').setLevel(logging.ERROR)

# --- HELPER FUNCTIONS ---

def get_next_trading_day():
    us_holidays = USFederalHolidayCalendar()
    us_business_day = CustomBusinessDay(calendar=us_holidays)
    today = pd.to_datetime(date.today())
    return (today + us_business_day).strftime('%Y-%m-%d')

def get_future_trading_date(days_ahead):
    us_holidays = USFederalHolidayCalendar()
    us_business_day = CustomBusinessDay(calendar=us_holidays)
    future_date = pd.to_datetime(date.today()) + us_business_day * days_ahead
    return future_date.strftime('%Y-%m-%d')

def get_chart_data(ticker_symbol, predicted_price=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="6mo")
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        if predicted_price is not None:
            dates.append(get_next_trading_day())
            prices.append(float(predicted_price))
        divs = ticker.dividends.tail(12)
        return {
            "dates": dates,
            "prices": prices,
            "dividend_dates": divs.index.strftime('%Y-%m-%d').tolist(),
            "dividend_amounts": divs.values.tolist()
        }
    except Exception as e:
        print(f"Error fetching chart data: {e}")
        return None

# --- MAIN MODEL LOGIC ---

def run_real_time_model(ticker, price_window=1000, div_window=20):
    stock_ticker = yf.Ticker(ticker)
    data = stock_ticker.history(period="max")

    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'.")
        return None

    data = data.loc["2010-01-01":].copy()
    if len(data) < 1261:
        print(f"Error: Not enough historical data for {ticker}.")
        return None

    # ----------------------
    # Feature Engineering
    # ----------------------
    # Pull Close, Volume, High, Low — needed for volume and range features
    price_data = data[["Close", "Volume", "High", "Low"]].copy()
    price_data["Tomorrow"] = price_data["Close"].shift(-1)
    price_data["Price_Target"] = (price_data["Tomorrow"] > price_data["Close"]).astype(int)
    price_data["Return"] = price_data["Close"].pct_change()

    price_predictors = []

    # 1. Rolling return/volatility/momentum features across multiple horizons
    horizons = [2, 5, 10, 20, 63, 126, 252, 504, 1260]
    for h in horizons:
        price_data[f"Vol_{h}"]         = price_data["Return"].rolling(h).std()
        price_data[f"Mom_{h}"]         = price_data["Return"].rolling(h).sum()
        price_data[f"Close_Ratio_{h}"] = price_data["Close"] / price_data["Close"].rolling(h).mean()
        price_predictors += [f"Vol_{h}", f"Mom_{h}", f"Close_Ratio_{h}"]

    # 2. Lagged returns (short-term mean reversion signal)
    for lag in range(1, 6):
        price_data[f"Return_Lag_{lag}"] = price_data["Return"].shift(lag)
        price_predictors.append(f"Return_Lag_{lag}")

    # 3. RSI (14-day) — overbought/oversold
    delta = price_data["Close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    price_data["RSI_14"] = (100 - (100 / (1 + gain / loss))).fillna(100)
    price_predictors.append("RSI_14")

    # 4. MACD (12/26/9)
    ema_12 = price_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_data["Close"].ewm(span=26, adjust=False).mean()
    price_data["MACD_Line"]   = ema_12 - ema_26
    price_data["MACD_Signal"] = price_data["MACD_Line"].ewm(span=9, adjust=False).mean()
    price_data["MACD_Hist"]   = price_data["MACD_Line"] - price_data["MACD_Signal"]
    price_predictors += ["MACD_Line", "MACD_Signal", "MACD_Hist"]

    # 5. Bollinger Band position — where is price within its band?
    #    BB_Pos = 0 means at lower band, 1 means at upper band, 0.5 = midpoint
    for bb_w in [20, 50]:
        roll_mean = price_data["Close"].rolling(bb_w).mean()
        roll_std  = price_data["Close"].rolling(bb_w).std()
        upper = roll_mean + 2 * roll_std
        lower = roll_mean - 2 * roll_std
        price_data[f"BB_Pos_{bb_w}"]   = (price_data["Close"] - lower) / (upper - lower + 1e-9)
        price_data[f"BB_Width_{bb_w}"] = (upper - lower) / (roll_mean + 1e-9)  # band width = volatility regime
        price_predictors += [f"BB_Pos_{bb_w}", f"BB_Width_{bb_w}"]

    # 6. ATR (14-day Average True Range) — normalised volatility
    high_low   = price_data["High"] - price_data["Low"]
    high_close = (price_data["High"] - price_data["Close"].shift()).abs()
    low_close  = (price_data["Low"]  - price_data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    price_data["ATR_14"] = true_range.rolling(14).mean() / price_data["Close"]  # normalised
    price_predictors.append("ATR_14")

    # 7. Volume features — abnormal volume often precedes directional moves
    price_data["Vol_Ratio_10"] = price_data["Volume"] / price_data["Volume"].rolling(10).mean()
    price_data["Vol_Ratio_20"] = price_data["Volume"] / price_data["Volume"].rolling(20).mean()
    price_predictors += ["Vol_Ratio_10", "Vol_Ratio_20"]

    # 8. High-Low daily range (normalised) — intraday volatility signal
    price_data["HL_Range"] = (price_data["High"] - price_data["Low"]) / price_data["Close"]
    price_predictors.append("HL_Range")

    price_data = price_data.dropna()
    train_price = price_data.iloc[-(price_window + 1):-1]
    test_price  = price_data.iloc[-1:].copy()
    today_close = float(test_price["Close"].values[0])

    # ----------------------
    # Feature Importance Pruning
    # ----------------------
    # A quick RF fit to rank features; drop the bottom 25% by importance.
    # Noisy low-importance features degrade RF probability calibration.
    quick_rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1, n_jobs=-1)
    quick_rf.fit(train_price[price_predictors], train_price["Price_Target"])
    importances  = pd.Series(quick_rf.feature_importances_, index=price_predictors)
    cutoff       = importances.quantile(0.25)          # bottom 25% threshold
    price_predictors = importances[importances >= cutoff].index.tolist()
    print(f"[{ticker}] Features after pruning: {len(price_predictors)}")

    # ----------------------
    # Hyperparameter Tuning (RandomizedSearchCV on RF)
    # ----------------------
    # Tune max_depth and min_samples_leaf — the two params that most affect
    # probability quality. n_iter=8 with 3-fold TSV is fast enough for prod.
    tscv_search = TimeSeriesSplit(n_splits=3)
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [6, 8, 10, 12, None],
        "min_samples_leaf": [5, 10, 20, 30],
        "max_features":     ["sqrt", "log2", 0.5],
    }
    base_rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    search  = RandomizedSearchCV(
        base_rf, param_dist, n_iter=8, cv=tscv_search,
        scoring="roc_auc",   # AUC optimises probability ranking, not just accuracy
        random_state=1, n_jobs=1
    )
    search.fit(train_price[price_predictors], train_price["Price_Target"])
    best_rf = search.best_estimator_
    print(f"[{ticker}] Best RF params: {search.best_params_}")

    # ----------------------
    # Voting Ensemble: RF + GradientBoosting
    # ----------------------
    # GBM uses a fundamentally different learning algorithm (sequential residual
    # fitting vs. parallel bootstrapping) so its errors are uncorrelated with RF.
    # Averaging their probabilities reduces variance in the confidence score.
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=1
    )

    # soft voting = average predicted probabilities (better than hard majority vote)
    ensemble = VotingClassifier(
        estimators=[("rf", best_rf), ("gb", gb_clf)],
        voting="soft",
        n_jobs=1
    )

    # ----------------------
    # Calibration
    # ----------------------
    # Calibrate the ensemble so that "60% confidence" genuinely means the model
    # was right ~60% of the time historically (isotonic regression on TSV folds)
    tscv_calib = TimeSeriesSplit(n_splits=5)
    calibrated_clf = CalibratedClassifierCV(ensemble, method="isotonic", cv=tscv_calib, n_jobs=1)
    calibrated_clf.fit(train_price[price_predictors], train_price["Price_Target"])

    test_price["Price_Predicted"] = calibrated_clf.predict(test_price[price_predictors])
    prob_up   = float(calibrated_clf.predict_proba(test_price[price_predictors])[0, 1])
    prob_down = 1.0 - prob_up
    predicted_direction = test_price["Price_Predicted"].values[0]
    # Show confidence for whichever direction was actually predicted:
    # if predicted Up → confidence = prob_up; if Down → confidence = prob_down
    price_conf = prob_up if predicted_direction == 1 else prob_down

    # ----------------------
    # Regressor — exact next-day price (tuned RF)
    # ----------------------
    param_dist_reg = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [6, 8, 10, None],
        "min_samples_leaf": [5, 10, 20],
    }
    base_reg    = RandomForestRegressor(random_state=1, n_jobs=-1)
    search_reg  = RandomizedSearchCV(
        base_reg, param_dist_reg, n_iter=8, cv=tscv_search,
        scoring="neg_mean_absolute_error", random_state=1, n_jobs=1
    )
    search_reg.fit(train_price[price_predictors], train_price["Tomorrow"])
    price_reg     = search_reg.best_estimator_
    print(f"[{ticker}] Best Reg params: {search_reg.best_params_}")

    forecasted_close = float(price_reg.predict(test_price[price_predictors])[0])

    # Training fit line + MAPE for chart
    train_fitted    = price_reg.predict(train_price[price_predictors])
    train_fit_dates  = train_price.index.strftime("%Y-%m-%d").tolist()
    train_fit_prices = [round(float(p), 2) for p in train_fitted]
    actuals          = train_price["Tomorrow"].values
    mape             = float(np.mean(np.abs((actuals - train_fitted) / actuals)) * 100)

    # Align regressor direction with classifier
    if test_price["Price_Predicted"].values[0] == 1:
        forecasted_close = max(forecasted_close, today_close * 1.001)
    else:
        forecasted_close = min(forecasted_close, today_close * 0.999)
    forecasted_close = round(forecasted_close, 2)

    next_trading_day_str = get_next_trading_day()

    # ----------------------
    # Smooth Daily Forecast Path (1 Year / 252 trading days)
    # ----------------------
    daily_vol = train_price["Return"].std()

    us_holidays      = USFederalHolidayCalendar()
    us_bday          = CustomBusinessDay(calendar=us_holidays)
    today_ts         = pd.to_datetime(date.today())
    all_future_dates = pd.date_range(start=today_ts + us_bday, periods=252, freq=us_bday)

    horizon_prices = {0: today_close}
    for h_label, h_days in [("1_Week", 5), ("1_Month", 21), ("1_Year", 252)]:
        lt_data = price_data[["Close"] + price_predictors].copy()
        lt_data["Future_Log_Return"] = np.log(lt_data["Close"].shift(-h_days) / lt_data["Close"])
        lt_data = lt_data.dropna()

        if len(lt_data) > 100:
            lt_train = lt_data.iloc[-price_window:]
            lt_reg   = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=1, n_jobs=-1
            )
            lt_reg.fit(lt_train[price_predictors], lt_train["Future_Log_Return"])
            pred_log_return  = float(lt_reg.predict(test_price[price_predictors])[0])
            max_move         = 0.90 if h_days == 252 else 0.60
            pred_log_return  = np.clip(pred_log_return, -max_move, max_move)
            horizon_prices[h_days] = float(round(today_close * np.exp(pred_log_return), 2))
        else:
            horizon_prices[h_days] = today_close

    anchors = sorted(horizon_prices.items())

    def interp_price(t):
        for i in range(len(anchors) - 1):
            t0, p0 = anchors[i]
            t1, p1 = anchors[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                return float(np.exp(np.log(p0) + frac * (np.log(p1) - np.log(p0))))
        return anchors[-1][1]

    chart_future_dates  = []
    chart_future_prices = []
    chart_future_upper  = []
    chart_future_lower  = []

    for t, future_date in enumerate(all_future_dates, start=1):
        price_t = round(interp_price(t), 2)
        moe_t   = 1.96 * daily_vol * np.sqrt(t)
        chart_future_dates.append(future_date.strftime("%Y-%m-%d"))
        chart_future_prices.append(price_t)
        chart_future_upper.append(float(round(price_t * (1 + moe_t), 2)))
        chart_future_lower.append(float(round(price_t * (1 - moe_t), 2)))

    extended_forecasts = {}
    for label, idx in [("1_Week", 4), ("1_Month", 20), ("1_Year", 251)]:
        extended_forecasts[label] = {
            "Date":  chart_future_dates[idx],
            "Price": chart_future_prices[idx],
            "Upper": chart_future_upper[idx],
            "Lower": chart_future_lower[idx],
        }

    # ----------------------
    # Dividend Model
    # ----------------------
    divs = data[["Dividends"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()

    # Initialise all dividend outputs as N/A — only overwritten if enough data exists.
    # Keeping these in scope here prevents NameError in the results block below.
    forecasted_div     = "N/A"
    today_div          = 0.0
    div_conf_up        = 0.5   # P(Up) — initialised to 50/50 uncertainty
    div_pred           = 0
    next_dividend_date = pd.NaT
    div_direction_text = "N/A"
    final_div_conf     = "N/A"

    if len(divs) < 10:
        # Insufficient dividend history — non-dividend stock or too new
        pass
    else:
        today_dt         = pd.to_datetime(date.today())
        last_div_date    = divs.index[-1].tz_localize(None)
        avg_days_between = divs.index.to_series().diff().mean().days
        projected_date   = last_div_date + pd.Timedelta(days=avg_days_between)
        while projected_date < today_dt:
            projected_date += pd.Timedelta(days=avg_days_between)
        next_dividend_date = projected_date

        divs["Next_Dividend"] = divs["Dividends"].shift(-1)
        divs["Div_Target"]    = (divs["Next_Dividend"] > divs["Dividends"]).astype(int)
        divs["Div_Growth_1"]  = divs["Dividends"].pct_change(1)
        divs["Div_Growth_2"]  = divs["Dividends"].pct_change(2)
        for w in [4, 6, 8]:
            divs[f"Rolling_Avg_{w}"] = divs["Dividends"].rolling(w).mean()
            divs[f"Rolling_Std_{w}"] = divs["Dividends"].rolling(w).std()
        div_predictors = ["Dividends", "Div_Growth_1", "Div_Growth_2",
                          "Rolling_Avg_4", "Rolling_Std_4",
                          "Rolling_Avg_6", "Rolling_Std_6",
                          "Rolling_Avg_8", "Rolling_Std_8"]

        divs      = divs.dropna()
        train_div = divs.iloc[-(div_window + 1):-1]
        test_div  = divs.iloc[-1:].copy()
        today_div = float(test_div["Dividends"].values[0])

        # --- Classifier: predict direction ---
        if train_div["Div_Target"].nunique() <= 1:
            # All historical dividends moved the same direction — treat as certain
            div_pred   = int(train_div["Div_Target"].iloc[0])
            div_conf_up = float(div_pred)   # 1.0 if always Up, 0.0 if always Down/Flat
        else:
            try:
                div_clf    = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
                tscv_div   = TimeSeriesSplit(n_splits=3)
                cal_div    = CalibratedClassifierCV(div_clf, method="isotonic", cv=tscv_div, n_jobs=1)
                cal_div.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred    = int(cal_div.predict(test_div[div_predictors])[0])
                div_conf_up = float(cal_div.predict_proba(test_div[div_predictors])[0, 1])
            except ValueError:
                div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
                div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred    = int(div_clf.predict(test_div[div_predictors])[0])
                div_conf_up = float(div_clf.predict_proba(test_div[div_predictors])[0, 1]) if len(div_clf.classes_) == 2 else float(div_pred)

        # --- Regressor: predict exact next dividend amount ---
        div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_reg.fit(train_div[div_predictors], train_div["Next_Dividend"])
        forecasted_div = float(div_reg.predict(test_div[div_predictors])[0])

        # Align regressor direction with classifier
        if div_pred == 1:
            forecasted_div = max(forecasted_div, today_div * 1.001)
        else:
            forecasted_div = min(forecasted_div, today_div * 0.999)
        forecasted_div = round(forecasted_div, 2)

        # --- Direction label + confidence in the PREDICTED direction ---
        # div_conf_up = P(Up). Confidence shown is always for the predicted direction,
        # matching the same fix applied to price_conf above.
        if forecasted_div > today_div:
            div_direction_text = "Up"
            final_div_conf     = round(div_conf_up * 100, 2)
        elif forecasted_div < today_div and forecasted_div > 0:
            div_direction_text = "Down"
            final_div_conf     = round((1.0 - div_conf_up) * 100, 2)
        else:
            div_direction_text = "Flat"
            # Flat: neither Up nor Down — confidence is how far the model is from 50/50
            # expressed as the stronger of the two probabilities
            flat_conf          = max(div_conf_up, 1.0 - div_conf_up)
            final_div_conf     = round(flat_conf * 100, 2)

    forecasted_yield = round((forecasted_div / today_close) * 100, 2) if (today_close > 0 and forecasted_div != "N/A") else "N/A"

    # ----------------------
    # Combine results
    # ----------------------

    return pd.DataFrame({
        "Next_Trading_Day":     [next_trading_day_str],
        "Ticker":               [ticker],
        "Price_Predicted":      ["Up" if test_price["Price_Predicted"].values[0] == 1 else "Down"],
        "Price_Confidence (%)": [round(price_conf * 100, 2)],
        "Forecasted_Close":     [forecasted_close],
        "Next_Dividend_Date":   [next_dividend_date.strftime("%Y-%m-%d") if pd.notna(next_dividend_date) else "N/A"],
        "Div_Predicted":        [div_direction_text],
        "Div_Confidence (%)":   [final_div_conf],
        "Forecasted_Dividend":  [forecasted_div],   # already "N/A" string if no dividend data
        "Forecasted_Yield (%)": [forecasted_yield], # already "N/A" string if no dividend data
        "Extended_Forecasts":   [extended_forecasts],
        "Chart_Future_Dates":   [chart_future_dates],
        "Chart_Future_Prices":  [chart_future_prices],
        "Chart_Future_Upper":   [chart_future_upper],
        "Chart_Future_Lower":   [chart_future_lower],
        "Train_Fit_Dates":      [train_fit_dates],
        "Train_Fit_Prices":     [train_fit_prices],
        "Model_MAPE":           [round(mape, 2)],
    })