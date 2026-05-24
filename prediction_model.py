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


def get_next_trading_day():
    us_holidays = USFederalHolidayCalendar()
    us_bday = CustomBusinessDay(calendar=us_holidays)
    return (pd.to_datetime(date.today()) + us_bday).strftime('%Y-%m-%d')


def get_future_trading_date(days_ahead):
    us_holidays = USFederalHolidayCalendar()
    us_bday = CustomBusinessDay(calendar=us_holidays)
    return (pd.to_datetime(date.today()) + us_bday * days_ahead).strftime('%Y-%m-%d')


def get_chart_data(ticker_symbol, predicted_price=None):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist   = ticker.history(period="6mo")
        dates  = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        if predicted_price is not None:
            dates.append(get_next_trading_day())
            prices.append(float(predicted_price))
        divs = ticker.dividends.tail(12)
        return {
            "dates": dates,
            "prices": prices,
            "dividend_dates": divs.index.strftime('%Y-%m-%d').tolist(),
            "dividend_amounts": divs.values.tolist(),
        }
    except Exception as e:
        print(f"Error fetching chart data: {e}")
        return None


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

    # -----------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------
    price_data = data[["Close", "Volume", "High", "Low"]].copy()
    price_data["Tomorrow"]     = price_data["Close"].shift(-1)
    price_data["Price_Target"] = (price_data["Tomorrow"] > price_data["Close"]).astype(int)
    price_data["Return"]       = price_data["Close"].pct_change()

    price_predictors = []

    # Rolling volatility, momentum, and price-vs-mean across multiple horizons
    for h in [2, 5, 10, 20, 63, 126, 252, 504, 1260]:
        price_data[f"Vol_{h}"]         = price_data["Return"].rolling(h).std()
        price_data[f"Mom_{h}"]         = price_data["Return"].rolling(h).sum()
        price_data[f"Close_Ratio_{h}"] = price_data["Close"] / price_data["Close"].rolling(h).mean()
        price_predictors += [f"Vol_{h}", f"Mom_{h}", f"Close_Ratio_{h}"]

    # Short-term mean-reversion signal
    for lag in range(1, 6):
        price_data[f"Return_Lag_{lag}"] = price_data["Return"].shift(lag)
        price_predictors.append(f"Return_Lag_{lag}")

    # RSI (14-day)
    delta = price_data["Close"].diff()
    gain  = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    price_data["RSI_14"] = (100 - (100 / (1 + gain / loss))).fillna(100)
    price_predictors.append("RSI_14")

    # MACD (12/26/9)
    ema_12 = price_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_data["Close"].ewm(span=26, adjust=False).mean()
    price_data["MACD_Line"]   = ema_12 - ema_26
    price_data["MACD_Signal"] = price_data["MACD_Line"].ewm(span=9, adjust=False).mean()
    price_data["MACD_Hist"]   = price_data["MACD_Line"] - price_data["MACD_Signal"]
    price_predictors += ["MACD_Line", "MACD_Signal", "MACD_Hist"]

    # Bollinger Band position and width (20 & 50-day)
    for bb_w in [20, 50]:
        roll_mean = price_data["Close"].rolling(bb_w).mean()
        roll_std  = price_data["Close"].rolling(bb_w).std()
        upper = roll_mean + 2 * roll_std
        lower = roll_mean - 2 * roll_std
        price_data[f"BB_Pos_{bb_w}"]   = (price_data["Close"] - lower) / (upper - lower + 1e-9)
        price_data[f"BB_Width_{bb_w}"] = (upper - lower) / (roll_mean + 1e-9)
        price_predictors += [f"BB_Pos_{bb_w}", f"BB_Width_{bb_w}"]

    # Normalised ATR (14-day) — true range accounts for gap opens
    hl  = price_data["High"] - price_data["Low"]
    hc  = (price_data["High"] - price_data["Close"].shift()).abs()
    lc  = (price_data["Low"]  - price_data["Close"].shift()).abs()
    price_data["ATR_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean() / price_data["Close"]
    price_predictors.append("ATR_14")

    # Volume ratio vs. rolling mean — abnormal volume can precede directional moves
    price_data["Vol_Ratio_10"] = price_data["Volume"] / price_data["Volume"].rolling(10).mean()
    price_data["Vol_Ratio_20"] = price_data["Volume"] / price_data["Volume"].rolling(20).mean()
    price_predictors += ["Vol_Ratio_10", "Vol_Ratio_20"]

    # Intraday high-low range normalised by close
    price_data["HL_Range"] = (price_data["High"] - price_data["Low"]) / price_data["Close"]
    price_predictors.append("HL_Range")

    price_data  = price_data.dropna()
    train_price = price_data.iloc[-(price_window + 1):-1]
    test_price  = price_data.iloc[-1:].copy()
    today_close = float(test_price["Close"].values[0])

    # -----------------------------------------------------------------
    # Feature importance pruning
    # Drop the bottom 25% of features by importance before calibration.
    # Noisy low-importance features degrade RF probability estimates.
    # -----------------------------------------------------------------
    quick_rf     = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1, n_jobs=-1)
    quick_rf.fit(train_price[price_predictors], train_price["Price_Target"])
    importances  = pd.Series(quick_rf.feature_importances_, index=price_predictors)
    price_predictors = importances[importances >= importances.quantile(0.25)].index.tolist()
    print(f"[{ticker}] Features after pruning: {len(price_predictors)}")

    # -----------------------------------------------------------------
    # Hyperparameter tuning
    # roc_auc scoring optimises probability ranking, not just accuracy.
    # -----------------------------------------------------------------
    tscv_search = TimeSeriesSplit(n_splits=3)
    param_dist  = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [6, 8, 10, 12, None],
        "min_samples_leaf": [5, 10, 20, 30],
        "max_features":     ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=1, n_jobs=-1),
        param_dist, n_iter=8, cv=tscv_search, scoring="roc_auc", random_state=1, n_jobs=1,
    )
    search.fit(train_price[price_predictors], train_price["Price_Target"])
    best_rf = search.best_estimator_
    print(f"[{ticker}] Best RF params: {search.best_params_}")

    # -----------------------------------------------------------------
    # Voting ensemble: RF + GradientBoosting
    # Their errors are uncorrelated (different learning algorithms),
    # so averaging probabilities reduces variance in the confidence score.
    # -----------------------------------------------------------------
    ensemble = VotingClassifier(
        estimators=[
            ("rf", best_rf),
            ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=1)),
        ],
        voting="soft",
        n_jobs=1,
    )
    tscv_calib    = TimeSeriesSplit(n_splits=5)
    calibrated_clf = CalibratedClassifierCV(ensemble, method="isotonic", cv=tscv_calib, n_jobs=1)
    calibrated_clf.fit(train_price[price_predictors], train_price["Price_Target"])

    test_price["Price_Predicted"] = calibrated_clf.predict(test_price[price_predictors])
    prob_up   = float(calibrated_clf.predict_proba(test_price[price_predictors])[0, 1])
    prob_down = 1.0 - prob_up
    # Report confidence in whichever direction was predicted, not always P(Up)
    price_conf = prob_up if test_price["Price_Predicted"].values[0] == 1 else prob_down

    # -----------------------------------------------------------------
    # Next-day price regressor (tuned RF)
    # -----------------------------------------------------------------
    search_reg = RandomizedSearchCV(
        RandomForestRegressor(random_state=1, n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None], "min_samples_leaf": [5, 10, 20]},
        n_iter=8, cv=tscv_search, scoring="neg_mean_absolute_error", random_state=1, n_jobs=1,
    )
    search_reg.fit(train_price[price_predictors], train_price["Tomorrow"])
    price_reg = search_reg.best_estimator_
    print(f"[{ticker}] Best Reg params: {search_reg.best_params_}")

    forecasted_close = float(price_reg.predict(test_price[price_predictors])[0])

    # Training fit line and MAPE returned to the frontend for chart overlay
    train_fitted     = price_reg.predict(train_price[price_predictors])
    train_fit_dates  = train_price.index.strftime("%Y-%m-%d").tolist()
    train_fit_prices = [round(float(p), 2) for p in train_fitted]
    mape             = float(np.mean(np.abs((train_price["Tomorrow"].values - train_fitted) / train_price["Tomorrow"].values)) * 100)

    # Align regressor direction with classifier
    if test_price["Price_Predicted"].values[0] == 1:
        forecasted_close = max(forecasted_close, today_close * 1.001)
    else:
        forecasted_close = min(forecasted_close, today_close * 0.999)
    forecasted_close = round(forecasted_close, 2)

    next_trading_day_str = get_next_trading_day()

    # -----------------------------------------------------------------
    # Long-term forecast path (252 trading days)
    # Three horizon-specific RF regressors predict cumulative log-returns
    # at 1W, 1M, and 1Y. The daily path is log-linearly interpolated
    # between those anchors — no historical drift assumption baked in.
    # -----------------------------------------------------------------
    daily_vol        = train_price["Return"].std()
    us_bday          = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    all_future_dates = pd.date_range(start=pd.to_datetime(date.today()) + us_bday, periods=252, freq=us_bday)

    horizon_prices = {0: today_close}
    for h_label, h_days in [("1_Week", 5), ("1_Month", 21), ("1_Year", 252)]:
        lt_data = price_data[["Close"] + price_predictors].copy()
        lt_data["Future_Log_Return"] = np.log(lt_data["Close"].shift(-h_days) / lt_data["Close"])
        lt_data = lt_data.dropna()
        if len(lt_data) > 100:
            lt_reg = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=1, n_jobs=-1)
            lt_reg.fit(lt_data.iloc[-price_window:][price_predictors], lt_data.iloc[-price_window:]["Future_Log_Return"])
            pred_log_return  = np.clip(float(lt_reg.predict(test_price[price_predictors])[0]), -0.90 if h_days == 252 else -0.60, 0.90 if h_days == 252 else 0.60)
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

    chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower = [], [], [], []
    for t, future_date in enumerate(all_future_dates, start=1):
        price_t = round(interp_price(t), 2)
        moe_t   = 1.96 * daily_vol * np.sqrt(t)
        chart_future_dates.append(future_date.strftime("%Y-%m-%d"))
        chart_future_prices.append(price_t)
        chart_future_upper.append(float(round(price_t * (1 + moe_t), 2)))
        chart_future_lower.append(float(round(price_t * (1 - moe_t), 2)))

    extended_forecasts = {
        label: {"Date": chart_future_dates[idx], "Price": chart_future_prices[idx],
                "Upper": chart_future_upper[idx], "Lower": chart_future_lower[idx]}
        for label, idx in [("1_Week", 4), ("1_Month", 20), ("1_Year", 251)]
    }

    # -----------------------------------------------------------------
    # Dividend model
    # All outputs are initialised to N/A so the results block is always
    # in scope, even for stocks with insufficient dividend history.
    # -----------------------------------------------------------------
    divs = data[["Dividends"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()

    forecasted_div     = "N/A"
    today_div          = 0.0
    div_conf_up        = 0.5
    div_pred           = 0
    next_dividend_date = pd.NaT
    div_direction_text = "N/A"
    final_div_conf     = "N/A"

    if len(divs) >= 10:
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
                          "Rolling_Avg_4", "Rolling_Std_4", "Rolling_Avg_6",
                          "Rolling_Std_6", "Rolling_Avg_8", "Rolling_Std_8"]
        divs      = divs.dropna()
        train_div = divs.iloc[-(div_window + 1):-1]
        test_div  = divs.iloc[-1:].copy()
        today_div = float(test_div["Dividends"].values[0])

        if train_div["Div_Target"].nunique() <= 1:
            div_pred    = int(train_div["Div_Target"].iloc[0])
            div_conf_up = float(div_pred)
        else:
            try:
                cal_div = CalibratedClassifierCV(
                    RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1),
                    method="isotonic", cv=TimeSeriesSplit(n_splits=3), n_jobs=1,
                )
                cal_div.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred    = int(cal_div.predict(test_div[div_predictors])[0])
                div_conf_up = float(cal_div.predict_proba(test_div[div_predictors])[0, 1])
            except ValueError:
                div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
                div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred    = int(div_clf.predict(test_div[div_predictors])[0])
                div_conf_up = float(div_clf.predict_proba(test_div[div_predictors])[0, 1]) if len(div_clf.classes_) == 2 else float(div_pred)

        div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_reg.fit(train_div[div_predictors], train_div["Next_Dividend"])
        forecasted_div = float(div_reg.predict(test_div[div_predictors])[0])
        forecasted_div = max(forecasted_div, today_div * 1.001) if div_pred == 1 else min(forecasted_div, today_div * 0.999)
        forecasted_div = round(forecasted_div, 2)

        # Confidence is always reported for the predicted direction, matching price_conf logic
        if forecasted_div > today_div:
            div_direction_text = "Up"
            final_div_conf     = round(div_conf_up * 100, 2)
        elif forecasted_div < today_div and forecasted_div > 0:
            div_direction_text = "Down"
            final_div_conf     = round((1.0 - div_conf_up) * 100, 2)
        else:
            div_direction_text = "Flat"
            final_div_conf     = round(max(div_conf_up, 1.0 - div_conf_up) * 100, 2)

    forecasted_yield = round((forecasted_div / today_close) * 100, 2) if (today_close > 0 and forecasted_div != "N/A") else "N/A"

    return pd.DataFrame({
        "Next_Trading_Day":     [next_trading_day_str],
        "Ticker":               [ticker],
        "Price_Predicted":      ["Up" if test_price["Price_Predicted"].values[0] == 1 else "Down"],
        "Price_Confidence (%)": [round(price_conf * 100, 2)],
        "Forecasted_Close":     [forecasted_close],
        "Next_Dividend_Date":   [next_dividend_date.strftime("%Y-%m-%d") if pd.notna(next_dividend_date) else "N/A"],
        "Div_Predicted":        [div_direction_text],
        "Div_Confidence (%)":   [final_div_conf],
        "Forecasted_Dividend":  [forecasted_div],
        "Forecasted_Yield (%)": [forecasted_yield],
        "Extended_Forecasts":   [extended_forecasts],
        "Chart_Future_Dates":   [chart_future_dates],
        "Chart_Future_Prices":  [chart_future_prices],
        "Chart_Future_Upper":   [chart_future_upper],
        "Chart_Future_Lower":   [chart_future_lower],
        "Train_Fit_Dates":      [train_fit_dates],
        "Train_Fit_Prices":     [train_fit_prices],
        "Model_MAPE":           [round(mape, 2)],
    })