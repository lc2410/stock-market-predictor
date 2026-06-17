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
    """Returns a calendar object used to skip weekends and US market holidays."""
    return CustomBusinessDay(calendar=USFederalHolidayCalendar())

def get_chart_data(ticker_symbol, predicted_price=None):
    """Retrieves the recent historical price and dividend data needed to draw the frontend charts."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5y")

        if not hist.empty:
            hist = hist.dropna(subset=['Close']) 
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep='last')]

        dates = hist.index.strftime('%Y-%m-%d').tolist() if not hist.empty else []
        prices = hist['Close'].tolist() if not hist.empty else []

        if predicted_price is not None and not hist.empty:
            anchor_date = hist.index[-1]
            dates.append((anchor_date + _get_us_bday()).strftime('%Y-%m-%d'))
            prices.append(float(predicted_price))

        divs = ticker.dividends.tail(20)
        if not divs.empty:
            divs = divs.dropna() 
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
    """Downloads all available historical stock data from Yahoo Finance and checks if there is enough data to train the models."""
    stock_ticker = yf.Ticker(ticker)
    data = stock_ticker.history(period="max")

    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'.")
        return None

    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep='last')]

    data = data.loc["2010-01-01":].copy()
    # Ensure there's at least 1300 data points for training the model
    if len(data) < 1300:
        print(f"Error: Not enough historical data for {ticker}.")
        return None

    return data

def _engineer_price_features(data):
    """Calculates technical indicators and engineered features to help the model understand recent price trends."""
    price_data = data[["Close", "Volume"]].copy()
    
    # 'Tomorrow' target (Required by the Regressor)
    price_data["Tomorrow"] = price_data["Close"].shift(-1)
    
    # Smooth the target for better short-term trend prediction
    price_data["Smoothed_Future"] = price_data["Close"].shift(-1).rolling(window=3).mean()
    price_data["Price_Target"] = (price_data["Smoothed_Future"] > price_data["Close"]).astype(int)

    # Price Action (The raw signal)
    price_data["Log_Return"] = np.log(price_data["Close"] / price_data["Close"].shift(1))
    price_data["Return_Lag_1"] = price_data["Log_Return"].shift(1)

    # Trend & Momentum (Is it going up, and how fast?)
    delta = price_data["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    price_data["RSI_14"] = (100 - (100 / (1 + gain / loss))).fillna(100)

    ema_12 = price_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_data["Close"].ewm(span=26, adjust=False).mean()
    price_data["MACD_Hist"] = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()

    # Volatility & Mean Reversion (Is it overextended?)
    roll_mean = price_data["Close"].rolling(20).mean()
    roll_std = price_data["Close"].rolling(20).std()
    price_data["BB_Width"] = (4 * roll_std) / (roll_mean + 1e-9)
    price_data["BB_Pos"] = (price_data["Close"] - (roll_mean - 2 * roll_std)) / (4 * roll_std + 1e-9)

    # Volume Context (Is there conviction behind the move?)
    price_data["Vol_Ratio_10"] = price_data["Volume"] / price_data["Volume"].rolling(10).mean()

    predictors = [
        "Log_Return", "Return_Lag_1", "RSI_14", "MACD_Hist", 
        "BB_Width", "BB_Pos", "Vol_Ratio_10"
    ]

    price_data = price_data.dropna(subset=predictors)

    return price_data, predictors

def _train_price_classifier(train, test, predictors, ticker):
    """Trains a machine learning model to predict whether the stock price will go UP or DOWN tomorrow, and calculates the confidence percentage."""
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
    """Trains a machine learning model to predict the exact dollar amount of tomorrow's closing price."""
    tscv = TimeSeriesSplit(n_splits=3)
    train_target_return = np.log(train["Tomorrow"] / train["Close"])
    
    search_reg = RandomizedSearchCV(
        RandomForestRegressor(random_state=1, n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [6, 8, 10, None], "min_samples_leaf": [5, 10, 20]},
        n_iter=8, cv=tscv, scoring="neg_mean_absolute_error", random_state=1, n_jobs=1,
    )
    search_reg.fit(train[predictors], train_target_return)
    price_reg = search_reg.best_estimator_
    print(f"[{ticker}] Best regressor params: {search_reg.best_params_}")

    pred_log_return = float(price_reg.predict(test[predictors])[0])
    forecasted_close = float(today_close * np.exp(pred_log_return))

    train_fitted_returns = price_reg.predict(train[predictors])
    train_fitted = train["Close"].values * np.exp(train_fitted_returns)
    
    # Slice historical fits to match the exact length of the requested target_dates
    train_fit_prices = [round(float(p), 2) for p in train_fitted[-len(target_dates):]]

    if direction == 1:
        forecasted_close = max(forecasted_close, today_close * 1.001)
    else:
        forecasted_close = min(forecasted_close, today_close * 0.999)

    return round(forecasted_close, 2), train_fit_prices

def _forecast_price_long_term(price_data, test_row, predictors, today_close, forecast_close, price_window, anchor_date):
    """Projects the estimated stock price out to 1 week, 1 month, and 1 year, and calculates the margin of error bounds for the chart."""
    
    # Calculate volatility strictly on the recent training window
    daily_vol = price_data["Log_Return"].iloc[-price_window:].std()
    
    us_bday = _get_us_bday()
    all_future_dates = pd.date_range(
        start=anchor_date + us_bday, periods=252, freq=us_bday
    )

    horizon_prices = {0: today_close, 1: forecast_close}
    for h_label, h_days in [("1_Week", 5), ("1_Month", 21), ("1_Year", 252)]:
        lt_data = price_data[["Close"] + predictors].copy()
        
        # This correctly targets the stationary log return of the future horizons
        lt_data["Future_Log_Return"] = np.log(lt_data["Close"].shift(-h_days) / lt_data["Close"])
        lt_data = lt_data.dropna()

        if len(lt_data) > 100:
            lt_reg = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=1, n_jobs=-1
            )
            lt_reg.fit(lt_data.iloc[-price_window:][predictors], lt_data.iloc[-price_window:]["Future_Log_Return"])
            
            clip_bound = 0.90 if h_days == 252 else 0.60
            pred_log_return = np.clip(float(lt_reg.predict(test_row[predictors])[0]), -clip_bound, clip_bound)
            
            # Reconstruct the absolute price anchor
            horizon_prices[h_days] = float(round(today_close * np.exp(pred_log_return), 2))
        else:
            horizon_prices[h_days] = today_close

    anchors = sorted(horizon_prices.items())

    def interp_price(t):
        """Smoothly connects the dots between our short-term and long-term price targets."""
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
        
        # Apply exponential math for the upper/lower bounds
        upper_bound = float(round(price_t * np.exp(moe_t), 2))
        lower_bound = float(round(price_t * np.exp(-moe_t), 2))
        
        chart_future_dates.append(future_date.strftime('%Y-%m-%d'))
        chart_future_prices.append(price_t)
        chart_future_upper.append(upper_bound)
        chart_future_lower.append(lower_bound)

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
    """Estimates the date of the next dividend payout and prepares historical dividend trends for the model to study."""
    # Add 252-day trailing price return as a base feature for dividend prediction
    data["Price_Return_252"] = data["Close"].pct_change(252)
    
    divs = data[["Dividends", "Price_Return_252"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()

    if len(divs) < 10:
        return None, None, pd.NaT, 0.0

    last_div_date = divs.index[-1]
    avg_days_between = divs.index.to_series().diff().mean().days
    
    if pd.isna(avg_days_between) or avg_days_between <= 0:
        avg_days_between = 90

    projected_date = last_div_date + pd.Timedelta(days=avg_days_between)
    while projected_date <= anchor_date:
        projected_date += pd.Timedelta(days=avg_days_between)
    next_dividend_date = projected_date

    divs["Next_Dividend"] = divs["Dividends"].shift(-1)
    divs["Div_Target"] = (divs["Next_Dividend"] > divs["Dividends"]).astype(int)
    
    # Looks at Immediate Growth
    divs["Div_Growth_1"] = divs["Dividends"].pct_change(1)
    # Looks at Short-Term Historical Trends (1 Year of Quarterly Payouts)
    divs["Rolling_Avg_4"] = divs["Dividends"].rolling(4).mean()

    div_predictors = [
        "Dividends", "Div_Growth_1", "Rolling_Avg_4", "Price_Return_252"
    ]
    
    today_div = float(divs["Dividends"].iloc[-1])
    divs = divs.dropna(subset=div_predictors)

    return divs, div_predictors, next_dividend_date, today_div

def _train_div_classifier(train_div, test_div, div_predictors):
    """Trains a machine learning model to predict if the next dividend payout will be HIGHER or LOWER than the previous one."""
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
        prob_up = float(cal_div.predict_proba(test_div[div_predictors])[0, 1])
    except ValueError:
        div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
        div_pred = int(div_clf.predict(test_div[div_predictors])[0])
        prob_up = float(div_clf.predict_proba(test_div[div_predictors])[0, 1]) if len(div_clf.classes_) == 2 else 0.5

    confidence = prob_up if div_pred == 1 else (1.0 - prob_up)
    return div_pred, confidence

def _train_div_regressor(train_div, test_div, div_predictors, div_pred, today_div):
    """Trains a machine learning model to predict the exact dollar amount of the upcoming dividend payout."""
    # Target the stationary Log Return (Growth) of the dividend
    train_target_growth = np.log(train_div["Next_Dividend"] / train_div["Dividends"])
    
    div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
    div_reg.fit(train_div[div_predictors], train_target_growth)
    
    # Predict the growth rate and reconstruct the absolute payout
    pred_growth = float(div_reg.predict(test_div[div_predictors])[0])
    forecasted_div = float(today_div * np.exp(pred_growth))

    # Nudge to stay consistent with the classifier
    forecasted_div = (
        max(forecasted_div, today_div * 1.001) if div_pred == 1
        else min(forecasted_div, today_div * 0.999)
    )

    # Reconstruct the historical training fit using the trained model
    train_fitted_growth = div_reg.predict(train_div[div_predictors])
    train_fitted_amounts = [round(float(x), 2) for x in (train_div["Dividends"].values * np.exp(train_fitted_growth))]

    return round(forecasted_div, 2), train_fitted_amounts

def _forecast_div_long_term(divs, div_predictors, test_div, today_div, forecasted_div, next_dividend_date, avg_days_between, div_window):
    """Projects the estimated dollar amounts for the 2nd, 3rd, and 4th upcoming dividend payouts."""
    payout_vol = divs["Dividends"].pct_change().std()

    div_future_dates  = [next_dividend_date.strftime('%Y-%m-%d')]
    div_future_amounts = [round(forecasted_div, 2)] 
    moe_1 = 1.96 * payout_vol * np.sqrt(1)
    div_future_upper  = [round(forecasted_div * (1 + moe_1), 2)] 
    div_future_lower  = [round(max(forecasted_div * (1 - moe_1), 0.0), 2)] 

    for cycle in range(2, 5):
        target_col = f"Div_Growth_{cycle}"
        lt_divs = divs[div_predictors].copy()
        
        # Predict the log growth from the current dividend to the future cycle
        future_div = lt_divs["Dividends"].shift(-cycle)
        lt_divs[target_col] = np.log(future_div / lt_divs["Dividends"])
        lt_divs = lt_divs.dropna()

        if len(lt_divs) > 8:
            lt_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
            lt_reg.fit(lt_divs.iloc[-div_window:][div_predictors], lt_divs.iloc[-div_window:][target_col])
            
            # Predict the growth and reconstruct the absolute amount
            pred_growth = float(lt_reg.predict(test_div[div_predictors])[0])
            predicted_amount = round(float(today_div * np.exp(pred_growth)), 2) 
        else:
            predicted_amount = round(today_div, 2) 

        moe = 1.96 * payout_vol * np.sqrt(cycle)
        projected_date = next_dividend_date + pd.Timedelta(days=avg_days_between * (cycle - 1))

        div_future_dates.append(projected_date.strftime('%Y-%m-%d'))
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

def run_real_time_model(ticker, price_window=1260, div_window=20):
    """The main orchestrator that runs all the individual machine learning steps and packages the final results to send to the frontend."""
    data = _fetch_data(ticker)
    if data is None:
        return None

    anchor_date = data.index[-1]
    
    # price prediction pipeline
    price_data, predictors = _engineer_price_features(data)
    train_price = price_data.iloc[-(price_window + 1):-1]
    test_price = price_data.iloc[-1:].copy()
    today_close = float(test_price["Close"].values[0])
    
    price_direction, price_conf = _train_price_classifier(train_price, test_price, predictors, ticker)

    target_dates = price_data.index[-price_window:]
    forecasted_close, train_fit_prices = _train_price_regressor(
        train_price, test_price, predictors, price_direction, today_close, ticker, target_dates
    )
    
    # Align training fit prices with their respective T+1 forecast dates
    train_fit_dates = target_dates.strftime('%Y-%m-%d').tolist()

    chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower, extended_forecasts = (
        _forecast_price_long_term(price_data, test_price, predictors, today_close, forecasted_close, price_window, anchor_date)
    )

    # dividend prediction pipeline
    divs, div_predictors, next_dividend_date, today_div = _engineer_div_features(data, anchor_date)
    
    div_direction = None
    div_conf = 0.0
    forecasted_div = "N/A"
    div_future_dates, div_future_amounts = [], []
    div_future_upper, div_future_lower = [], []
    div_extended_forecasts = {}
    train_fit_div_dates, train_fit_div_amounts = [], []

    if divs is not None:
        avg_days_between = divs.index.to_series().diff().mean().days
        if pd.isna(avg_days_between) or avg_days_between <= 0:
            avg_days_between = 90 
            
        train_div = divs.iloc[-(div_window + 1):-1]
        test_div = divs.iloc[-1:].copy()

        div_direction, div_conf = _train_div_classifier(train_div, test_div, div_predictors)
        
        # Capture the training fit amounts and extract the target dates
        forecasted_div, train_fit_div_amounts = _train_div_regressor(train_div, test_div, div_predictors, div_direction, today_div)
        train_fit_div_dates = divs.index[-len(train_fit_div_amounts):].strftime('%Y-%m-%d').tolist()

        div_future_dates, div_future_amounts, div_future_upper, div_future_lower, div_extended_forecasts = (
            _forecast_div_long_term(divs, div_predictors, test_div, today_div, forecasted_div, next_dividend_date, avg_days_between, div_window)
        )

   
    # Format Dates
    next_trading_day_str = (anchor_date + _get_us_bday()).strftime('%Y-%m-%d')
    next_div_date_str = next_dividend_date.strftime('%Y-%m-%d') if pd.notna(next_dividend_date) else "N/A"

    # Format UI Strings
    price_pred_str = "Up" if price_direction == 1 else "Down"
    div_pred_str = "N/A" if div_direction is None else ("Up" if div_direction == 1 else "Down")

    # Format Percentages
    price_conf_pct = round(price_conf * 100, 2)
    div_conf_pct = "N/A" if div_direction is None else round(div_conf * 100, 2)
    
    forecasted_yield = (
        round((forecasted_div / today_close) * 100, 2)
        if today_close > 0 and forecasted_div != "N/A"
        else "N/A"
    )

    # Fetch Company Metadata
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    # Return fully prepared Data
    return pd.DataFrame({
        "Next_Trading_Day":        [next_trading_day_str],
        "Ticker":                  [ticker],
        "Company_Name":            [company_name],
        "Price_Predicted":         [price_pred_str],
        "Price_Confidence (%)":    [price_conf_pct],
        "Forecasted_Close":        [forecasted_close],
        "Next_Dividend_Date":      [next_div_date_str],
        "Div_Predicted":           [div_pred_str],
        "Div_Confidence (%)":      [div_conf_pct],
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
        "Train_Fit_Div_Dates":     [train_fit_div_dates],
        "Train_Fit_Div_Amounts":   [train_fit_div_amounts],
    })