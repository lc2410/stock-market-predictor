import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import date
import logging

# Suppress informational messages from yfinance on import
logging.getLogger('yfinance').setLevel(logging.ERROR)

# --- HELPER FUNCTIONS ---

def get_next_trading_day():
    """
    Calculates the next trading day using the US Federal Holiday calendar.
    Skips weekends and official holidays.
    """
    us_holidays = USFederalHolidayCalendar()
    us_business_day = CustomBusinessDay(calendar=us_holidays)
    
    today = pd.to_datetime(date.today())
    next_day = today + us_business_day
    
    return next_day.strftime('%Y-%m-%d')

def get_chart_data(ticker_symbol, predicted_price=None):
    """
    Fetches 6mo history and appends the predicted price for the next day.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 1. Price History (6 months)
        hist = ticker.history(period="6mo")
        
        # Convert Timestamp index to string dates
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()

        # 2. Append Future Prediction (if provided)
        if predicted_price is not None:
            # Uses the robust holiday logic now
            next_date = get_next_trading_day() 
            dates.append(next_date)
            # Ensure predicted_price is a float
            prices.append(float(predicted_price))

        # 3. Dividend History (Last 12 payouts)
        divs = ticker.dividends.tail(12)
        div_dates = divs.index.strftime('%Y-%m-%d').tolist()
        div_amounts = divs.values.tolist()
        
        return {
            "dates": dates,
            "prices": prices,
            "dividend_dates": div_dates,
            "dividend_amounts": div_amounts
        }
    except Exception as e:
        print(f"Error fetching chart data: {e}")
        return None

# --- MAIN MODEL LOGIC ---

def run_real_time_model(ticker, price_window=1000, div_window=20):
    # Fetch historical data
    stock_ticker = yf.Ticker(ticker)
    data = stock_ticker.history(period="max")

    # Validate that data was downloaded
    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'.")
        return None

    # Ensure there's enough data for the longest horizon (1260 days)
    data = data.loc["2010-01-01":].copy()
    if len(data) < 1261: 
        print(f"Error: Not enough historical data for {ticker} to perform analysis (needs data since 2010).")
        return None

    # ----------------------
    # Price Model
    # ----------------------
    price_data = data[["Close"]].copy()
    price_data["Tomorrow"] = price_data["Close"].shift(-1)
    price_data["Price_Target"] = (price_data["Tomorrow"] > price_data["Close"]).astype(int)

    # Price Features
    price_data["Return"] = price_data["Close"].pct_change()
    horizons = [2, 5, 10, 20, 63, 126, 252, 504, 1260] 
    price_predictors = []
    
    for h in horizons:
        price_data[f"Vol_{h}"] = price_data["Return"].rolling(h).std()
        price_data[f"Mom_{h}"] = price_data["Return"].rolling(h).sum()
        price_data[f"Close_Ratio_{h}"] = price_data["Close"] / price_data["Close"].rolling(h).mean()
        price_predictors += [f"Vol_{h}", f"Mom_{h}", f"Close_Ratio_{h}"]

    for lag in range(1, 6):
        price_data[f"Return_Lag_{lag}"] = price_data["Return"].shift(lag)
        price_predictors.append(f"Return_Lag_{lag}")

    price_data = price_data.dropna()
    train_price = price_data.iloc[-(price_window+1):-1]
    test_price = price_data.iloc[-1:].copy()

    # Predicts price direction (Up/Down) and calibrates confidence
    price_clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, random_state=1, n_jobs=1)
    calibrated_price_clf = CalibratedClassifierCV(price_clf, method='isotonic', cv=5, n_jobs=1)
    calibrated_price_clf.fit(train_price[price_predictors], train_price["Price_Target"])
    test_price["Price_Predicted"] = calibrated_price_clf.predict(test_price[price_predictors])
    price_conf = calibrated_price_clf.predict_proba(test_price[price_predictors])[0,1]

    # Forecasts the exact closing price
    price_reg = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=10, random_state=1, n_jobs=1)
    price_reg.fit(train_price[price_predictors], train_price["Tomorrow"])
    forecasted_close = price_reg.predict(test_price[price_predictors])[0]

    # Align regressor with classifier
    today_close = test_price["Close"].values[0]
    if test_price["Price_Predicted"].values[0] == 1:
        forecasted_close = max(forecasted_close, today_close * 1.001)
    else:
        forecasted_close = min(forecasted_close, today_close * 0.999)
    forecasted_close = round(forecasted_close, 2)

    # Use the shared helper function for consistency
    next_trading_day_str = get_next_trading_day()

    # ----------------------
    # Dividend Model
    # ----------------------
    divs = data[["Dividends"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()
    
    # Handle cases with insufficient dividend data
    if len(divs) < 10:
        forecasted_div, div_conf, div_pred, next_dividend_date = 0, 0, 0, pd.NaT
    else:
        # Project the next dividend date
        today = pd.to_datetime(date.today())
        last_div_date = divs.index[-1].tz_localize(None)
        dividend_dates = divs.index.to_series()
        avg_days_between_divs = dividend_dates.diff().mean().days
        
        projected_date = last_div_date + pd.Timedelta(days=avg_days_between_divs)
        
        while projected_date < today:
            projected_date += pd.Timedelta(days=avg_days_between_divs)
        next_dividend_date = projected_date

        # Dividend ML model
        divs["Next_Dividend"] = divs["Dividends"].shift(-1)
        divs["Div_Target"] = (divs["Next_Dividend"] > divs["Dividends"]).astype(int)

        divs["Div_Growth_1"] = divs["Dividends"].pct_change(1)
        divs["Div_Growth_2"] = divs["Dividends"].pct_change(2)
        divs["Rolling_Avg_4"] = divs["Dividends"].rolling(4).mean()
        divs["Rolling_Std_4"] = divs["Dividends"].rolling(4).std()
        divs["Rolling_Avg_6"] = divs["Dividends"].rolling(6).mean()
        divs["Rolling_Std_6"] = divs["Dividends"].rolling(6).std()
        divs["Rolling_Avg_8"] = divs["Dividends"].rolling(8).mean()
        divs["Rolling_Std_8"] = divs["Dividends"].rolling(8).std()
        div_predictors = ["Dividends", "Div_Growth_1", "Div_Growth_2",
                          "Rolling_Avg_4", "Rolling_Std_4",
                          "Rolling_Avg_6", "Rolling_Std_6",
                          "Rolling_Avg_8", "Rolling_Std_8"]

        divs = divs.dropna()
        train_div = divs.iloc[-(div_window+1):-1]
        test_div = divs.iloc[-1:].copy()

        # Check if we actually have both classes (Up and Down) to train on
        if train_div["Div_Target"].nunique() <= 1:
            # If the dividend never changed in the training window, 
            # we can't train a binary classifier. Default to historical fact.
            div_pred = train_div["Div_Target"].iloc[0]
            # If pred is 1, confidence of going up is 1.0. If pred is 0, confidence is 0.0.
            div_conf = float(div_pred) 
        else:
            try:
                div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
                # Lowering cv to 3 helps prevent folds from missing a class in small 20-row datasets
                calibrated_div_clf = CalibratedClassifierCV(div_clf, method='isotonic', cv=3, n_jobs=1)
                calibrated_div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred = calibrated_div_clf.predict(test_div[div_predictors])[0]
                div_conf = calibrated_div_clf.predict_proba(test_div[div_predictors])[0,1]
            except ValueError:
                # Ultimate fallback: If a CV fold still ends up with only 1 class, skip calibration
                div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
                div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
                div_pred = div_clf.predict(test_div[div_predictors])[0]
                # Safely grab probability
                if len(div_clf.classes_) == 2:
                    div_conf = div_clf.predict_proba(test_div[div_predictors])[0, 1]
                else:
                    div_conf = float(div_pred)
        
        div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_reg.fit(train_div[div_predictors], train_div["Next_Dividend"])
        forecasted_div = div_reg.predict(test_div[div_predictors])[0]

        today_div = test_div["Dividends"].values[0]
        if div_pred == 1:
            forecasted_div = max(forecasted_div, today_div * 1.001)
        else:
            forecasted_div = min(forecasted_div, today_div * 0.999)
        forecasted_div = round(forecasted_div, 2)
        
    forecasted_yield = round((forecasted_div / today_close) * 100, 2) if today_close > 0 else 0

    # ----------------------
    # Combine results
    # ----------------------
    # Determine the word to use for Dividend Direction
    if pd.isna(next_dividend_date):
        div_direction_text = "N/A"
        final_div_conf = "N/A"
    elif forecasted_div > today_div:
        div_direction_text = "Up"
        final_div_conf = round(div_conf * 100, 2)
    elif forecasted_div < today_div and forecasted_div > 0:
        div_direction_text = "Down"
        # If it's going down, confidence is the inverse of the 'Up' probability
        final_div_conf = round((1 - div_conf) * 100, 2) 
    else:
        div_direction_text = "Flat"
        # If historical data was perfectly flat (triggering our bypass), we are 100% confident it stays flat
        final_div_conf = 100.0 if train_div["Div_Target"].nunique() <= 1 else round((1 - div_conf) * 100, 2)

    return pd.DataFrame({
        "Next_Trading_Day": [next_trading_day_str],
        "Ticker": [ticker],
        "Price_Predicted": ["Up" if test_price["Price_Predicted"].values[0] == 1 else "Down"],
        "Price_Confidence (%)": [round(price_conf*100,2)],
        "Forecasted_Close": [forecasted_close],
        "Next_Dividend_Date": [next_dividend_date.strftime('%Y-%m-%d') if pd.notna(next_dividend_date) else "N/A"],
        "Div_Predicted": [div_direction_text],
        "Div_Confidence (%)": [final_div_conf],
        "Forecasted_Dividend": [forecasted_div if pd.notna(next_dividend_date) else "N/A"],
        "Forecasted_Yield (%)": [forecasted_yield if pd.notna(next_dividend_date) else "N/A"]
    })