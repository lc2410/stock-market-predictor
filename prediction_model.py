import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import date
import logging

# Suppress informational messages from yfinance on import
logging.getLogger('yfinance').setLevel(logging.ERROR)

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
    if len(data) < 1261: # Need at least 1260 for rolling window + 1 for the target day
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
    horizons = [2, 5, 10, 20, 63, 126, 252, 504, 1260] # Corresponds to 2d, 1w, 2w, 1m, 3m, 6m, 1y, 2y, 5y
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

    us_holidays = USFederalHolidayCalendar()
    us_business_day = CustomBusinessDay(calendar=us_holidays)
    today = pd.to_datetime(date.today())
    next_trading_day = today + us_business_day

    # ----------------------
    # Dividend Model
    # ----------------------
    divs = data[["Dividends"]].copy()
    divs = divs[divs["Dividends"] > 0].copy()
    
    # Handle cases with insufficient dividend data
    if len(divs) < 10:
        forecasted_div, div_conf, div_pred, next_dividend_date = 0, 0, 0, pd.NaT
    else:
        # Project the next dividend date using a heuristic based on past dividend frequency
        today = pd.to_datetime(date.today())
        last_div_date = divs.index[-1].tz_localize(None)
        dividend_dates = divs.index.to_series()
        avg_days_between_divs = dividend_dates.diff().mean().days
        
        projected_date = last_div_date + pd.Timedelta(days=avg_days_between_divs)
        
        # If the projected date has already passed, keep adding the interval until it's in the future
        while projected_date < today:
            projected_date += pd.Timedelta(days=avg_days_between_divs)
        next_dividend_date = projected_date

        # Dividend ML model for amount and direction
        divs["Next_Dividend"] = divs["Dividends"].shift(-1)
        divs["Div_Target"] = (divs["Next_Dividend"] > divs["Dividends"]).astype(int)

        # Dividend Features
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

        # Predicts dividend direction (Up/Down) and calibrates confidence
        div_clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        calibrated_div_clf = CalibratedClassifierCV(div_clf, method='isotonic', cv=5, n_jobs=1)
        calibrated_div_clf.fit(train_div[div_predictors], train_div["Div_Target"])
        div_pred = calibrated_div_clf.predict(test_div[div_predictors])[0]
        div_conf = calibrated_div_clf.predict_proba(test_div[div_predictors])[0,1]
        
        # Forecasts the exact dividend amount
        div_reg = RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=1, n_jobs=1)
        div_reg.fit(train_div[div_predictors], train_div["Next_Dividend"])
        forecasted_div = div_reg.predict(test_div[div_predictors])[0]

        # Align dividend regressor
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
    return pd.DataFrame({
        "Next_Trading_Day": [next_trading_day.strftime('%Y-%m-%d')],
        "Ticker": [ticker],
        "Price_Predicted": ["Up" if test_price["Price_Predicted"].values[0] == 1 else "Down"],
        "Price_Confidence (%)": [round(price_conf*100,2)],
        "Forecasted_Close": [forecasted_close],
        "Next_Dividend_Date": [next_dividend_date.strftime('%Y-%m-%d') if pd.notna(next_dividend_date) else "N/A"],
        "Div_Predicted": ["Up" if div_pred == 1 else "Down" if pd.notna(next_dividend_date) and forecasted_div > 0 else "N/A"],
        "Div_Confidence (%)": [round(div_conf*100,2) if pd.notna(next_dividend_date) else "N/A"],
        "Forecasted_Dividend": [forecasted_div if pd.notna(next_dividend_date) else "N/A"],
        "Forecasted_Yield (%)": [forecasted_yield if pd.notna(next_dividend_date) else "N/A"]
    })