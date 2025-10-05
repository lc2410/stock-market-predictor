# Real-Time Stock & Dividend Forecaster 📈

## 🚀 Overview
This project provides a real-time, next-day forecast for a stock's closing price and its next dividend payment. It uses a sophisticated machine learning pipeline that analyzes historical data to predict not only the direction (Up/Down) of a movement but also the specific value, complete with a calibrated confidence score for each prediction.

The model is designed to be run for any stock ticker available on Yahoo Finance.

## 🛠️ Features
* **Dual Forecasting:** Predicts both the next-day stock price and the next dividend payout.
* **Two-Model System:** For each forecast (price and dividend), it uses a "Director" model to predict the direction and a "Forecaster" model to predict the magnitude.
* **Calibrated Confidence:** Uses `CalibratedClassifierCV` to provide a more reliable and realistic confidence percentage for its directional predictions.
* **Coherent Predictions:** A unique alignment step ensures that the forecasted value (e.g., price) is logically consistent with the predicted direction (e.g., Up), preventing contradictory outputs.
* **Advanced Feature Engineering:** Goes beyond simple price history to analyze volatility, momentum, and price-to-average ratios across multiple time horizons.

## 🤖 The Machine Learning Process

The model's procedure is broken down into two main pipelines: one for price and one for dividends. Both follow a similar logical flow.

### 1. Data Collection
Historical daily price data (Open, High, Low, Close) and dividend payout history are fetched from **Yahoo Finance** using the `yfinance` library.

### 2. Feature Engineering
The model creates a rich set of features ("clues") from the raw historical data to find predictive patterns.

* **Price Features:**
    * **Multi-Horizon Analysis:** It calculates Volatility, Momentum, and Close Price-to-Average Ratios over multiple timeframes: 2 days, 1 week, 2 weeks, 1 month, 3 months, 6 months, 1 year, 2 years, and 5 years.
    * **Sequential Returns:** It uses the specific daily returns from the last 5 trading days as individual features to capture short-term sequential patterns.

* **Dividend Features:**
    * **Growth Rate:** The percentage change from the previous one and two dividend payouts.
    * **Rolling Statistics:** The rolling average and standard deviation over the last 4, 6, and 8 dividend payouts to capture the stability and trend of the dividend policy.

### 3. The "Prediction Team" Model
For both price and dividends, the model uses a two-part prediction team:

* **The Director (Classifier):** A `CalibratedClassifierCV` wrapping a `RandomForestClassifier`.
    * **Job:** To predict the binary direction: will the next value be **Up (1)** or **Down (0)**?
    * **Specialty:** The calibration step ensures the outputted `Confidence (%)` is statistically reliable.

* **The Forecaster (Regressor):** A `RandomForestRegressor`.
    * **Job:** To predict the **exact numerical value** of the next closing price or dividend amount.
    * **Specialty:** It provides a precise value forecast based on the same features as the Director.

### 4. Prediction & Alignment
After both the Director and Forecaster make their initial predictions, a final alignment step ensures the output is logical. For example, if the Director predicts the price will go **Up**, the system guarantees the Forecaster's final output value is higher than the current day's close.

## 🔧 Setup & Installation

Follow these steps to set up and run the project.

**1. Clone the repository:**
Open your terminal and run the following commands to clone the repository and navigate into the project directory: [https://github.com/lc2410/stock-market-predictor.git](https://github.com/lc2410/stock-market-predictor.git)
```bash
git clone https://github.com/lc2410/stock-market-predictor.git
cd stock-market-predictor
```

**2. Check Python Version:**
This script was developed and tested with **Python 3.12.10**. Download python using the link here: [https://www.python.org/downloads/release/python-31210/](https://www.python.org/downloads/release/python-31210/)

**3. Install Required Libraries:**
The repository includes a `requirements.txt` file that lists all necessary libraries. Install them using pip:
```bash
pip install -r requirements.txt
```

---
## ▶️ How to Run

To run a forecast, execute the script from your terminal:
```bash
python prediction-model.py
```
From here it will ask you to type in a stock ticker symbol (e.g., AAPL, VOO) to predict the next day's price and next dividend payout. Once, your done looking at what you need, you can type `quit` to end the session.

## 📊 Sample Output
The script will print a clean, formatted DataFrame to your console with the final forecast:

```
--- Real-Time Next-Day Price + Dividend Forecast (Rounded) ---
   Next_Trading_Day   Close  Price_Predicted  Price_Confidence (%)  Forecasted_Close Next_Dividend_Date  Div_Predicted  Div_Confidence (%)  Forecasted_Dividend  Forecasted_Yield (%)
0        2025-10-06  485.55                1                 62.34            487.12         2025-09-26              1               75.50                 1.85                  0.38
```

## 📚 Core Technologies
* **Python 3.12.10**
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For building and training the machine learning models.
* **yfinance:** For sourcing historical stock and dividend data.