from flask import Flask, jsonify, render_template
from flask_cors import CORS
from prediction_model import run_real_time_model, get_chart_data
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search/<string:query>', methods=['GET'])
def search(query):
    try:
        # Query Yahoo Finance's autocomplete API
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Filter for equities/ETFs and return symbol + name
        quotes = data.get('quotes', [])
        results = [
            {"symbol": q.get("symbol"), "name": q.get("shortname", "")} 
            for q in quotes if q.get("quoteType") in ["EQUITY", "ETF"]
        ]
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Search API error: {e}")
        return jsonify([])

@app.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    app.logger.info(f"Received prediction request for ticker: {ticker}")
    try:
        prediction_df = run_real_time_model(ticker.upper())
        if prediction_df is None:
            app.logger.warning(f"Could not generate prediction for {ticker}.")
            return jsonify({"error": f"Invalid ticker or insufficient data for {ticker}."}), 404
        
        result = prediction_df.to_dict(orient='records')[0]
        
        chart_data = get_chart_data(ticker, None)
        result['Chart_History'] = chart_data

        app.logger.info(f"Successfully generated prediction for {ticker}.")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error for ticker {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)