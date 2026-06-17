from flask import Blueprint, jsonify, render_template
import requests
import logging
import pandas as pd
from backend.models.forecast_model import run_real_time_model, get_chart_data

# This file defines the API routes for the Flask application.
api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

def sanitize_for_json(obj):
    """Recursively scrubs NaN and Infinity from the payload so JSON.parse never crashes."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        # Convert NaN, inf, and -inf to None (which becomes 'null' in JSON)
        if pd.isna(obj) or obj == float('inf') or obj == float('-inf'):
            return None
    elif pd.isna(obj): # Catches pd.NaT and other numpy NaN types
        return None
    return obj

@api_bp.route('/')
def home():
    return render_template('index.html')

@api_bp.route('/search/<string:query>', methods=['GET'])
def search(query):
    """Proxy Yahoo Finance autocomplete to bypass browser CORS restrictions."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        quotes = data.get('quotes', [])
        results = [
            {"symbol": q.get("symbol"), "name": q.get("shortname", "")} 
            for q in quotes if q.get("quoteType") in ["EQUITY", "ETF"]
        ]
        return jsonify(results)
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify([])

@api_bp.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    """Run the ML pipeline and return the forecast alongside historical chart data."""
    safe_ticker = ticker.replace('\n', '').replace('\r', '')
    logger.info(f"Received prediction request for ticker: {safe_ticker}")
    try:
        prediction_df = run_real_time_model(ticker.upper())
        if prediction_df is None:
            logger.warning(f"Could not generate prediction for {safe_ticker}.")
            return jsonify({"error": f"Invalid ticker or insufficient data for {safe_ticker}."}), 404
        
        result = prediction_df.to_dict(orient='records')[0]
        result['Chart_History'] = get_chart_data(ticker, None)

        clean_result = sanitize_for_json(result)

        logger.info(f"Successfully generated prediction for {safe_ticker}.")
        return jsonify(clean_result)

    except Exception as e:
        logger.error(f"Error for ticker {safe_ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500