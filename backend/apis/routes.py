from flask import Blueprint, jsonify, render_template
import requests
import logging
import pandas as pd
import yfinance as yf
from cachetools import cached, TTLCache
from backend.models.forecast_model import run_real_time_model, get_chart_data, _get_us_bday
from backend.models.sentiment_analysis import fetch_news_sentiment, calculate_stock_grade

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)
forecast_cache = TTLCache(maxsize=100, ttl=3600)

def sanitize_for_json(obj):
    """Recursively scrubs NaN and Infinity from the payload so JSON.parse never crashes."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if pd.isna(obj) or obj == float('inf') or obj == float('-inf'):
            return None
    elif pd.isna(obj):
        return None
    return obj

def build_frontend_payload(ticker, raw_ml_data, chart_history, nlp_data, info):
    """Formats raw ML/NLP math into UI-ready strings and percentages."""
    
    # Fetch Basic Company Info
    try:
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    # Format Dates
    next_trading_day = (raw_ml_data["anchor_date"] + _get_us_bday()).strftime('%Y-%m-%d')
    next_div_date = raw_ml_data["next_dividend_date"].strftime('%Y-%m-%d') if pd.notna(raw_ml_data["next_dividend_date"]) else "N/A"

    # Format UI Strings & Percentages
    price_pred_str = "Up" if raw_ml_data["price_direction"] == 1 else "Down"
    div_pred_str = "N/A" if raw_ml_data["div_direction"] is None else ("Up" if raw_ml_data["div_direction"] == 1 else "Down")

    price_conf_pct = round(raw_ml_data["price_conf"] * 100, 2)
    div_conf_pct = "N/A" if raw_ml_data["div_direction"] is None else round(raw_ml_data["div_conf"] * 100, 2)
    
    forecasted_yield = (
        round((raw_ml_data["forecasted_div"] / raw_ml_data["today_close"]) * 100, 2)
        if raw_ml_data["today_close"] > 0 and raw_ml_data["forecasted_div"] != "N/A"
        else "N/A"
    )

    # Construct Final JSON
    return {
        "Ticker": ticker,
        "Company_Name": company_name,
        "Next_Trading_Day": next_trading_day,
        
        # New Grading & NLP Fields
        "Stock_Grade": nlp_data["grade"],
        "News_Sentiment": nlp_data["sentiment"],
        "AI_Reasoning": nlp_data["reasoning"],
        
        # Price Forecast
        "Price_Predicted": price_pred_str,
        "Price_Confidence (%)": price_conf_pct,
        "Forecasted_Close": raw_ml_data["forecasted_close"],
        
        # Dividend Forecast
        "Next_Dividend_Date": next_div_date,
        "Div_Predicted": div_pred_str,
        "Div_Confidence (%)": div_conf_pct,
        "Forecasted_Dividend": raw_ml_data["forecasted_div"],
        "Forecasted_Yield (%)": forecasted_yield,
        
        # Arrays & Charts
        "Extended_Forecasts": raw_ml_data["extended_forecasts"],
        "Chart_Future_Dates": raw_ml_data["chart_future_dates"],
        "Chart_Future_Prices": raw_ml_data["chart_future_prices"],
        "Chart_Future_Upper": raw_ml_data["chart_future_upper"],
        "Chart_Future_Lower": raw_ml_data["chart_future_lower"],
        "Train_Fit_Dates": raw_ml_data["train_fit_dates"],
        "Train_Fit_Prices": raw_ml_data["train_fit_prices"],
        "Div_Extended_Forecasts": raw_ml_data["div_extended_forecasts"],
        "Div_Future_Dates": raw_ml_data["div_future_dates"],
        "Div_Future_Amounts": raw_ml_data["div_future_amounts"],
        "Div_Future_Upper": raw_ml_data["div_future_upper"],
        "Div_Future_Lower": raw_ml_data["div_future_lower"],
        "Train_Fit_Div_Dates": raw_ml_data["train_fit_div_dates"],
        "Train_Fit_Div_Amounts": raw_ml_data["train_fit_div_amounts"],
        "Chart_History": chart_history
    }

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
            for q in quotes if "symbol" in q
        ]
        return jsonify(results)
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify([])

@api_bp.route('/predict/<string:ticker>', methods=['GET'])
@cached(cache=forecast_cache)
def predict(ticker):
    """Orchestrates ML math, NLP sentiment, Fundamentals, and UI formatting into a single payload."""
    safe_ticker = ticker.replace('\n', '').replace('\r', '').upper()
    logger.info(f"Received prediction request for ticker: {safe_ticker}")
    
    try:
        # Fetch Company Fundamentals
        stock_obj = yf.Ticker(safe_ticker)
        try:
            info = stock_obj.info
        except Exception:
            info = {}

        quote_type = info.get("quoteType", "").upper()
        is_fund = quote_type in ["ETF", "MUTUALFUND"]
        top_holdings = []
        top_sectors = []
        
        if is_fund:
            try:
                # A. Extract Top 10 Holdings as Dictionaries
                holdings_data = stock_obj.funds_data.top_holdings
                if holdings_data is not None and not holdings_data.empty:
                    for sym, row in holdings_data.head(10).iterrows():
                        weight = None
                        company_name = sym 
                        
                        for val in row.values:
                            if isinstance(val, (float, int)): weight = val
                            elif isinstance(val, str) and val.strip(): company_name = val.strip()
                        
                        # Store as structured dict
                        val_str = f"{weight * 100:.2f}%" if weight is not None and weight <= 1.0 else (f"{weight:.2f}%" if weight is not None else "")
                        top_holdings.append({
                            "symbol": sym,
                            "name": company_name,
                            "weight": val_str
                        })
                
                # B. Extract Sector Weightings as Dictionaries
                sector_data = stock_obj.funds_data.sector_weightings
                if sector_data is not None:
                    sector_dict = sector_data.to_dict() if isinstance(sector_data, pd.Series) else sector_data
                    sorted_sectors = sorted(sector_dict.items(), key=lambda item: item[1], reverse=True)
                    
                    for raw_sector, weight in sorted_sectors:
                        if isinstance(weight, (float, int)) and weight > 0:
                            clean_sec = raw_sector.replace('_', ' ').title()
                            if clean_sec.lower() == 'realestate': clean_sec = 'Real Estate'
                            elif clean_sec.lower() == 'basicmaterials': clean_sec = 'Basic Materials'
                            elif clean_sec.lower() == 'financialservices': clean_sec = 'Financial Services'
                            elif clean_sec.lower() == 'communicationservices': clean_sec = 'Communication Services'
                            
                            val_str = f"{weight * 100:.2f}%" if weight <= 1.0 else f"{weight:.2f}%"
                            top_sectors.append({
                                "sector": clean_sec,
                                "weight": val_str
                            })

            except Exception as e:
                logger.warning(f"Failed to parse Fund data: {e}")

        # Run Quantitative ML
        raw_ml_data = run_real_time_model(safe_ticker)
        if raw_ml_data is None:
            return jsonify({"error": f"Invalid ticker or insufficient data for {safe_ticker}."}), 404
        
        chart_history = get_chart_data(safe_ticker, None)

        # Run NLP Sentiment Analysis
        sentiment_score, news_dict = fetch_news_sentiment(safe_ticker)
        
        # Calculate Overall Grade
        div_dir = -1 if raw_ml_data["div_direction"] is None else raw_ml_data["div_direction"]
        stock_grade, general_sentiment, fundamentals_dict = calculate_stock_grade(
            raw_ml_data["price_direction"], 
            raw_ml_data["price_conf"], 
            div_dir, 
            sentiment_score,
            info,
            is_fund
        )
        
        # Assemble Master Reasoning JSON Object
        master_reasoning = {
            "news": news_dict,
            "fundamentals": fundamentals_dict,
            "etf_holdings": top_holdings,
            "etf_sectors": top_sectors
        }
        
        nlp_data = {
            "sentiment": general_sentiment,
            "reasoning": master_reasoning,
            "grade": stock_grade
        }

        # Format and Package the Final JSON Payload
        final_payload = build_frontend_payload(safe_ticker, raw_ml_data, chart_history, nlp_data, info)
        clean_result = sanitize_for_json(final_payload)

        logger.info(f"Successfully generated prediction for {safe_ticker}.")
        return jsonify(clean_result)

    except Exception as e:
        logger.error(f"Error for ticker {safe_ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500