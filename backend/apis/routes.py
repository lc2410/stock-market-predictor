from flask import Blueprint, jsonify, render_template, Response, stream_with_context
import requests
import json
import logging
import pandas as pd
import yfinance as yf
from cachetools import cached, TTLCache
from backend.models.utils.forecasting_model_utils import get_us_bday, fetch_data, get_chart_data, generate_future_chart_data
from backend.models.price_forecasting import run_price_prediction
from backend.models.dividend_forecasting import run_dividend_prediction
from backend.models.sentiment_analysis import analyze_news_sentiment, calculate_asset_grade

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

def build_frontend_payload(ticker, raw_ml_data, chart_history, nlp_data, info, is_crypto=False):
    """Formats raw ML/NLP math into UI-ready strings and percentages."""
    
    # Fetch Basic Company Info
    try:
        company_name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    # Format Dates
    if is_crypto:
        next_trading_day = (raw_ml_data["anchor_date"] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        next_trading_day = (raw_ml_data["anchor_date"] + get_us_bday()).strftime('%Y-%m-%d')

    next_div_date = raw_ml_data["next_dividend_date"].strftime('%Y-%m-%d') if pd.notna(raw_ml_data["next_dividend_date"]) else "N/A"

    # Construct Final JSON
    return {
        "Ticker": ticker,
        "Company_Name": company_name,
        "Next_Trading_Day": next_trading_day,
        "Next_Dividend_Date": next_div_date,
        "Today_Close": raw_ml_data["today_close"],
        
        # New Grading & NLP Fields
        "Stock_Grade": nlp_data["grade"],
        "News_Sentiment": nlp_data["sentiment"],
        "AI_Reasoning": nlp_data["reasoning"],
        
        # Nested Multi-Horizon Forecasts
        "Price_Forecasts": raw_ml_data["price_forecasts"],
        "Div_Forecasts": raw_ml_data["div_forecasts"],
        
        # Arrays & Charts
        "Chart_Future_Dates": raw_ml_data["chart_future_dates"],
        "Chart_Future_Prices": raw_ml_data["chart_future_prices"],
        "Chart_Future_Upper": raw_ml_data["chart_future_upper"],
        "Chart_Future_Lower": raw_ml_data["chart_future_lower"],
        "Train_Fit_Dates": raw_ml_data["train_fit_dates"],
        "Train_Fit_Prices": raw_ml_data["train_fit_prices"],
        "Div_Future_Dates": raw_ml_data["div_future_dates"],
        "Div_Future_Amounts": raw_ml_data["div_future_amounts"],
        "Div_Future_Upper": raw_ml_data["div_future_upper"],
        "Div_Future_Lower": raw_ml_data["div_future_lower"],
        "Train_Fit_Div_Dates": raw_ml_data["train_fit_div_dates"],
        "Train_Fit_Div_Amounts": raw_ml_data["train_fit_div_amounts"],
        "Chart_History": chart_history
    }

def _fetch_company_fundamentals(safe_ticker):
    """Helper to fetch company info, determine asset type, and extract fund holdings/sectors if applicable."""
    stock_obj = yf.Ticker(safe_ticker)
    try:
        info = stock_obj.info
    except Exception:
        info = {}

    quote_type = info.get("quoteType", "").upper()
    is_fund = quote_type in ["ETF", "MUTUALFUND"]
    is_crypto = quote_type == "CRYPTOCURRENCY"
    top_holdings = []
    top_sectors = []
    
    if is_fund:
        try:
            holdings_data = stock_obj.funds_data.top_holdings
            if holdings_data is not None and not holdings_data.empty:
                for sym, row in holdings_data.head(10).iterrows():
                    weight = None
                    company_name = sym 
                    for val in row.values:
                        if isinstance(val, (float, int)): weight = val
                        elif isinstance(val, str) and val.strip(): company_name = val.strip()
                    val_str = f"{weight * 100:.2f}%" if weight is not None and weight <= 1.0 else (f"{weight:.2f}%" if weight is not None else "")
                    top_holdings.append({"symbol": sym, "name": company_name, "weight": val_str})
            
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
                        top_sectors.append({"sector": clean_sec, "weight": val_str})
        except Exception as e:
            logger.warning(f"Failed to parse Fund data: {e}")
            
    return info, is_fund, is_crypto, top_holdings, top_sectors

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
    logger.info(f" Received prediction request for ticker: {safe_ticker}")
    
    try:
        # Fetch Company Fundamentals
        info, is_fund, is_crypto, top_holdings, top_sectors = _fetch_company_fundamentals(safe_ticker)

        # Run Quantitative ML
        price_data_raw, div_data_raw = fetch_data(safe_ticker, target_window=1260, is_crypto=is_crypto)
        if price_data_raw is None:
            return jsonify({"error": f"Invalid ticker or insufficient data for {safe_ticker}."}), 404
            
        anchor_date = price_data_raw.index[-1]
        today_close = float(price_data_raw["Close"].iloc[-1])

        # Run Price ML
        price_results = run_price_prediction(price_data_raw, is_crypto=is_crypto, price_window=1260)
        chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower = generate_future_chart_data(
            price_results["p_anchors"], price_results["p_lower"], price_results["p_upper"], anchor_date, is_crypto, is_div=False
        )

        # Run Dividend ML
        div_results = run_dividend_prediction(div_data_raw, anchor_date, div_window=25)
        div_future_dates, div_future_amounts, div_future_upper, div_future_lower = generate_future_chart_data(
            div_results["d_anchors"], div_results["d_lower"], div_results["d_upper"], anchor_date, is_crypto, is_div=True, avg_days_between=div_results["avg_days_between"]
        )
        
        train_fit_div_dates = div_results["train_fit_div_dates"][-5:] if div_results["has_enough_div_data"] else []
        train_fit_div_amounts = div_results["train_fit_div_amounts"][-5:] if div_results["has_enough_div_data"] else []
        
        days_in_year = 365 if is_crypto else 252
        train_fit_dates = price_results["train_fit_dates"][-days_in_year:]
        train_fit_prices = price_results["train_fit_prices"][-days_in_year:]
        
        chart_history = get_chart_data(
            price_data=price_data_raw, 
            div_data=div_data_raw,
            is_crypto=is_crypto, 
            show_all_prices=not price_results["has_enough_price_data"], 
            show_all_divs=not div_results["has_enough_div_data"]
        )
        
        raw_ml_data = {
            "anchor_date": anchor_date,
            "today_close": today_close,
            "next_dividend_date": div_results["next_dividend_date"],
            "price_forecasts": price_results["price_forecasts"],
            "chart_future_dates": chart_future_dates,
            "chart_future_prices": chart_future_prices,
            "chart_future_upper": chart_future_upper,
            "chart_future_lower": chart_future_lower,
            "train_fit_dates": train_fit_dates,
            "train_fit_prices": train_fit_prices,
            "div_forecasts": div_results["div_forecasts"],
            "div_future_dates": div_future_dates,
            "div_future_amounts": div_future_amounts,
            "div_future_upper": div_future_upper,
            "div_future_lower": div_future_lower,
            "train_fit_div_dates": train_fit_div_dates,
            "train_fit_div_amounts": train_fit_div_amounts,
        }

        # Run NLP Sentiment Analysis
        sentiment_score, news_dict = analyze_news_sentiment(safe_ticker)
        
        # 6. Calculate Final Asset Grade
        stock_grade, general_sentiment, fundamentals_dict = calculate_asset_grade(
            raw_ml_data.get("price_forecasts", {}), 
            raw_ml_data.get("div_forecasts", {}), 
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
        final_payload = build_frontend_payload(safe_ticker, raw_ml_data, chart_history, nlp_data, info, is_crypto)
        clean_result = sanitize_for_json(final_payload)

        logger.info(f"Successfully generated prediction for {safe_ticker}.")
        return jsonify(clean_result)

    except Exception as e:
        logger.error(f"Error for ticker {safe_ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@api_bp.route('/predict_stream/<string:ticker>', methods=['GET'])
def predict_stream(ticker):
    """SSE Streaming Endpoint for UI Progress updates."""
    safe_ticker = ticker.replace('\n', '').replace('\r', '').upper()
    
    def generate():
        try:
            # Check Cache Manually
            if safe_ticker in forecast_cache:
                yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'result': forecast_cache[safe_ticker]})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'processing', 'step': 'Gathering financial data', 'progress': 15})}\n\n"
            
            # Fetch Company Fundamentals
            info, is_fund, is_crypto, top_holdings, top_sectors = _fetch_company_fundamentals(safe_ticker)

            price_data_raw, div_data_raw = fetch_data(safe_ticker, target_window=1260, is_crypto=is_crypto)
            if price_data_raw is None:
                yield f"data: {json.dumps({'status': 'error', 'error': f'Invalid ticker or insufficient data for {safe_ticker}.'})}\n\n"
                return
                
            anchor_date = price_data_raw.index[-1]
            today_close = float(price_data_raw["Close"].iloc[-1])

            yield f"data: {json.dumps({'status': 'processing', 'step': 'Predicting future prices', 'progress': 30})}\n\n"
            
            # Run Price ML
            price_results = run_price_prediction(price_data_raw, is_crypto=is_crypto, price_window=1260)
            chart_future_dates, chart_future_prices, chart_future_upper, chart_future_lower = generate_future_chart_data(
                price_results["p_anchors"], price_results["p_lower"], price_results["p_upper"], anchor_date, is_crypto, is_div=False
            )

            yield f"data: {json.dumps({'status': 'processing', 'step': 'Predicting dividend payouts', 'progress': 45})}\n\n"
            
            # Run Dividend ML
            div_results = run_dividend_prediction(div_data_raw, anchor_date, div_window=25)
            div_future_dates, div_future_amounts, div_future_upper, div_future_lower = generate_future_chart_data(
                div_results["d_anchors"], div_results["d_lower"], div_results["d_upper"], anchor_date, is_crypto, is_div=True, avg_days_between=div_results["avg_days_between"]
            )
            
            train_fit_div_dates = div_results["train_fit_div_dates"][-5:] if div_results["has_enough_div_data"] else []
            train_fit_div_amounts = div_results["train_fit_div_amounts"][-5:] if div_results["has_enough_div_data"] else []
            
            days_in_year = 365 if is_crypto else 252
            train_fit_dates = price_results["train_fit_dates"][-days_in_year:]
            train_fit_prices = price_results["train_fit_prices"][-days_in_year:]
            
            chart_history = get_chart_data(
                price_data=price_data_raw, 
                div_data=div_data_raw,
                is_crypto=is_crypto, 
                show_all_prices=not price_results["has_enough_price_data"], 
                show_all_divs=not div_results["has_enough_div_data"]
            )
            
            raw_ml_data = {
                "anchor_date": anchor_date,
                "today_close": today_close,
                "next_dividend_date": div_results["next_dividend_date"],
                "price_forecasts": price_results["price_forecasts"],
                "chart_future_dates": chart_future_dates,
                "chart_future_prices": chart_future_prices,
                "chart_future_upper": chart_future_upper,
                "chart_future_lower": chart_future_lower,
                "train_fit_dates": train_fit_dates,
                "train_fit_prices": train_fit_prices,
                "div_forecasts": div_results["div_forecasts"],
                "div_future_dates": div_future_dates,
                "div_future_amounts": div_future_amounts,
                "div_future_upper": div_future_upper,
                "div_future_lower": div_future_lower,
                "train_fit_div_dates": train_fit_div_dates,
                "train_fit_div_amounts": train_fit_div_amounts,
            }

            yield f"data: {json.dumps({'status': 'processing', 'step': 'Reading latest news', 'progress': 60})}\n\n"
            
            # Run NLP Sentiment Analysis (this takes the longest)
            sentiment_score, news_dict = analyze_news_sentiment(safe_ticker)
            
            yield f"data: {json.dumps({'status': 'processing', 'step': 'Analyzing market sentiment', 'progress': 85})}\n\n"
            
            stock_grade, general_sentiment, fundamentals_dict = calculate_asset_grade(
                raw_ml_data.get("price_forecasts", {}), 
                raw_ml_data.get("div_forecasts", {}), 
                sentiment_score,
                info,
                is_fund
            )
            
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

            final_payload = build_frontend_payload(safe_ticker, raw_ml_data, chart_history, nlp_data, info, is_crypto)
            clean_result = sanitize_for_json(final_payload)

            # Store in cache
            forecast_cache[safe_ticker] = clean_result

            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'result': clean_result})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error for ticker {safe_ticker}: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'error': 'An internal server error occurred.'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')