import os
import yfinance as yf
from transformers import pipeline

print("Loading FinBERT NLP Model into memory... please wait.")
model_dir = os.path.join(os.path.dirname(__file__), "finbert_weights")
if os.path.exists(model_dir):
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_dir)
else:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_news_sentiment(ticker):
    """Scrapes Yahoo Finance news, scores them, and returns pure data arrays instead of HTML."""
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news 
        
        if not news_items:
            return 0.0, {"neutral": "No recent news articles available to analyze."}

        news_items = news_items[:10]

        headlines = []
        links = []
        summaries = []
        publishers = []
        for item in news_items:
            title = None
            link = None
            summary = ""
            publisher = "Unknown Publisher"
            
            if 'title' in item:
                title = item['title']
                link = item.get('link', None)
                summary = item.get('summary', '')
                publisher = item.get('publisher', 'Unknown Publisher')
            elif 'content' in item and isinstance(item['content'], dict) and 'title' in item['content']:
                title = item['content']['title']
                
                url_data = item['content'].get('clickThroughUrl') or item['content'].get('canonicalUrl')
                if isinstance(url_data, dict) and 'url' in url_data:
                    link = url_data['url']
                else:
                    link = item['content'].get('url', None)
                
                summary = item['content'].get('summary', '')
                provider = item['content'].get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName', 'Unknown Publisher')
                elif isinstance(provider, str):
                    publisher = provider

            if title:
                headlines.append(title)
                links.append(link if link else "#")
                summaries.append(summary)
                publishers.append(publisher)
        
        if not headlines:
            return 0.0, {"neutral": "Recent news format was unreadable."}

        results = sentiment_analyzer(headlines)
        score_total = 0
        scored_headlines = []
        
        for hl, lnk, sumry, pub, res in zip(headlines, links, summaries, publishers, results):
            if res['label'] == 'positive':
                val = res['score']
            elif res['label'] == 'negative':
                val = -res['score']
            else:
                val = 0
                
            score_total += val
            scored_headlines.append({
                "headline": hl, 
                "score": val, 
                "url": lnk,
                "summary": sumry,
                "publisher": pub
            })
            
        final_score = score_total / len(headlines)

        bullish_drivers = sorted([x for x in scored_headlines if x['score'] > 0.4], key=lambda x: x['score'], reverse=True)[:2]
        bearish_drivers = sorted([x for x in scored_headlines if x['score'] < -0.4], key=lambda x: x['score'])[:2]
        
        # Return a clean dictionary of strings
        news_data = {}
        if bullish_drivers:
            news_data["positive"] = [
                {"title": x['headline'], "url": x['url'], "summary": x['summary'], "publisher": x['publisher']} 
                for x in bullish_drivers
            ]
        if bearish_drivers:
            news_data["negative"] = [
                {"title": x['headline'], "url": x['url'], "summary": x['summary'], "publisher": x['publisher']} 
                for x in bearish_drivers
            ]
        if not bullish_drivers and not bearish_drivers:
            news_data["neutral"] = "News headlines are currently neutral."

        return round(final_score, 2), news_data

    except Exception:
        return 0.0, {"neutral": "No recent news data available for this specific asset."}

def calculate_asset_grade(price_forecasts, div_forecasts, sentiment_score, info, is_fund=False):
    """Symmetric grading system centered at 50 to prevent positive bias."""
    score = 50  # Centered base score
    pos_factors = []
    neg_factors = []
    
    # Machine Learning Price Forecast (+/- 8)
    if price_forecasts:
        up_count = sum(1 for data in price_forecasts.values() if data["Direction"] == "Up")
        down_count = sum(1 for data in price_forecasts.values() if data["Direction"] == "Down")
        conf_sum = sum(data["Direction_Confidence"] / 100.0 for data in price_forecasts.values())
        
        avg_conf = conf_sum / len(price_forecasts)
        if up_count > down_count:
            score += 8
            pos_factors.append(f"ML Price Forecast: Predicts a price increase across {len(price_forecasts)} horizons ({avg_conf*100:.1f}% avg confidence).")
        elif down_count > up_count:
            score -= 8
            neg_factors.append(f"ML Price Forecast: Predicts a price decrease across {len(price_forecasts)} horizons ({avg_conf*100:.1f}% avg confidence).")
        
    # Machine Learning Dividend Forecast (+/- 2)
    if div_forecasts:
        up_div = sum(1 for d in div_forecasts.values() if d["Direction"] == "Up")
        down_div = sum(1 for d in div_forecasts.values() if d["Direction"] == "Down")
        div_conf_sum = sum(d["Direction_Confidence"] / 100.0 for d in div_forecasts.values())
        
        avg_div_conf = div_conf_sum / len(div_forecasts)
        if up_div > down_div:
            score += 2
            pos_factors.append(f"ML Dividend Forecast: Predicts a dividend increase across {len(div_forecasts)} horizons ({avg_div_conf*100:.1f}% avg confidence).")
        elif down_div > up_div:
            score -= 2
            neg_factors.append(f"ML Dividend Forecast: Predicts a dividend decrease across {len(div_forecasts)} horizons ({avg_div_conf*100:.1f}% avg confidence).")
        
    # News Sentiment Info (+/- 15)
    adjusted_sentiment = sentiment_score - 0.10
    score += (adjusted_sentiment * 15)
    if adjusted_sentiment > 0.05:
        pos_factors.append("Media: News headlines are mostly positive.")
    elif adjusted_sentiment < -0.25:
        neg_factors.append("Media: News headlines are mostly negative.")

    # Wall Street Consensus (+5 / -15)
    recommendation = info.get("recommendationKey", "none").lower()
    if recommendation == "strong_buy":
        score += 5
        pos_factors.append("Wall Street Consensus: Strong Buy.")
    elif recommendation == "buy":
        score += 2
        pos_factors.append("Wall Street Consensus: Buy.")
    elif recommendation in ["underperform", "sell", "strong_sell"]:
        score -= 15
        clean_rec = recommendation.replace("_", " ").title()
        neg_factors.append(f"Wall Street Consensus: {clean_rec}.")
    elif recommendation == "hold":
        score -= 5
        neg_factors.append("Wall Street Consensus: Hold.")
        
    # Target Price Upside Potential
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    target_mean = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    
    if current_price and target_mean and target_high:
        if current_price > target_high:
            score -= 15
            neg_factors.append("Valuation: Price is way higher than Wall Street targets.")
        elif current_price > target_mean:
            score -= 10
            neg_factors.append("Valuation: Price is slightly higher than Wall Street targets.")
        elif current_price < target_mean:
            upside = (target_mean - current_price) / current_price
            if upside > 0.05:
                score += 5
                pos_factors.append("Valuation: Room to grow to meet Wall Street targets.")
    
    # Profitability & Fund Structure (+10 / -15)
    if is_fund or info.get("quoteType") in ["CRYPTOCURRENCY", "INDEX"]:
        score += 10 
        pos_factors.append("Asset Structure: Broad market fund/index or decentralized crypto.")
    else:
        eps = info.get("trailingEps", 0)
        if eps and eps > 0: 
            score += 10
            pos_factors.append("Profits: The company is making money.")
        elif eps and eps < 0: 
            score -= 15 
            neg_factors.append("Profits: The company is losing money.")
            
    # Volatility (+8 / -15)
    beta = info.get("beta")
    if beta is not None:
        if beta < 0.85: 
            score += 8
            pos_factors.append("Risk: Low price swings compared to the market.")
        elif beta > 1.25: 
            score -= 15
            neg_factors.append("Risk: High price swings compared to the market.")
        
    # Yield (+8 / -4)
    yield_pct = info.get("dividendYield", 0) or info.get("yield", 0)
    quote_type = info.get("quoteType", "")
    is_crypto_or_commodity = quote_type in ["CRYPTOCURRENCY", "FUTURE", "CURRENCY"] or info.get("exchange") in ["CCC", "CMX"]
    
    if not is_crypto_or_commodity:
        if yield_pct and yield_pct > 0.03: 
            score += 8
            pos_factors.append(f"Dividends: Pays a high cash yield ({yield_pct*100:.2f}%).")
        elif yield_pct and yield_pct > 0:
            score += 3
        else:
            score -= 4
            neg_factors.append("Income: No dividend yield.")
            
    # Scale & Liquidity (+5 / -8)
    mcap = info.get("marketCap", 0)
    assets = info.get("totalAssets", 0)
    vol = info.get("averageVolume", 0) or info.get("volume24Hr", 0)
    
    if mcap > 100_000_000_000 or assets > 1_000_000_000:
        score += 5
        pos_factors.append("Size: Massive, highly stable company.")
    elif mcap > 0 and mcap < 2_000_000_000:
        score -= 8
        neg_factors.append("Size: Smaller company, carries more risk.")
        
    if vol > 1_000_000:
        score += 5
        pos_factors.append("Trading: Very easy to buy and sell shares.")
    elif vol > 0 and vol < 500_000:
        score -= 8
        neg_factors.append("Trading: Harder to buy and sell shares quickly.")
        
    # Momentum (+/- 12)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    high = info.get("fiftyTwoWeekHigh", 0)
    low = info.get("fiftyTwoWeekLow", 0)
    
    if current_price and high and low and (high - low) > 0:
        range_pct = (current_price - low) / (high - low)
        if range_pct > 0.75: 
            score += 12
            pos_factors.append("Momentum: Price is trending near its yearly high.")
        elif range_pct < 0.25: 
            score -= 12
            neg_factors.append("Momentum: Price is trending near its yearly low.")

    # Alpha / Relative S&P 500 Outperformance (+5 / -5)
    stock_52w = info.get("52WeekChange")
    sp52w = info.get("SandP52WeekChange")
    if stock_52w is not None and sp52w is not None:
        alpha = stock_52w - sp52w
        if alpha > 0.10:
            score += 5
            pos_factors.append(f"Market: Beating the S&P 500 by {alpha*100:.1f}%.")
        elif alpha < -0.10:
            score -= 5
            neg_factors.append(f"Market: Losing to the S&P 500 by {abs(alpha)*100:.1f}%.")

    # Asset-Class Specific Logic
    if quote_type in ["CRYPTOCURRENCY"]:
        # Crypto
        vol_mcap = info.get("volume24HrMarketCapPercent", 0)
        if vol_mcap > 0.05:
            score += 5
            pos_factors.append("Crypto: Network is highly active.")
        elif vol_mcap > 0 and vol_mcap < 0.01:
            score -= 5
            neg_factors.append("Crypto: Network is stagnant.")
            
        circ = info.get("circulatingSupply", 0)
        max_sup = info.get("maxSupply", 0)
        if max_sup > 0 and (circ / max_sup) >= 0.90:
            score += 5
            pos_factors.append("Crypto: Almost fully mined (Scarcity).")
            
    elif quote_type in ["FUTURE", "CURRENCY"] or info.get("exchange") in ["CMX", "NYM", "CCY"]:
        # Commodities / Currencies
        open_int = info.get("openInterest", 0)
        if open_int > 100_000:
            score += 5
            pos_factors.append("Big Money: High institutional backing.")
        elif open_int > 0 and open_int < 10_000:
            score -= 5
            neg_factors.append("Big Money: Low institutional backing.")
            
        score += 2
        pos_factors.append("Safety: Acts as a safe-haven asset.")

    elif is_fund:
        # ETFs / Index Funds
        expense = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")
        if expense is not None:
            if expense < 0.0010:
                score += 5
                pos_factors.append("Fees: Ultra-low management fees.")
            elif expense > 0.0075:
                score -= 5
                neg_factors.append("Fees: Expensive management fees.")
                
        five_yr = info.get("fiveYearAverageReturn")
        if five_yr is not None:
            if five_yr > 0.10:
                score += 5
                pos_factors.append("Growth: Strong 5-year track record.")
            elif five_yr < 0:
                score -= 5
                neg_factors.append("Growth: Poor 5-year track record.")

    else:
        # Individual Stocks
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None:
            if pe > 0 and pe < 15:
                score += 5
                pos_factors.append(f"Valuation: Cheap relative to earnings (P/E: {pe:.1f}).")
            elif pe > 30:
                score -= 5
                neg_factors.append(f"Valuation: Expensive relative to earnings (P/E: {pe:.1f}).")
                
        peg = info.get("pegRatio") or info.get("trailingPegRatio")
        if peg is not None:
            if peg > 0 and peg < 1.0:
                score += 5
                pos_factors.append(f"Growth: Cheap relative to growth speed (PEG: {peg:.2f}).")
            elif peg > 2.0:
                score -= 5
                neg_factors.append(f"Growth: Expensive relative to growth speed (PEG: {peg:.2f}).")
        
    # Ensure score strictly stays between 0 and 100
    print("raw score:", score)
    score = max(0, min(100, score))
    
    # Shifted Grade Scale for a Strict 50-center balance
    if score >= 90: grade = "A+"
    elif score >= 80: grade = "A"
    elif score >= 65: grade = "B"
    elif score >= 50: grade = "C"
    elif score >= 35: grade = "D"
    else: grade = "F"

    if score >= 60: general_sentiment = "Bullish"
    elif score <= 40: general_sentiment = "Bearish"
    else: general_sentiment = "Neutral"
    
    # Return raw data arrays instead of HTML
    fundamentals_data = {}
    if pos_factors: fundamentals_data["positive"] = pos_factors
    if neg_factors: fundamentals_data["negative"] = neg_factors

    return grade, general_sentiment, fundamentals_data