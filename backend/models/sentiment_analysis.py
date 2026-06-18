import yfinance as yf
from transformers import pipeline

print("Loading FinBERT NLP Model into memory... please wait.")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_news_sentiment(ticker):
    """Scrapes Yahoo Finance news, scores them, and returns pure data arrays instead of HTML."""
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news 
        
        if not news_items:
            return 0.0, {"neutral": "No recent news articles available to analyze."}

        news_items = news_items[:10]

        headlines = []
        for item in news_items:
            if 'title' in item:
                headlines.append(item['title'])
            elif 'content' in item and isinstance(item['content'], dict) and 'title' in item['content']:
                headlines.append(item['content']['title'])
        
        if not headlines:
            return 0.0, {"neutral": "Recent news format was unreadable."}

        results = sentiment_analyzer(headlines)
        score_total = 0
        scored_headlines = []
        
        for hl, res in zip(headlines, results):
            if res['label'] == 'positive':
                val = res['score']
            elif res['label'] == 'negative':
                val = -res['score']
            else:
                val = 0
                
            score_total += val
            scored_headlines.append({"headline": hl, "score": val})
            
        final_score = score_total / len(headlines)

        bullish_drivers = sorted([x for x in scored_headlines if x['score'] > 0.4], key=lambda x: x['score'], reverse=True)[:2]
        bearish_drivers = sorted([x for x in scored_headlines if x['score'] < -0.4], key=lambda x: x['score'])[:2]
        
        # Return a clean dictionary of strings
        news_data = {}
        if bullish_drivers:
            news_data["positive"] = [x['headline'] for x in bullish_drivers]
        if bearish_drivers:
            news_data["negative"] = [x['headline'] for x in bearish_drivers]
        if not bullish_drivers and not bearish_drivers:
            news_data["neutral"] = "News headlines are currently neutral; no major positive or negative catalysts detected."

        return round(final_score, 2), news_data

    except Exception:
        return 0.0, {"neutral": "No recent news data available for this specific asset."}

def calculate_stock_grade(price_direction, price_conf, div_direction, sentiment_score, info, is_fund=False):
    """Symmetric grading system centered at 50 to prevent positive bias."""
    score = 50  # Centered base score
    pos_factors = []
    neg_factors = []
    
    # Machine Learning Price Forecast (+/- 15)
    if price_direction == 1: 
        score += (15 * price_conf)
        pos_factors.append(f"Machine Learning projects an upward price trend with {price_conf*100:.1f}% confidence based on recent technical indicators.")
    else: 
        score -= (15 * price_conf)
        neg_factors.append(f"Machine Learning projects a downward price trend with {price_conf*100:.1f}% confidence based on recent technical indicators.")
        
    # Machine Learning Dividend Forecast (+/- 5)
    if div_direction == 1: 
        score += 5
        pos_factors.append("Algorithm projects the next dividend payout will increase, indicating strong forward-looking cash flow.")
    elif div_direction == 0: 
        score -= 5
        neg_factors.append("Algorithm projects the next dividend payout will decrease, signaling potential cash conservation.")
        
    # News Sentiment Info (+/- 15)
    score += (sentiment_score * 15)
    if sentiment_score > 0.15:
        pos_factors.append(f"Strong positive recent news sentiment (AI Score: {sentiment_score:.2f}). Scores above +0.15 indicate a highly bullish media narrative.")
    elif sentiment_score < -0.15:
        neg_factors.append(f"Strong negative recent news sentiment (AI Score: {sentiment_score:.2f}). A score above +0.15 is required to shift the algorithmic narrative to positive.")

    # Wall Street Consensus (+/- 10)
    recommendation = info.get("recommendationKey", "none").lower()
    if recommendation in ["strong_buy", "buy"]:
        score += 10
        clean_rec = recommendation.replace("_", " ").title()
        pos_factors.append(f"Wall Street consensus is '{clean_rec}', signaling strong institutional confidence.")
    elif recommendation in ["underperform", "sell", "strong_sell"]:
        score -= 10
        clean_rec = recommendation.replace("_", " ").title()
        neg_factors.append(f"Wall Street consensus is '{clean_rec}', signaling institutional caution or expected underperformance.")
    elif recommendation == "hold":
        pos_factors.append("Wall Street consensus is 'Hold', suggesting a balanced risk-to-reward ratio.")
    elif recommendation == "none" and (is_fund or info.get("quoteType") in ["CRYPTOCURRENCY", "INDEX"]):
        pos_factors.append("Wall Street consensus is unavailable (Typical for broad funds, indices, or decentralized crypto).")
    
    # Profitability & Fund Structure (+5 / -10)
    if is_fund or info.get("quoteType") in ["CRYPTOCURRENCY", "INDEX"]:
        score += 5 
        pos_factors.append("Asset provides broad market exposure or is decentralized, shielding it from single-company earnings risks.")
    else:
        eps = info.get("trailingEps", 0)
        if eps and eps > 0: 
            score += 5
            pos_factors.append(f"Demonstrates core profitability with a positive trailing Earnings Per Share (EPS: ${eps}), lowering baseline operational risk.")
        elif eps and eps < 0: 
            score -= 10 
            neg_factors.append(f"Currently operating with a negative trailing Earnings Per Share (EPS: ${eps}), highlighting foundational business challenges.")
            
    # Volatility (+/- 5)
    beta = info.get("beta")
    if beta:
        if beta < 1.0: 
            score += 5
            pos_factors.append(f"Low market volatility (Beta: {beta:.2f} < 1.0), making it a highly stable asset during market turbulence.")
        elif beta > 1.5: 
            score -= 5
            neg_factors.append(f"High market volatility (Beta: {beta:.2f} > 1.5), meaning it may experience aggressive price swings relative to the broader market.")
        
    # Yield (+5)
    yield_pct = info.get("dividendYield", 0) or info.get("yield", 0)
    if yield_pct and yield_pct > 0.03: 
        score += 5
        pos_factors.append(f"Offers a lucrative dividend yield of {yield_pct*100:.2f}%, providing excellent passive income potential.")
    elif yield_pct and yield_pct > 0:
        score += 2
        
    # Scale & Liquidity (+/- 5)
    mcap = info.get("marketCap", 0)
    assets = info.get("totalAssets", 0)
    vol = info.get("averageVolume", 0)
    
    if mcap > 10_000_000_000 or assets > 1_000_000_000:
        score += 5
        pos_factors.append("Massive scale (>$10B Valuation) implies high structural stability and resistance to economic shocks.")
    elif mcap > 0 and mcap < 1_000_000_000:
        score -= 5
        neg_factors.append("Small market capitalization (<$1B Valuation) implies higher risk and susceptibility to price manipulation.")
        
    if vol > 1_000_000:
        score += 2
        pos_factors.append("High average trading volume (>1M shares) ensures excellent liquidity for easy trade execution.")
        
    # Momentum (+/- 5)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    high = info.get("fiftyTwoWeekHigh", 0)
    low = info.get("fiftyTwoWeekLow", 0)
    
    if current_price and high and low and (high - low) > 0:
        range_pct = (current_price - low) / (high - low)
        if range_pct > 0.80: 
            score += 5
            pos_factors.append("Trading in the top 20% of its 52-week range, demonstrating strong upward price momentum.")
        elif range_pct < 0.20: 
            score -= 5
            neg_factors.append("Trading in the bottom 20% of its 52-week range, demonstrating weak price momentum and potential downward pressure.")
        
    # Ensure score strictly stays between 0 and 100
    score = max(0, min(100, score))
        
    # Shifted Grade Scale for a 50-center balance
    if score >= 85: grade = "A+"
    elif score >= 70: grade = "A"
    elif score >= 60: grade = "B"
    elif score >= 50: grade = "C"
    elif score >= 40: grade = "D"
    else: grade = "F"

    if score >= 55: general_sentiment = "Bullish"
    elif score <= 45: general_sentiment = "Bearish"
    else: general_sentiment = "Neutral"
    
    # Return raw data arrays instead of HTML
    fundamentals_data = {}
    if pos_factors: fundamentals_data["positive"] = pos_factors
    if neg_factors: fundamentals_data["negative"] = neg_factors

    return grade, general_sentiment, fundamentals_data