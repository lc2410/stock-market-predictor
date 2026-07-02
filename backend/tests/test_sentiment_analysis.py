import pytest
from backend.models.sentiment_analysis import analyze_news_sentiment, calculate_asset_grade
from unittest.mock import patch, MagicMock

@patch('backend.models.sentiment_analysis.yf.Ticker')
@patch('backend.models.sentiment_analysis.sentiment_analyzer')
def test_analyze_news_sentiment_success(mock_analyzer, mock_ticker):
    """Tests that FinBERT correctly processes and categorizes headlines."""
    mock_instance = MagicMock()
    mock_instance.news = [
        {"title": "Company profits soar"},
        {"content": {"title": "Company faces massive lawsuit"}}
    ]
    mock_ticker.return_value = mock_instance
    
    mock_analyzer.return_value = [
        {'label': 'positive', 'score': 0.8},
        {'label': 'negative', 'score': 0.6}
    ]
    
    score, news_dict = analyze_news_sentiment("TEST")
    assert score == 0.1
    assert "positive" in news_dict
    assert "negative" in news_dict
    assert news_dict["positive"][0]["title"] == "Company profits soar"
    assert news_dict["negative"][0]["title"] == "Company faces massive lawsuit"

@patch('backend.models.sentiment_analysis.yf.Ticker')
def test_analyze_news_sentiment_no_news(mock_ticker):
    """Tests the fallback when Yahoo returns empty news arrays."""
    mock_instance = MagicMock()
    mock_instance.news = []
    mock_ticker.return_value = mock_instance
    
    score, news_dict = analyze_news_sentiment("TEST")
    assert score == 0.0
    assert "neutral" in news_dict

def test_calculate_asset_grade_bullish():
    """Tests the grading engine for a high-performing stock."""
    info = {
        "recommendationKey": "strong_buy",
        "trailingEps": 5.0,
        "beta": 0.8,
        "dividendYield": 0.05,
        "marketCap": 20000000000,
        "averageVolume": 2000000,
        "currentPrice": 150,
        "targetMeanPrice": 200,
        "targetHighPrice": 250,
        "fiftyTwoWeekHigh": 155,
        "fiftyTwoWeekLow": 100,
        "trailingPE": 10,
        "pegRatio": 0.8,
        "52WeekChange": 0.30,
        "SandP52WeekChange": 0.10
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade(
        price_forecasts={"Next_Day": {"Direction": "Up", "Direction_Confidence": 90.0}},
        div_forecasts={"Next_Payout": {"Direction": "Up", "Direction_Confidence": 85.0}},
        sentiment_score=0.8,
        info=info,
        is_fund=False
    )
    
    assert grade in ["A+", "A", "B"]
    assert sentiment == "Bullish"
    assert "positive" in fundamentals
    assert not "negative" in fundamentals

def test_calculate_asset_grade_bearish():
    """Tests the grading engine for a distressed stock."""
    info = {
        "recommendationKey": "strong_sell",
        "trailingEps": -2.0,
        "beta": 2.0,
        "dividendYield": 0.0,
        "averageVolume": 100000,
        "currentPrice": 50,
        "targetMeanPrice": 40,
        "targetHighPrice": 45,
        "fiftyTwoWeekHigh": 150,
        "fiftyTwoWeekLow": 48,
        "trailingPE": 50,
        "pegRatio": 3.0,
        "52WeekChange": -0.20,
        "SandP52WeekChange": 0.10
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade(
        price_forecasts={"Next_Day": {"Direction": "Down", "Direction_Confidence": 80.0}},
        div_forecasts={"Next_Payout": {"Direction": "Down", "Direction_Confidence": 75.0}},
        sentiment_score=-0.7,
        info=info,
        is_fund=False
    )
    
    assert grade in ["D", "F"]
    assert sentiment == "Bearish"
    assert "negative" in fundamentals

def test_calculate_asset_grade_fund_fallback():
    """Tests the smart fallback logic for ETFs and Cryptocurrencies."""
    info = {"quoteType": "ETF"}
    
    grade, sentiment, fundamentals = calculate_asset_grade(
        price_forecasts={"Next_Day": {"Direction": "Up", "Direction_Confidence": 60.0}},
        div_forecasts={"Next_Payout": {"Direction": "Up", "Direction_Confidence": 65.0}},
        sentiment_score=0.1,
        info=info,
        is_fund=True
    )
    
    # Assert that it receives the positive fallback strings instead of penalties
    positives = fundamentals.get("positive", [])
    assert any("Asset Structure: Broad market" in item for item in positives)

def test_calculate_asset_grade_crypto():
    """Tests the grading engine specifically for Cryptocurrencies."""
    info = {
        "quoteType": "CRYPTOCURRENCY",
        "volume24HrMarketCapPercent": 0.08,  # > 0.05 is bullish
        "circulatingSupply": 19000000,
        "maxSupply": 21000000 # > 90% is bullish
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, False)
    positives = fundamentals.get("positive", [])
    assert any("Network is highly active" in item for item in positives)
    assert any("Almost fully mined" in item for item in positives)

def test_calculate_asset_grade_crypto_bearish():
    info = {
        "quoteType": "CRYPTOCURRENCY",
        "volume24HrMarketCapPercent": 0.005,  # < 0.01 is bearish
    }
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, False)
    negatives = fundamentals.get("negative", [])
    assert any("Network is stagnant" in item for item in negatives)

def test_calculate_asset_grade_commodity():
    """Tests the grading engine specifically for Commodities/Macro."""
    info = {
        "quoteType": "FUTURE",
        "exchange": "CMX",
        "openInterest": 150000
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, False)
    positives = fundamentals.get("positive", [])
    assert any("High institutional backing" in item for item in positives)
    assert any("safe-haven" in item for item in positives)

def test_calculate_asset_grade_commodity_bearish():
    info = {
        "quoteType": "FUTURE",
        "openInterest": 5000
    }
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, False)
    negatives = fundamentals.get("negative", [])
    assert any("Low institutional backing" in item for item in negatives)

def test_calculate_asset_grade_etf_advanced():
    """Tests the advanced ETF specific metrics."""
    info = {
        "quoteType": "ETF",
        "netExpenseRatio": 0.0005, # < 0.10%
        "fiveYearAverageReturn": 0.15 # > 10%
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, True)
    positives = fundamentals.get("positive", [])
    assert any("Ultra-low management fees" in item for item in positives)
    assert any("Strong 5-year track record" in item for item in positives)

def test_calculate_asset_grade_etf_advanced_bearish():
    info = {
        "quoteType": "ETF",
        "netExpenseRatio": 0.0080, # > 0.75%
        "fiveYearAverageReturn": -0.05 # < 0%
    }
    
    grade, sentiment, fundamentals = calculate_asset_grade({}, {}, 0, info, True)
    negatives = fundamentals.get("negative", [])
    assert any("Expensive management fees" in item for item in negatives)
    assert any("Poor 5-year track record" in item for item in negatives)