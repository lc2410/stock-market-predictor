import pytest
from backend.models.sentiment_analysis import fetch_news_sentiment, calculate_stock_grade
from unittest.mock import patch, MagicMock

@patch('backend.models.sentiment_analysis.yf.Ticker')
@patch('backend.models.sentiment_analysis.sentiment_analyzer')
def test_fetch_news_sentiment_success(mock_analyzer, mock_ticker):
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
    
    score, news_dict = fetch_news_sentiment("TEST")
    assert score == 0.1
    assert "positive" in news_dict
    assert "negative" in news_dict
    assert "Company profits soar" in news_dict["positive"]

@patch('backend.models.sentiment_analysis.yf.Ticker')
def test_fetch_news_sentiment_no_news(mock_ticker):
    """Tests the fallback when Yahoo returns empty news arrays."""
    mock_instance = MagicMock()
    mock_instance.news = []
    mock_ticker.return_value = mock_instance
    
    score, news_dict = fetch_news_sentiment("TEST")
    assert score == 0.0
    assert "neutral" in news_dict

def test_calculate_stock_grade_bullish():
    """Tests the grading engine for a high-performing stock."""
    info = {
        "recommendationKey": "strong_buy",
        "trailingEps": 5.0,
        "beta": 0.8,
        "dividendYield": 0.05,
        "marketCap": 20000000000,
        "averageVolume": 2000000,
        "currentPrice": 150,
        "fiftyTwoWeekHigh": 155,
        "fiftyTwoWeekLow": 100
    }
    
    grade, sentiment, fundamentals = calculate_stock_grade(
        price_direction=1,
        price_conf=0.9,
        div_direction=1,
        sentiment_score=0.8,
        info=info,
        is_fund=False
    )
    
    assert grade in ["A+", "A", "B"]
    assert sentiment == "Bullish"
    assert "positive" in fundamentals
    assert not "negative" in fundamentals

def test_calculate_stock_grade_bearish():
    """Tests the grading engine for a distressed stock."""
    info = {
        "recommendationKey": "strong_sell",
        "trailingEps": -2.0,
        "beta": 2.0,
        "dividendYield": 0.0,
        "marketCap": 500000000,
        "averageVolume": 500000,
        "currentPrice": 50,
        "fiftyTwoWeekHigh": 150,
        "fiftyTwoWeekLow": 48
    }
    
    grade, sentiment, fundamentals = calculate_stock_grade(
        price_direction=0,
        price_conf=0.8,
        div_direction=0,
        sentiment_score=-0.7,
        info=info,
        is_fund=False
    )
    
    assert grade in ["D", "F"]
    assert sentiment == "Bearish"
    assert "negative" in fundamentals

def test_calculate_stock_grade_fund_fallback():
    """Tests the smart fallback logic for ETFs and Cryptocurrencies."""
    info = {"quoteType": "ETF"}
    
    grade, sentiment, fundamentals = calculate_stock_grade(
        price_direction=1,
        price_conf=0.6,
        div_direction=1,
        sentiment_score=0.1,
        info=info,
        is_fund=True
    )
    
    # Assert that it receives the positive fallback strings instead of penalties
    positives = fundamentals.get("positive", [])
    assert any("broad market exposure" in item for item in positives)
    assert any("Wall Street consensus is unavailable" in item for item in positives)