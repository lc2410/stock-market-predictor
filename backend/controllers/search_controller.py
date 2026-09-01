"""Search controller proxying Yahoo Finance autocomplete."""

from flask import Blueprint, jsonify
import requests
import logging

search_bp = Blueprint('search', __name__)
logger = logging.getLogger(__name__)

@search_bp.route('/search/<string:query>', methods=['GET'])
def search(query):
    """Proxy Yahoo Finance autocomplete to bypass browser CORS restrictions."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        quotes = data.get('quotes', [])
        results = [
            {"symbol": quote.get("symbol"), "name": quote.get("shortname", "")} 
            for quote in quotes if "symbol" in quote
        ]
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Search API error: {e}")
        return jsonify([])
