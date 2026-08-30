"""External data service for fetching stock market data from Yahoo Finance and Wikipedia."""

import pandas as pd
import requests
import yfinance as yf
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


def fetch_benchmark_tickers():
    """Scrapes Wikipedia to build ticker lists for major US benchmark indices."""
    benchmark_tickers = {"Dow 30": [], "Nasdaq 100": [], "S&P 500": [], "Russell 1000": []}
    
    # Robust "Fuzzy" Wikipedia Scraper
    def get_tickers_from_wiki(url):
        """Scrapes a Wikipedia page to extract ticker symbols and sectors."""
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            html = requests.get(url, headers=headers).text
            tables = pd.read_html(StringIO(html))
            
            best_table = None
            symbol_col = None
            max_len = 0
            
            # 1. Fuzzy match to find the table that contains a Ticker/Symbol column
            for table in tables:
                current_symbol_col = next((c for c in table.columns if isinstance(c, str) and any(keyword in c.lower() for keyword in ['symbol', 'ticker'])), None)
                if current_symbol_col and len(table) > max_len:
                    max_len = len(table)
                    best_table = table
                    symbol_col = current_symbol_col
                            
            if best_table is not None:
                # 2. Fuzzy match to find the Sector/Industry column
                sector_col = next((c for c in best_table.columns if isinstance(c, str) and any(keyword in c.lower() for keyword in ['sector', 'industry'])), None)
                
                symbols = best_table[symbol_col].dropna().tolist()
                sectors = best_table[sector_col].fillna('Unknown').tolist() if sector_col else ['Unknown'] * len(symbols)
                    
                return [{"ticker_symbol": str(symbol).replace('.', '-'), "sector": str(sector).strip()} for symbol, sector in zip(symbols, sectors)]
        except Exception as e:
            logger.error(f"Scraping error {url}: {e}")
        return []

    logger.info("Using fuzzy Wikipedia scraper for all benchmarks...")
    benchmark_tickers["S&P 500"] = get_tickers_from_wiki('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    benchmark_tickers["Dow 30"] = get_tickers_from_wiki('https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies')
    benchmark_tickers["Nasdaq 100"] = get_tickers_from_wiki('https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies')
    benchmark_tickers["Russell 1000"] = get_tickers_from_wiki('https://en.wikipedia.org/wiki/List_of_Russell_1000_companies')
        
    return benchmark_tickers

def fetch_benchmarks():
    """Fetches benchmark indices history and ticker data."""
    benchmarks = {
        "^DJI": "Dow 30",
        "^IXIC": "Nasdaq 100",
        "^GSPC": "S&P 500",
        "^RUI": "Russell 1000"
    }
    results = []
    try:
        end_date = datetime.now() + timedelta(days=1)
        start_date = end_date - timedelta(days=366) # Ensure we encompass a full year to match tickers
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = yf.download(list(benchmarks.keys()), start=start_str, end=end_str, interval="1d", group_by="ticker", progress=False)
        fetched_benchmark_tickers = fetch_benchmark_tickers()
        
        all_tickers = set()
        for t_list in fetched_benchmark_tickers.values():
            for t_obj in t_list:
                all_tickers.add(t_obj["ticker_symbol"])
        all_tickers = list(all_tickers)
        
        quotes = {}
        if all_tickers:
            logger.info(f"Fetching quotes for {len(all_tickers)} tickers via bulk API...")
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            try:
                session.get("https://fc.yahoo.com", timeout=5)
                crumb_res = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=5)
                crumb = crumb_res.text
            except Exception as e:
                logger.error(f"Failed to get crumb: {e}")
                crumb = ""
            
            for i in range(0, len(all_tickers), 100):
                chunk = all_tickers[i:i+100]
                try:
                    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={','.join(chunk)}&crumb={crumb}"
                    res = session.get(url, timeout=10)
                    if res.status_code == 200:
                        for result_item in res.json().get('quoteResponse', {}).get('result', []):
                            sym = result_item.get('symbol')
                            if sym:
                                quotes[sym] = {
                                    "company_name": result_item.get('shortName') or result_item.get('longName') or sym,
                                    "market_cap": result_item.get('marketCap', 0),
                                    "change": result_item.get('regularMarketChangePercent', 0),
                                    "price": result_item.get('regularMarketPrice', 0)
                                }
                except Exception as e:
                    logger.error(f"Quote fetch error: {e}")
                
                time.sleep(1)  # Delay between custom quote chunks
            
        for benchmark_symbol, benchmark_name in benchmarks.items():
            if isinstance(data.columns, pd.MultiIndex):
                df = data[benchmark_symbol].dropna()
            else:
                df = data.dropna() if len(benchmarks) == 1 else data
            
            if df.empty:
                continue
            
            prices = df['Close'].tolist()
            dates = df.index.strftime('%Y-%m-%d').tolist()
            opens = df['Open'].tolist()
            highs = df['High'].tolist()
            lows = df['Low'].tolist()
            volumes = df['Volume'].tolist()
            
            current_price = prices[-1]
            prev_price = prices[-2] if len(prices) > 1 else current_price
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            current_benchmark_tickers = []
            missing_mcap_tickers = []
            
            if benchmark_name in fetched_benchmark_tickers:
                for t_obj in fetched_benchmark_tickers[benchmark_name]:
                    c_ticker = t_obj["ticker_symbol"]
                    c_sector = t_obj["sector"]
                    
                    q_data = quotes.get(c_ticker)
                    if q_data:
                        mcap = q_data.get("market_cap", 0)
                        if not mcap:
                            missing_mcap_tickers.append(c_ticker)
                            
                        current_benchmark_tickers.append({
                            "ticker_symbol": c_ticker,
                            "company_name": q_data.get("company_name", c_ticker),
                            "change": q_data.get("change", 0),
                            "market_cap": mcap,
                            "price": q_data.get("price", 0),
                            "sector": c_sector
                        })
                        
            if missing_mcap_tickers:
                def fetch_mcap(ticker):
                    try:
                        time.sleep(0.5)  # small delay to prevent rate limiting
                        return ticker, yf.Ticker(ticker).fast_info['market_cap']
                    except Exception:
                        return ticker, 0
                        
                with ThreadPoolExecutor(max_workers=5) as executor:
                    mcap_results = dict(executor.map(fetch_mcap, missing_mcap_tickers))
                    
                for c in current_benchmark_tickers:
                    if not c["market_cap"] and c["ticker_symbol"] in mcap_results:
                        c["market_cap"] = mcap_results[c["ticker_symbol"]]
                        
            valid_caps = [c["market_cap"] for c in current_benchmark_tickers if c.get("market_cap") is not None and c["market_cap"] > 0]
            avg_cap = (sum(valid_caps) / len(valid_caps)) if valid_caps else 100_000_000_000
            total_cap = sum(c["market_cap"] if (c.get("market_cap") is not None and c["market_cap"] > 0) else avg_cap for c in current_benchmark_tickers)
            for c in current_benchmark_tickers:
                mcap = c["market_cap"] if (c.get("market_cap") is not None and c["market_cap"] > 0) else avg_cap
                c["weight"] = (mcap / total_cap * 100) if total_cap > 0 else 0

            results.append({
                "benchmark_symbol": benchmark_symbol,
                "benchmark_name": benchmark_name,
                "current_price": current_price,
                "change_pct": change_pct,
                "history": prices,
                "dates": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "volume": volumes,
                "tickers": current_benchmark_tickers
            })
    except Exception as e:
        logger.error(f"Error fetching benchmarks: {e}")
    return results

def fetch_headlines():
    """Fetches general market news."""
    from datetime import datetime
    try:
        spy = yf.Ticker("SPY")
        news = spy.news
        headlines = []
        for news_item in news[:15]:
            if "content" in news_item:
                content = news_item["content"]
                title = content.get("title", "")
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "")
                link_obj = content.get("clickThroughUrl", content.get("canonicalUrl", {}))
                link = link_obj.get("url", "") if isinstance(link_obj, dict) else ""
                time_str = content.get("pubDate", "")
                summary = content.get("summary", "")
                
                try:
                    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    dt = datetime.min
                    
                headlines.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "time": time_str,
                    "dt": dt,
                    "summary": summary
                })
        
        headlines.sort(key=lambda x: x["dt"], reverse=True)
        
        final_headlines = []
        for headline_item in headlines[:10]:
            headline_item.pop("dt", None)
            final_headlines.append(headline_item)
            
        return final_headlines
    except Exception as e:
        logger.error(f"Error fetching headlines: {e}")
        return []
