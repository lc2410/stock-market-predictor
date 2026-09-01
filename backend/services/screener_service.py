"""Screener service for aggregating stock screening dashboard data."""

import pandas as pd
import logging
from utils.db_utils import get_latest_benchmarks, get_latest_headlines, get_historical_prices_df
from utils.service_utils import calculate_change_pct, calculate_average_volume, calculate_volatility, calculate_ttm_dividend_yield, calculate_rsi

logger = logging.getLogger(__name__)

def _calculate_ticker_metrics(ticker, df):
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    change_pct = calculate_change_pct(current_price, prev_price)
    
    current_volume = df['Volume'].iloc[-1]
    avg_volume = calculate_average_volume(df, window=30)
    
    high_today = df['High'].iloc[-1]
    low_today = df['Low'].iloc[-1]
    
    period_high = df['High'].max()
    period_low = df['Low'].min()
    
    if len(df) > 1:
        prev_df = df.iloc[:-1]
        prev_period_high = prev_df['High'].max()
        prev_period_low = prev_df['Low'].min()
        prev_high_date = prev_df['High'].idxmax().strftime('%Y-%m-%d')
        prev_low_date = prev_df['Low'].idxmin().strftime('%Y-%m-%d')
    else:
        prev_period_high = period_high
        prev_period_low = period_low
        prev_high_date = df.index[-1].strftime('%Y-%m-%d')
        prev_low_date = df.index[-1].strftime('%Y-%m-%d')
        
    breakout_high_pct = ((current_price - prev_period_high) / prev_period_high) * 100 if prev_period_high > 0 else 0
    breakout_low_pct = ((current_price - prev_period_low) / prev_period_low) * 100 if prev_period_low > 0 else 0
    
    volatility = calculate_volatility(high_today, low_today)
    rsi = calculate_rsi(df).iloc[-1]
    
    dividend_payout = 0
    dividend_yield = 0
    last_dividend_date = None
    if 'Dividends' in df.columns:
        div_series = pd.to_numeric(df['Dividends'], errors='coerce').fillna(0)
        one_year_ago = df.index.max() - pd.Timedelta(days=365)
        recent_dividends = div_series[df.index >= one_year_ago]
        dividend_payout = recent_dividends.sum()
        dividend_yield = calculate_ttm_dividend_yield(df, current_price) * 100
        nonzero_divs = div_series[div_series > 0]
        if not nonzero_divs.empty:
            last_dividend_date = nonzero_divs.index[-1].strftime('%Y-%m-%d')
    
    avg_volume_52w = float(df['Volume'].mean())
    vol_change_pct = ((current_volume - avg_volume_52w) / avg_volume_52w) * 100 if avg_volume_52w > 0 else 0.0

    return {
        "symbol": ticker,
        "name": ticker,
        "price": float(current_price),
        "prev_price": float(prev_price),
        "change": float(change_pct),
        "volume": float(current_volume),
        "rsi": float(rsi) if not pd.isna(rsi) else 0.0,
        "volatility": float(volatility) if not pd.isna(volatility) else 0.0,
        "avg_volume": float(avg_volume) if not pd.isna(avg_volume) else 0.0,
        "avg_volume_52w": avg_volume_52w,
        "vol_change_pct": float(vol_change_pct),
        "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 0.0,
        "dividend_payout": float(dividend_payout) if not pd.isna(dividend_payout) else 0.0,
        "dividend_yield": float(dividend_yield) if not pd.isna(dividend_yield) else 0.0,
        "last_dividend_date": last_dividend_date,
        "period_high": float(period_high) if not pd.isna(period_high) else float(current_price),
        "period_low": float(period_low) if not pd.isna(period_low) else float(current_price),
        "prev_period_high": float(prev_period_high) if not pd.isna(prev_period_high) else float(current_price),
        "prev_period_low": float(prev_period_low) if not pd.isna(prev_period_low) else float(current_price),
        "prev_high_date": prev_high_date,
        "prev_low_date": prev_low_date,
        "breakout_high_pct": float(breakout_high_pct),
        "breakout_low_pct": float(breakout_low_pct)
    }

def _compute_benchmark_scans(benchmark_metrics):
    benchmark_scans = {
        "new_high": [],
        "new_low": [],
        "overbought": [],
        "oversold": [],
        "unusual_volume": [],
        "most_volatile": [],
        "biggest_dividends": []
    }
    benchmark_movers = {
        "day_gainers": [],
        "day_losers": [],
        "most_actives": []
    }
    
    for metric in benchmark_metrics:
        if metric["price"] > metric.get("prev_period_high", metric["price"]):
            benchmark_scans["new_high"].append(metric)
        if metric["price"] < metric.get("prev_period_low", metric["price"]):
            benchmark_scans["new_low"].append(metric)
        if metric["rsi"] > 70:
            benchmark_scans["overbought"].append(metric)
        if metric["rsi"] < 30:
            benchmark_scans["oversold"].append(metric)
            
    benchmark_scans["new_high"].sort(key=lambda x: x["breakout_high_pct"], reverse=True)
    benchmark_scans["new_low"].sort(key=lambda x: x["breakout_low_pct"])
    benchmark_scans["overbought"].sort(key=lambda x: x["rsi"], reverse=True)
    benchmark_scans["oversold"].sort(key=lambda x: x["rsi"])
    
    gainers = [m for m in benchmark_metrics if m["change"] > 0]
    losers = [m for m in benchmark_metrics if m["change"] < 0]
    
    benchmark_movers["day_gainers"] = sorted(gainers, key=lambda x: x["change"], reverse=True)[:10]
    benchmark_movers["day_losers"] = sorted(losers, key=lambda x: x["change"])[:10]
    benchmark_movers["most_actives"] = sorted(benchmark_metrics, key=lambda x: x["volume"], reverse=True)[:10]
    benchmark_scans["most_volatile"] = sorted(benchmark_metrics, key=lambda x: x["volatility"], reverse=True)[:10]
    
    unusual_vol = [m for m in benchmark_metrics if m.get("vol_change_pct", 0) > 0]
    benchmark_scans["unusual_volume"] = sorted(unusual_vol, key=lambda x: x.get("vol_change_pct", 0), reverse=True)[:10]
    
    dividend_payers = [metric for metric in benchmark_metrics if metric["dividend_payout"] > 0]
    benchmark_scans["biggest_dividends"] = sorted(dividend_payers, key=lambda x: x["dividend_yield"], reverse=True)[:10]
    
    for scan_key in benchmark_scans.keys():
        benchmark_scans[scan_key] = benchmark_scans[scan_key][:10]
        
    return benchmark_movers, benchmark_scans

def process_custom_scans_by_benchmark(data, benchmarks_list):
    """Calculates custom scans using entire benchmark universe data, partitioned by benchmark."""
    movers_by_benchmark = {}
    scans_by_benchmark = {}
    
    stock_metrics = {}
    tickers = []
    
    if isinstance(data.columns, pd.MultiIndex):
        tickers = list(data.columns.levels[0])
    
    for ticker in tickers:
        df = data[ticker].dropna(subset=['Close'])
        if df.empty or len(df) < 15:
            continue
        stock_metrics[ticker] = _calculate_ticker_metrics(ticker, df)
        
    for benchmark in benchmarks_list:
        benchmark_name = benchmark["name"]
        benchmark_tickers = [constituent["symbol"] for constituent in benchmark["constituents"]]
        
        benchmark_metrics = []
        for ticker in benchmark_tickers:
            if ticker in stock_metrics:
                metric_copy = stock_metrics[ticker].copy()
                constituent_obj = next((c for c in benchmark["constituents"] if c["symbol"] == ticker), None)
                if constituent_obj:
                    constituent_obj["change"] = metric_copy["change"]
                    constituent_obj["price"] = metric_copy["price"]
                    constituent_obj["volume"] = metric_copy["volume"]
                    metric_copy["name"] = constituent_obj["name"]
                
                benchmark_metrics.append(metric_copy)
                
        movers, scans = _compute_benchmark_scans(benchmark_metrics)
        scans_by_benchmark[benchmark_name] = scans
        movers_by_benchmark[benchmark_name] = movers
        
    return movers_by_benchmark, scans_by_benchmark

def get_screener_dashboard_data():
    """Aggregates all screener data from local SQLite database."""
    logger.info("Fetching fresh screener data from SQLite")
    
    benchmarks = get_latest_benchmarks()
    news = get_latest_headlines()
    data = get_historical_prices_df()
    
    movers, scans = {}, {}
    if not data.empty:
        movers, scans = process_custom_scans_by_benchmark(data, benchmarks)
        
    return {
        "benchmarks": benchmarks,
        "market_movers": movers,
        "custom_scans": scans,
        "headlines": news
    }
