"""
benchmark_tickers.py

This file contains DML (Data Manipulation Language) queries for the 'benchmark_tickers' table.
It includes statements for managing the many-to-many relationship between benchmarks and individual stock tickers.
"""
DELETE_ALL_BENCHMARK_TICKERS = "DELETE FROM benchmark_tickers"

INSERT_BENCHMARK_TICKER = """
INSERT OR REPLACE INTO benchmark_tickers (benchmark_symbol, ticker_symbol, weight)
VALUES (?, ?, ?)
"""

SELECT_BENCHMARK_CONSTITUENTS = """
SELECT 
    t.ticker_symbol, 
    t.company_name, 
    t.sector, 
    t.market_cap, 
    bt.weight
FROM benchmark_tickers bt
JOIN tickers t ON bt.ticker_symbol = t.ticker_symbol
WHERE bt.benchmark_symbol = ?
"""
