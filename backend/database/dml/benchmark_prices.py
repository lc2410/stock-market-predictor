"""
benchmark_prices.py

This file contains DML (Data Manipulation Language) queries for the 'benchmark_prices' table.
It includes statements for deleting, inserting, and selecting historical price data for market benchmarks.
"""
DELETE_ALL_BENCHMARK_PRICES = "DELETE FROM benchmark_prices"

INSERT_BENCHMARK_PRICE = """
INSERT INTO benchmark_prices (
    price_date, benchmark_symbol, close_price, open_price, high_price, low_price, volume
)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SELECT_BENCHMARK_PRICES = "SELECT * FROM benchmark_prices WHERE benchmark_symbol = ? ORDER BY price_date ASC"
