"""
tickers.py

This file contains DML (Data Manipulation Language) queries for the 'tickers' table.
It includes statements for deleting and inserting individual stock ticker metadata.
"""
DELETE_ALL_TICKERS = "DELETE FROM tickers"

INSERT_TICKER = """
INSERT OR REPLACE INTO tickers (ticker_symbol, company_name, sector, market_cap)
VALUES (?, ?, ?, ?)
"""
