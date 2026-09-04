/*
 * tickers.sql
 * 
 * This DDL script defines the 'tickers' table which stores individual stock ticker information, 
 * including the company name, sector, and market cap.
 */
CREATE TABLE tickers (
    ticker_symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(255),
    market_cap FLOAT
);
