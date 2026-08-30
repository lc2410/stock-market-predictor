/*
 * ticker_prices.sql
 * 
 * This DDL script defines the 'ticker_prices' table which stores historical daily 
 * OHLCV price data for individual stock tickers, along with dividends and stock splits.
 */
CREATE TABLE IF NOT EXISTS ticker_prices (
    price_date TIMESTAMP,
    ticker_symbol VARCHAR(10),
    close_price FLOAT,
    dividends FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    open_price FLOAT,
    stock_splits FLOAT,
    volume INTEGER,
    FOREIGN KEY (ticker_symbol) REFERENCES tickers(ticker_symbol)
);
