/*
 * ticker_prices.sql
 * 
 * This DDL script defines the 'ticker_prices' table which stores historical daily 
 * OHLCV price data for individual stock tickers, along with dividends and stock splits.
 */
CREATE TABLE ticker_prices (
    price_date TIMESTAMP,
    ticker_symbol VARCHAR2(10),
    close_price NUMBER,
    dividends NUMBER,
    high_price NUMBER,
    low_price NUMBER,
    open_price NUMBER,
    stock_splits NUMBER,
    volume NUMBER,
    FOREIGN KEY (ticker_symbol) REFERENCES tickers(ticker_symbol)
);
