/*
 * benchmark_prices.sql
 * 
 * This DDL script defines the 'benchmark_prices' table which stores historical daily 
 * OHLCV price data for the market benchmarks, linking back to the 'benchmarks' table.
 */
CREATE TABLE benchmark_prices (
    price_date TIMESTAMP,
    benchmark_symbol VARCHAR(10),
    close_price FLOAT,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    volume INTEGER,
    FOREIGN KEY (benchmark_symbol) REFERENCES benchmarks(benchmark_symbol)
);
