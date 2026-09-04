/*
 * benchmark_prices.sql
 * 
 * This DDL script defines the 'benchmark_prices' table which stores historical daily 
 * OHLCV price data for the market benchmarks, linking back to the 'benchmarks' table.
 */
CREATE TABLE benchmark_prices (
    price_date TIMESTAMP,
    benchmark_symbol VARCHAR2(10),
    close_price NUMBER,
    open_price NUMBER,
    high_price NUMBER,
    low_price NUMBER,
    volume NUMBER,
    FOREIGN KEY (benchmark_symbol) REFERENCES benchmarks(benchmark_symbol)
);
