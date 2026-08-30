/*
 * benchmark_tickers.sql
 * 
 * This DDL script defines the 'benchmark_tickers' junction table which maps individual 
 * stock tickers to their respective market benchmarks, along with their index weight.
 */
CREATE TABLE IF NOT EXISTS benchmark_tickers (
    benchmark_symbol VARCHAR(10),
    ticker_symbol VARCHAR(10),
    weight FLOAT,
    PRIMARY KEY (benchmark_symbol, ticker_symbol),
    FOREIGN KEY (benchmark_symbol) REFERENCES benchmarks(benchmark_symbol),
    FOREIGN KEY (ticker_symbol) REFERENCES tickers(ticker_symbol)
);
