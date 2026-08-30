/*
 * benchmarks.sql
 * 
 * This DDL script defines the 'benchmarks' table which stores the main market indices 
 * (e.g., SPY, QQQ, DIA) along with their current price and daily change percentage.
 */
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_symbol VARCHAR(10) PRIMARY KEY,
    benchmark_name VARCHAR(255),
    current_price FLOAT,
    change_pct FLOAT
);
