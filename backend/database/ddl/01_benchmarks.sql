/*
 * benchmarks.sql
 * 
 * This DDL script defines the 'benchmarks' table which stores the main market indices 
 * (e.g., SPY, QQQ, DIA) along with their current price and daily change percentage.
 */
CREATE TABLE benchmarks (
    benchmark_symbol VARCHAR2(10) PRIMARY KEY,
    benchmark_name VARCHAR2(255),
    current_price NUMBER,
    change_pct NUMBER
);
