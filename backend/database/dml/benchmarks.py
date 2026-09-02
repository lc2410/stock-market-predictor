"""
benchmarks.py

This file contains DML (Data Manipulation Language) queries for the 'benchmarks' table.
It includes statements for deleting, inserting, and selecting benchmark summary data.
"""
DELETE_ALL_BENCHMARKS = "DELETE FROM benchmarks"

INSERT_BENCHMARK = """
INSERT OR REPLACE INTO benchmarks (
    benchmark_symbol, benchmark_name, current_price, change_pct
)
VALUES (?, ?, ?, ?)
"""

SELECT_ALL_BENCHMARKS = "SELECT * FROM benchmarks"
