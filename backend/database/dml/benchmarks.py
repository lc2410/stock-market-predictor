"""
benchmarks.py

This file contains DML (Data Manipulation Language) queries for the 'benchmarks' table.
It includes statements for deleting, inserting, and selecting benchmark summary data.
"""
DELETE_ALL_BENCHMARKS = "DELETE FROM benchmarks"

INSERT_BENCHMARK = """
INSERT INTO benchmarks (
    benchmark_symbol, benchmark_name, current_price, change_pct
)
VALUES (:1, :2, :3, :4)
"""

SELECT_ALL_BENCHMARKS = "SELECT * FROM benchmarks"
