"""
headlines.py

This file contains DML (Data Manipulation Language) queries for the 'headlines' table.
It includes statements for deleting, inserting, and selecting market news headlines.
"""
DELETE_ALL_HEADLINES = "DELETE FROM headlines"

INSERT_HEADLINE = """
INSERT INTO headlines (title, publisher, link, summary, published_at)
VALUES (:1, :2, :3, :4, :5)
"""

SELECT_ALL_HEADLINES = "SELECT title, publisher, link, summary, published_at FROM headlines ORDER BY id DESC"
