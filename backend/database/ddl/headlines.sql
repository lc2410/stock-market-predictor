/*
 * headlines.sql
 * 
 * This DDL script defines the 'headlines' table which stores cached market news articles, 
 * including titles, publishers, links, summaries, and publication dates.
 */
CREATE TABLE IF NOT EXISTS headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    publisher TEXT,
    link TEXT,
    summary TEXT,
    published_at TEXT
);
