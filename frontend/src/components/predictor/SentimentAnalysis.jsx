import MetricCard from "../cards/MetricCard";
import ProgressBar from "../common/ProgressBar";
import "./SentimentAnalysis.css";

import { ExternalLink } from "lucide-react";

/**
 * Renders individual news item links.
 * Opens a modal with article details or links directly to the external source.
 */
function NewsItem({ headline, onOpenModal }) {
  const safeTitle = headline.title || "";
  return (
    <li className="sentiment-news-item">
      <button
        className="news-item-link"
        onClick={() =>
          onOpenModal({
            title: safeTitle,
            summary: headline.summary || "No summary available.",
            publisher: headline.publisher || "Unknown Publisher",
            url: headline.url || "#",
          })
        }
      >
        &ldquo;{safeTitle}&rdquo;
      </button>
      <a
        href={headline.url || "#"}
        target="_blank"
        rel="noopener noreferrer"
        className="news-export-inline"
        title="Read external article"
      >
        <ExternalLink className="export-icon" />
      </a>
    </li>
  );
}

/**
 * Renders detailed AI reasoning based on news, fundamentals, and ETF data.
 * Conditionally displays sections only if data is available to keep the UI clean.
 */
function ReasoningSection({ reasoning, onOpenModal }) {
  if (!reasoning || typeof reasoning === "string") {
    return (
      <span className="sentiment-reasoning-empty">
        {reasoning || "No recent data available."}
      </span>
    );
  }

  return (
    <>
      {}
      {reasoning.news?.positive?.length > 0 && (
        <div className="sentiment-category">
          <strong className="text-success">Positive Press:</strong>
          <ul className="sentiment-list">
            {reasoning.news.positive.map((h, i) => (
              <NewsItem key={i} headline={h} onOpenModal={onOpenModal} />
            ))}
          </ul>
        </div>
      )}
      {reasoning.news?.negative?.length > 0 && (
        <div className="sentiment-category">
          <strong className="text-danger">Negative Press:</strong>
          <ul className="sentiment-list">
            {reasoning.news.negative.map((h, i) => (
              <NewsItem key={i} headline={h} onOpenModal={onOpenModal} />
            ))}
          </ul>
        </div>
      )}
      {reasoning.news?.neutral && (
        <div className="sentiment-category">
          <span className="text-muted">{reasoning.news.neutral}</span>
        </div>
      )}

      {}
      {reasoning.fundamentals?.positive?.length > 0 && (
        <div className="sentiment-category-mt">
          <strong className="text-success">General Strengths:</strong>
          <ul className="sentiment-list-mt">
            {reasoning.fundamentals.positive.map((f, i) => (
              <li key={i} className="sentiment-list-item">
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
      {reasoning.fundamentals?.negative?.length > 0 && (
        <div className="sentiment-category-mt">
          <strong className="text-danger">General Risks:</strong>
          <ul className="sentiment-list-mt">
            {reasoning.fundamentals.negative.map((f, i) => (
              <li key={i} className="sentiment-list-item">
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {}
      {reasoning.etf_holdings?.length > 0 && (
        <div className="etf-section">
          <strong className="etf-section-title">
            Top 10 Fund Holdings by Weight:
          </strong>
          <div className="etf-flex-col">
            {reasoning.etf_holdings.map((h, index) => {
              const nameDisplay =
                h.name !== h.symbol ? `${h.name} (${h.symbol})` : h.symbol;
              const pctValue = Number.parseFloat(h.weight) || 0;
              return (
                <div key={index} className="etf-item-container">
                  <div className="etf-item-header">
                    <span className="etf-item-name">
                      <span className="etf-item-index">{index + 1}.</span>
                      {nameDisplay}
                    </span>
                    <span className="etf-item-weight">
                      {h.weight || "0.00%"}
                    </span>
                  </div>
                  <ProgressBar pctValue={pctValue} />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {}
      {reasoning.etf_sectors?.length > 0 && (
        <div className="etf-section">
          <strong className="etf-section-title">
            Economic Sector Exposure:
          </strong>
          <div className="etf-flex-col">
            {reasoning.etf_sectors.map((s, i) => {
              const pctValue = Number.parseFloat(s.weight) || 0;
              return (
                <div key={i} className="etf-item-container">
                  <div className="etf-item-header">
                    <span className="etf-item-name">{s.sector}</span>
                    <span className="etf-item-weight">
                      {s.weight || "0.00%"}
                    </span>
                  </div>
                  <ProgressBar pctValue={pctValue} gradient />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}

/**
 * Main sentiment tab content wrapper.
 * Displays overall stock grade/sentiment metrics and the detailed AI reasoning.
 */
export default function SentimentTab({ data, onOpenModal }) {
  const getGradeColor = (grade) =>
    grade.includes("A") || grade.includes("B")
      ? "up"
      : grade.includes("D") || grade.includes("F")
        ? "down"
        : "";
  const getSentimentColor = (sentiment) =>
    sentiment === "Bullish" ? "up" : sentiment === "Bearish" ? "down" : "";

  return (
    <div id="tab-sentiment" className="tab-content active">
      <div className="metrics-grid sentiment-metrics-grid">
        <MetricCard
          label="Grade"
          value={data.Stock_Grade}
          extraClass={getGradeColor(data.Stock_Grade)}
        />
        <MetricCard
          label="General Sentiment"
          value={data.News_Sentiment}
          extraClass={getSentimentColor(data.News_Sentiment)}
        />
      </div>
      <div className="sentiment-reasoning-container">
        <strong className="sentiment-reasoning-title">
          AI Sentiment Reasoning:
        </strong>
        <ReasoningSection
          reasoning={data.AI_Reasoning}
          onOpenModal={onOpenModal}
        />
      </div>
    </div>
  );
}
