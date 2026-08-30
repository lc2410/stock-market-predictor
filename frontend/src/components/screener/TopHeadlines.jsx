import { Newspaper, Info } from "lucide-react";
import "./TopHeadlines.css";

/**
 * Sidebar component that displays a list of the latest news headlines.
 * Opens a detailed view/modal when a headline is clicked.
 */
export default function TopHeadlines({ headlines, onNewsClick }) {
  const recentDate =
    headlines && headlines.length > 0 && headlines[0].time
      ? new Date(headlines[0].time).toLocaleDateString()
      : new Date().toLocaleDateString();

  return (
    <aside className="screener-sidebar">
      <div className="screener-table-header screener-performance-header">
        <div className="screener-table-title">
          <Newspaper className="icon-neutral" size={24} />
          <h2>Top Headlines</h2>
          <span
            data-tooltip={`Latest News Headlines (as of ${recentDate})`}
            className="info-tooltip-container top-headlines-tooltip"
          >
            <Info size={16} />
          </span>
        </div>
      </div>
      <div className="headlines-card">
        <div className="headlines-content">
          {headlines &&
            headlines.map((news, idx) => (
              <div
                key={idx}
                className="headline-item"
                onClick={() => onNewsClick(news)}
              >
                <div className="headline-publisher">{news.publisher}</div>
                <a
                  href={news.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="headline-title"
                  onClick={(e) => e.preventDefault()}
                >
                  {news.title}
                </a>
                <div className="headline-time">
                  {news.time
                    ? new Date(news.time).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : ""}
                </div>
              </div>
            ))}
          {(!headlines || headlines.length === 0) && (
            <div className="no-data">No headlines available</div>
          )}
        </div>
      </div>
    </aside>
  );
}
