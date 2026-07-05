/**
 * SentimentTab Component
 * ----------------------
 * Responsible for rendering the Market Sentiment analysis view.
 * It visualizes the overall AI Stock Grade, General Sentiment, and displays
 * the foundational metrics (like EPS, Market Cap, Beta) driving the score.
 * It also renders the NLP-processed news articles categorized into Positive and
 * Negative press, allowing users to open article summaries in a modal.
 */
import MetricCard from '../cards/MetricCard';

const ExportIcon = () => (
  <svg
    className="export-icon"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" y1="14" x2="21" y2="3" />
  </svg>
);

function NewsItem({ headline, onOpenModal }) {
  const safeTitle = headline.title || '';
  return (
    <li style={{ marginBottom: '8px', textAlign: 'left', lineHeight: '1.4' }}>
      <button
        className="news-item-link"
        style={{ textAlign: 'left' }}
        onClick={() =>
          onOpenModal({
            title: safeTitle,
            summary: headline.summary || 'No summary available.',
            publisher: headline.publisher || 'Unknown Publisher',
            url: headline.url || '#',
          })
        }
      >
        &ldquo;{safeTitle}&rdquo;
      </button>
      <a
        href={headline.url || '#'}
        target="_blank"
        rel="noopener noreferrer"
        className="news-export-inline"
        title="Read external article"
      >
        <ExportIcon />
      </a>
    </li>
  );
}

function ProgressBar({ pctValue, gradient = false }) {
  return (
    <div style={{ width: '100%', height: '6px', background: 'var(--outline-border)', borderRadius: '3px', overflow: 'hidden' }}>
      <div
        style={{
          width: `${pctValue}%`,
          height: '100%',
          background: gradient
            ? 'linear-gradient(90deg, var(--brand-primary), rgba(var(--brand-rgb), 0.7))'
            : 'var(--brand-primary)',
          borderRadius: '3px',
          transition: 'width 0.4s ease',
        }}
      />
    </div>
  );
}

function ReasoningSection({ reasoning, onOpenModal }) {
  if (!reasoning || typeof reasoning === 'string') {
    return (
      <span style={{ color: 'var(--text-muted)' }}>
        {reasoning || 'No recent data available.'}
      </span>
    );
  }

  return (
    <>
      {/* News sentiment */}
      {reasoning.news?.positive?.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <strong style={{ color: 'var(--brand-success)' }}>Positive Press:</strong>
          <ul style={{ marginTop: '4px', paddingLeft: '24px', listStyleType: 'disc' }}>
            {reasoning.news.positive.map((h, i) => (
              <NewsItem key={i} headline={h} onOpenModal={onOpenModal} />
            ))}
          </ul>
        </div>
      )}
      {reasoning.news?.negative?.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <strong style={{ color: 'var(--brand-danger)' }}>Negative Press:</strong>
          <ul style={{ marginTop: '4px', paddingLeft: '24px', listStyleType: 'disc' }}>
            {reasoning.news.negative.map((h, i) => (
              <NewsItem key={i} headline={h} onOpenModal={onOpenModal} />
            ))}
          </ul>
        </div>
      )}
      {reasoning.news?.neutral && (
        <div style={{ marginBottom: '12px' }}>
          <span style={{ color: 'var(--text-muted)' }}>{reasoning.news.neutral}</span>
        </div>
      )}

      {/* Fundamental catalysts */}
      {reasoning.fundamentals?.positive?.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <strong style={{ color: 'var(--brand-success)' }}>General Strengths:</strong>
          <ul style={{ marginTop: '6px', paddingLeft: '24px', listStyleType: 'disc' }}>
            {reasoning.fundamentals.positive.map((f, i) => (
              <li key={i} style={{ marginBottom: '6px' }}>{f}</li>
            ))}
          </ul>
        </div>
      )}
      {reasoning.fundamentals?.negative?.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <strong style={{ color: 'var(--brand-danger)' }}>General Risks:</strong>
          <ul style={{ marginTop: '6px', paddingLeft: '24px', listStyleType: 'disc' }}>
            {reasoning.fundamentals.negative.map((f, i) => (
              <li key={i} style={{ marginBottom: '6px' }}>{f}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ETF Holdings */}
      {reasoning.etf_holdings?.length > 0 && (
        <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(var(--brand-rgb), 0.15)' }}>
          <strong style={{ color: 'var(--text-main)', display: 'block', marginBottom: '14px' }}>
            Top 10 Fund Holdings by Weight:
          </strong>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {reasoning.etf_holdings.map((h, index) => {
              const nameDisplay = h.name !== h.symbol ? `${h.name} (${h.symbol})` : h.symbol;
              const pctValue = parseFloat(h.weight) || 0;
              return (
                <div key={index} style={{ marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px', fontSize: '13px' }}>
                    <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>
                      <span style={{ color: 'var(--text-muted)', marginRight: '4px' }}>{index + 1}.</span>
                      {nameDisplay}
                    </span>
                    <span style={{ color: 'var(--text-main)', fontWeight: 600, fontFamily: 'monospace' }}>
                      {h.weight || '0.00%'}
                    </span>
                  </div>
                  <ProgressBar pctValue={pctValue} />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ETF Sectors */}
      {reasoning.etf_sectors?.length > 0 && (
        <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(var(--brand-rgb), 0.15)' }}>
          <strong style={{ color: 'var(--text-main)', display: 'block', marginBottom: '14px' }}>
            Economic Sector Exposure:
          </strong>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {reasoning.etf_sectors.map((s, i) => {
              const pctValue = parseFloat(s.weight) || 0;
              return (
                <div key={i} style={{ marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px', fontSize: '13px' }}>
                    <span style={{ color: 'var(--text-main)', fontWeight: 500 }}>{s.sector}</span>
                    <span style={{ color: 'var(--text-main)', fontWeight: 600, fontFamily: 'monospace' }}>
                      {s.weight || '0.00%'}
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

export default function SentimentTab({ data, onOpenModal }) {
  const getGradeColor = (grade) =>
    grade.includes('A') || grade.includes('B') ? 'up' : grade.includes('D') || grade.includes('F') ? 'down' : '';
  const getSentimentColor = (sentiment) =>
    sentiment === 'Bullish' ? 'up' : sentiment === 'Bearish' ? 'down' : '';

  return (
    <div id="tab-sentiment" className="tab-content active">
      <div className="dashboard-grid" style={{ marginTop: '10px', marginBottom: '16px' }}>
        <MetricCard label="Grade" value={data.Stock_Grade} extraClass={getGradeColor(data.Stock_Grade)} />
        <MetricCard label="General Sentiment" value={data.News_Sentiment} extraClass={getSentimentColor(data.News_Sentiment)} />
      </div>
      <div
        style={{
          background: 'rgba(var(--brand-rgb), 0.05)',
          border: '1px solid rgba(var(--brand-rgb), 0.2)',
          borderLeft: '4px solid rgba(var(--brand-rgb), 1)',
          padding: '16px',
          borderRadius: '8px',
          marginBottom: '20px',
          fontSize: '14px',
          lineHeight: '1.6',
        }}
      >
        <strong style={{ color: 'var(--text-main)', display: 'block', marginBottom: '6px' }}>
          AI Sentiment Reasoning:
        </strong>
        <ReasoningSection reasoning={data.AI_Reasoning} onOpenModal={onOpenModal} />
      </div>
    </div>
  );
}
