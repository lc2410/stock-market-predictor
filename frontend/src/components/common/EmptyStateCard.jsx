import "./EmptyStateCard.css";

// Simple placeholder card displayed when no data is available or a section is empty
export default function EmptyStateCard({ message, style = {} }) {
  return (
    <div className="metric-card empty-state-card" style={style}>
      {message}
    </div>
  );
}
