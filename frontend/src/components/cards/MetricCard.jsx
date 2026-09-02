import "./MetricCard.css";

/**
 * A minimal, reusable card for displaying a single metric label and its value.
 */
export default function MetricCard({ label, value, extraClass = "" }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${extraClass}`}>{value}</span>
    </div>
  );
}
