/**
 * Small reusable card for displaying a labeled metric value (e.g., grade, sentiment).
 */
export default function MetricCard({ label, value, extraClass = '' }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${extraClass}`}>{value}</span>
    </div>
  );
}
