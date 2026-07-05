import { formatDate, formatMoney } from '../../utils/formatters';

/**
 * Large horizon forecast card for displaying prediction summaries.
 * Shows direction pill + confidence badge on the left, and forecasted amount +
 * expected range on the right.
 */
export default function HorizonCard({
  title,
  dateStr,
  direction,
  dirConf,
  amtTitle,
  amt,
  amtLower,
  amtUpper,
  dirLabel = 'Predicted Direction (vs. Last Recorded Price)',
}) {
  const isUp = direction === 'Up';
  const dirClass = isUp ? 'pill-up' : 'pill-down';
  const arrowIcon = isUp ? '↑' : '↓';

  return (
    <div className="premium-horizon-card">
      <div className="horizon-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h4 className="horizon-title">{title}</h4>
        </div>
        <div className="horizon-date-badge">{formatDate(dateStr)}</div>
      </div>
      <div className="horizon-body">
        <div className="stat-box">
          <span className="stat-label">{dirLabel}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
            <span className={`direction-pill ${dirClass}`}>
              {arrowIcon} {direction}
            </span>
            <span className="conf-badge">{dirConf}% Conf.</span>
          </div>
        </div>
        <div className="stat-divider" />
        <div className="stat-box">
          <span className="stat-label">{amtTitle}</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '4px' }}>
            <span className="stat-val">{formatMoney(amt)}</span>
            <span className="conf-badge">
              Range: {formatMoney(amtLower)} &ndash; {formatMoney(amtUpper)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
