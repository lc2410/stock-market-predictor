import "./HorizonCard.css";
import { formatDate, formatMoney } from "../../utils/formatters";

/**
 * A card component displaying key data points and predictions 
 * (e.g., predicted direction, confidence, and target range).
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
  dirLabel = "Predicted Direction (vs. Last Recorded Price)",
}) {
  const isUp = direction === "Up";
  const dirClass = isUp ? "pill-up" : "pill-down";
  const arrowIcon = isUp ? "↑" : "↓";

  return (
    <div className="premium-horizon-card">
      <div className="horizon-header">
        <div className="horizon-header-title">
          <h4 className="horizon-title">{title}</h4>
        </div>
        <div className="horizon-date-badge">{formatDate(dateStr)}</div>
      </div>
      <div className="horizon-body">
        <div className="stat-box">
          <span className="stat-label">{dirLabel}</span>
          <div className="horizon-body-direction">
            <span className={`direction-pill ${dirClass}`}>
              {arrowIcon} {direction}
            </span>
            <span className="conf-badge">{dirConf}% Conf.</span>
          </div>
        </div>
        <div className="stat-divider" />
        <div className="stat-box">
          <span className="stat-label">{amtTitle}</span>
          <div className="horizon-body-range">
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
