import GenericTable from "../common/GenericTable";
import { getPredictorTableColumns } from "./predictorTableConfig";
import "./PredictorTable.css";

/**
 * Reusable table component for displaying historical and projected financial metrics.
 * Conditionally adjusts the title and columns if projection data is missing.
 */
export default function PredictorTable({
  title,
  subtitle,
  dateHeader,
  histHeader,
  projHeader,
  rows,
}) {
  if (!rows || !rows.length) return null;

  const hasProj = rows.some((r) => r.proj !== null && r.proj !== undefined);
  const finalTitle = hasProj
    ? title
    : title.includes("Price")
      ? "Closed Stock Price History"
      : "Dividend Payout History";

  const columns = getPredictorTableColumns(
    dateHeader,
    histHeader,
    projHeader,
    hasProj,
  );

  return (
    <div className="predictor-table-card">
      <div className="predictor-table-header">
        <div className="predictor-table-title">
          <h3>{finalTitle}</h3>
        </div>
        {subtitle && <div className="predictor-table-subtitle">{subtitle}</div>}
      </div>
      <GenericTable
        columns={columns}
        data={rows}
        tableClassName=""
        wrapperClassName="predictor-table-content data-table-scroll"
      />
    </div>
  );
}
