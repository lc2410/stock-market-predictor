import { Info } from "lucide-react";
import GenericTable from "../common/GenericTable";
import { formatDate } from "../../utils/formatters";
import "./ScreenerTable.css";

/**
 * Reusable table component for displaying various stock screener metrics.
 * Dynamically builds columns based on the specified 'type' (e.g. movers, volume, dividend).
 */
export default function ScreenerTable({
  title,
  description,
  data,
  type,
  orderBy,
  onTickerSearch,
}) {
  if (!data || data.length === 0) return null;

  const columns = [
    {
      header: "Ticker",
      key: "symbol",
      cellClassName: `symbol-cell ${orderBy === "symbol" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "symbol" ? "sorted-header" : "",
      render: (row) => (
        <span
          className="symbol-ticker clickable"
          onClick={() => onTickerSearch && onTickerSearch(row.symbol)}
        >
          {row.name ? `${row.name} (${row.symbol})` : row.symbol}
        </span>
      ),
    },
  ];

  const currentPriceCol = {
    header: "Current Day's Price",
    key: "price",
    cellClassName: `price-cell ${orderBy === "price" ? "sorted-column" : ""}`,
    headerClassName: orderBy === "price" ? "sorted-header" : "",
    render: (row) => `$${parseFloat(row.price).toFixed(2)}`,
  };

  if (type === "movers") {
    columns.push({
      header: "Previous Day's Price",
      key: "prev_price",
      cellClassName: `price-cell ${orderBy === "prev_price" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "prev_price" ? "sorted-header" : "",
      render: (row) => `$${parseFloat(row.prev_price).toFixed(2)}`,
    });
    columns.push(currentPriceCol);
    columns.push({
      header: "Percentage Change",
      key: "change",
      cellClassName: (row) =>
        `change-cell ${row.change >= 0 ? "positive" : "negative"} ${orderBy === "change" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "change" ? "sorted-header" : "",
      render: (row) => {
        const isPositive = row.change >= 0;
        return (
          <>
            {isPositive ? "+" : "-"}
            {Math.abs(row.change).toFixed(2)}%
          </>
        );
      },
    });
  }

  if (type === "volume") {
    columns.push(currentPriceCol);
    columns.push({
      header: "Current Volume",
      key: "volume",
      cellClassName: orderBy === "volume" ? "sorted-column" : "",
      headerClassName: orderBy === "volume" ? "sorted-header" : "",
      render: (row) => `${(row.volume / 1000000).toFixed(1)}M`,
    });
  }

  if (type === "dividend") {
    columns.push(currentPriceCol);
    columns.push({
      header: "Last Payout",
      key: "dividend_payout",
      cellClassName: `price-cell ${orderBy === "dividend_payout" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "dividend_payout" ? "sorted-header" : "",
      render: (row) => `$${parseFloat(row.dividend_payout).toFixed(2)}`,
    });
    columns.push({
      header: "Last Payout Date",
      key: "last_dividend_date",
      cellClassName: orderBy === "last_dividend_date" ? "sorted-column" : "",
      headerClassName: orderBy === "last_dividend_date" ? "sorted-header" : "",
      render: (row) =>
        row.last_dividend_date ? formatDate(row.last_dividend_date) : "N/A",
    });
    columns.push({
      header: "Yield",
      key: "dividend_yield",
      cellClassName: orderBy === "dividend_yield" ? "sorted-column" : "",
      headerClassName: orderBy === "dividend_yield" ? "sorted-header" : "",
      render: (row) => `${parseFloat(row.dividend_yield).toFixed(2)}%`,
    });
  }

  if (type === "new_high") {
    columns.push({
      header: "Previous 52-Week High Price",
      key: "prev_period_high",
      cellClassName: `price-cell ${orderBy === "prev_period_high" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "prev_period_high" ? "sorted-header" : "",
      render: (row) => `$${parseFloat(row.prev_period_high).toFixed(2)}`,
    });
    columns.push({
      header: "Previous 52-Week High Price Date",
      key: "prev_high_date",
      cellClassName: orderBy === "prev_high_date" ? "sorted-column" : "",
      headerClassName: orderBy === "prev_high_date" ? "sorted-header" : "",
      render: (row) =>
        row.prev_high_date ? formatDate(row.prev_high_date) : "N/A",
    });
    columns.push(currentPriceCol);
    columns.push({
      header: "Change",
      key: "breakout_high_pct",
      cellClassName: (row) =>
        `change-cell ${row.breakout_high_pct >= 0 ? "positive" : "negative"} ${orderBy === "breakout_high_pct" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "breakout_high_pct" ? "sorted-header" : "",
      render: (row) =>
        `${row.breakout_high_pct >= 0 ? "+" : "-"}${Math.abs(row.breakout_high_pct).toFixed(2)}%`,
    });
  }

  if (type === "new_low") {
    columns.push({
      header: "Previous 52-Week Low Price",
      key: "prev_period_low",
      cellClassName: `price-cell ${orderBy === "prev_period_low" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "prev_period_low" ? "sorted-header" : "",
      render: (row) => `$${parseFloat(row.prev_period_low).toFixed(2)}`,
    });
    columns.push({
      header: "Previous 52-Week Low Price Date",
      key: "prev_low_date",
      cellClassName: orderBy === "prev_low_date" ? "sorted-column" : "",
      headerClassName: orderBy === "prev_low_date" ? "sorted-header" : "",
      render: (row) =>
        row.prev_low_date ? formatDate(row.prev_low_date) : "N/A",
    });
    columns.push(currentPriceCol);
    columns.push({
      header: "Change",
      key: "breakout_low_pct",
      cellClassName: (row) =>
        `change-cell ${row.breakout_low_pct >= 0 ? "positive" : "negative"} ${orderBy === "breakout_low_pct" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "breakout_low_pct" ? "sorted-header" : "",
      render: (row) =>
        `${row.breakout_low_pct >= 0 ? "+" : "-"}${Math.abs(row.breakout_low_pct).toFixed(2)}%`,
    });
  }

  if (type === "volatility") {
    columns.push(currentPriceCol);
    columns.push({
      header: "Volatility",
      key: "volatility",
      cellClassName: orderBy === "volatility" ? "sorted-column" : "",
      headerClassName: orderBy === "volatility" ? "sorted-header" : "",
      render: (row) => `${parseFloat(row.volatility).toFixed(2)}%`,
    });
  }

  if (type === "overbought" || type === "oversold") {
    columns.push(currentPriceCol);
    columns.push({
      header: "RSI",
      key: "rsi",
      cellClassName: orderBy === "rsi" ? "sorted-column" : "",
      headerClassName: orderBy === "rsi" ? "sorted-header" : "",
      render: (row) => parseFloat(row.rsi).toFixed(1),
    });
  }

  if (type === "unusual_volume") {
    columns.push(currentPriceCol);
    columns.push({
      header: "Average 52-Week Volume",
      key: "avg_volume_52w",
      cellClassName: orderBy === "avg_volume_52w" ? "sorted-column" : "",
      headerClassName: orderBy === "avg_volume_52w" ? "sorted-header" : "",
      render: (row) => `${(row.avg_volume_52w / 1000000).toFixed(1)}M`,
    });
    columns.push({
      header: "Current Volume",
      key: "volume",
      cellClassName: orderBy === "volume" ? "sorted-column" : "",
      headerClassName: orderBy === "volume" ? "sorted-header" : "",
      render: (row) => `${(row.volume / 1000000).toFixed(1)}M`,
    });
    columns.push({
      header: "Change",
      key: "vol_change_pct",
      cellClassName: (row) =>
        `change-cell ${row.vol_change_pct >= 0 ? "positive" : "negative"} ${orderBy === "vol_change_pct" ? "sorted-column" : ""}`,
      headerClassName: orderBy === "vol_change_pct" ? "sorted-header" : "",
      render: (row) =>
        `${row.vol_change_pct >= 0 ? "+" : "-"}${Math.abs(row.vol_change_pct).toFixed(2)}%`,
    });
  }

  return (
    <div className="screener-table-card">
      <div className="screener-table-header">
        <div className="screener-table-title">
          <h3>{title}</h3>
          {description && (
            <div className="info-tooltip-container">
              <Info size={14} className="icon-neutral info-icon" />
              <div className="custom-tooltip">{description}</div>
            </div>
          )}
        </div>
      </div>
      <GenericTable
        columns={columns}
        data={data}
        tableClassName=""
        wrapperClassName="screener-table-content"
      />
    </div>
  );
}
