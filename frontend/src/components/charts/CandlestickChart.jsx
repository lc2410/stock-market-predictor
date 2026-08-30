import { Chart, registerables } from "chart.js";
import "chartjs-adapter-date-fns";
import {
  CandlestickController,
  CandlestickElement,
} from "chartjs-chart-financial";
import GenericChart from "../common/GenericChart";
import "./CandlestickChart.css";

Chart.register(...registerables, CandlestickController, CandlestickElement);

/**
 * Renders a Candlestick chart for historical stock prices (Open, High, Low, Close)
 * using Chart.js financial controllers.
 */
export default function CandlestickChart({
  data,
  theme = "dark",
  chartTitle,
  chartSubtitlePrice,
  viewState,
}) {
  if (!data || !data.open || !data.high || !data.low) return null;

  const isDark = theme === "dark";

  const posColor = isDark ? "#059669" : "#34d399";
  const negColor = isDark ? "#dc2626" : "#f87171";
  const gridColor = isDark
    ? "rgba(255, 255, 255, 0.05)"
    : "rgba(0, 0, 0, 0.05)";
  const textColor = isDark ? "rgba(255, 255, 255, 0.5)" : "rgba(0, 0, 0, 0.5)";

  // Format the raw API history data into the OHLC format required by Chart.js
  const candleData = data.history.map((close, i) => ({
    x: new Date(data.dates[i].replace(/-/g, "/")).valueOf(),
    o: data.open[i],
    h: data.high[i],
    l: data.low[i],
    c: close,
  }));

  const config = {
    type: "candlestick",
    data: {
      datasets: [
        {
          data: candleData,
          backgroundColors: { up: posColor, down: negColor, unchanged: "#999" },
          borderColors: { up: posColor, down: negColor, unchanged: "#999" },
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 10, bottom: 10, left: 0, right: 10 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          displayColors: false,
          callbacks: {
            title: (items) =>
              new Date(items[0].raw.x).toLocaleDateString(undefined, {
                year: "numeric",
                month: "short",
                day: "numeric",
              }),
            label: (item) => {
              const ohlc = item.raw;
              return [
                `Open: $${ohlc.o.toFixed(2)}`,
                `Closed: $${ohlc.c.toFixed(2)}`,
                "──────────────",
                `High: $${ohlc.h.toFixed(2)}`,
                `Low: $${ohlc.l.toFixed(2)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "month", tooltipFormat: "MMM d, yyyy" },
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            autoSkip: false,
            maxRotation: 45,
            minRotation: 45,
            font: { size: 11 },
          },
          min: viewState?.min,
          max: viewState?.max,
        },
        y: {
          display: true,
          position: "left",
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            callback: (value) =>
              "$" +
              value.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              }),
          },
          min: Math.min(...data.low) * 0.995,
          max: Math.max(...data.high) * 1.005,
        },
      },
      animation: { duration: 0 },
    },
  };

  const chartLegend = (
    <div className="candlestick-legend-container">
      <div className="candlestick-legend-item">
        <div
          className="candlestick-swatch-box"
          style={{ backgroundColor: posColor }}
        ></div>
        <span>
          <strong>Green Body:</strong> Close &gt; Open
        </span>
      </div>
      <div className="candlestick-legend-item">
        <div
          className="candlestick-swatch-box"
          style={{ backgroundColor: negColor }}
        ></div>
        <span>
          <strong>Red Body:</strong> Open &gt; Close
        </span>
      </div>
      <div className="candlestick-legend-item">
        <div
          className="candlestick-swatch-line"
          style={{ backgroundColor: textColor }}
        ></div>
        <span>
          <strong>Thin Lines:</strong> High &amp; Low
        </span>
      </div>
    </div>
  );

  return (
    <GenericChart
      config={config}
      updateTrigger={[data, theme, viewState]}
      className="candlestick-chart-wrapper"
      chartTitle={chartTitle}
      chartSubtitlePrice={chartSubtitlePrice}
      chartLegend={chartLegend}
    />
  );
}
