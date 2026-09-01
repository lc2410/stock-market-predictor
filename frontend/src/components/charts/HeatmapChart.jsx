import { Chart, registerables } from "chart.js";
import { TreemapController, TreemapElement } from "chartjs-chart-treemap";
import GenericChart from "../common/GenericChart";
import "./HeatmapChart.css";

Chart.register(...registerables, TreemapController, TreemapElement);

const getSymFromCtx = (raw) => {
  if (!raw) return null;
  return raw.g || (raw._data && Array.isArray(raw._data) ? raw._data[0].symbol : (raw._data ? raw._data.symbol : null));
};

const getBlockColorFn = (theme, groupBySector, symbolMap) => (ctx) => {
  if (ctx.type !== "data" || !ctx.raw) return "transparent";
  if (groupBySector && ctx.raw && ctx.raw.l === 0) {
    return theme === "light" ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)";
  }
  const sym = getSymFromCtx(ctx.raw);
  const dataObj = symbolMap[sym];
  if (!dataObj || dataObj.change === undefined) return "#444";
  const chg = dataObj.change;
  if (theme === "light") {
    if (chg <= -2.5) return "#ef4444";
    if (chg <= -1.5) return "#f87171";
    if (chg < 0) return "#fda4af";
    if (chg === 0) return "#cbd5e1";
    if (chg < 1.5) return "#86efac";
    if (chg < 2.5) return "#4ade80";
    return "#22c55e";
  } else {
    if (chg <= -2.5) return "#881337";
    if (chg <= -1.5) return "#be123c";
    if (chg < 0) return "#e11d48";
    if (chg === 0) return "#475569";
    if (chg < 1.5) return "#15803d";
    if (chg < 2.5) return "#166534";
    return "#14532d";
  }
};

const getLabelsFormatterFn = (symbolMap) => (ctx) => {
  if (!ctx.raw) return "";
  if (ctx.raw.w !== undefined && ctx.raw.h !== undefined) {
    if (ctx.raw.w < 18 || ctx.raw.h < 12) return "";
  }
  const sym = getSymFromCtx(ctx.raw);
  const dataObj = symbolMap[sym];
  if (!dataObj) return "";
  const lines = [dataObj.symbol];
  if (ctx.raw.h !== undefined && ctx.raw.h < 28) {
    return lines;
  }
  if (dataObj.change !== undefined) {
    const prefix = dataObj.change > 0 ? "+" : "";
    lines.push(`${prefix}${dataObj.change.toFixed(2)}%`);
  }
  return lines;
};

/**
 * Renders a treemap heatmap visualizing market constituents, optionally grouped by sector. 
 * Block sizes represent index weight/market cap, and colors indicate daily return performance.
 */
export default function HeatmapChart({
  data,
  groupBySector = false,
  theme = "dark",
  chartTitle,
  chartSubtitlePrice,
}) {
  if (!data?.constituents?.length) return null;

  const symbolMap = {};
  const treemapData = data.constituents.map((c) => {
    const item = {
      symbol: c.symbol,
      name: c.name || c.symbol,
      sector: c.sector || "Unknown",
      change: c.change,
      marketCap: c.marketCap || 0,
      weight: c.weight || 0,
      v: c.weight + 0.2,
    };
    symbolMap[c.symbol] = item;
    return item;
  });

  const getBlockColor = getBlockColorFn(theme, groupBySector, symbolMap);
  const getLabelsFormatter = getLabelsFormatterFn(symbolMap);

  const config = {
    type: "treemap",
    data: {
      datasets: [
        {
          tree: treemapData,
          key: "v",
          groups: groupBySector ? ["sector", "symbol"] : ["symbol"],
          backgroundColor: getBlockColor,
          hoverBackgroundColor: getBlockColor,
          borderColor: (ctx) => {
            if (groupBySector && ctx.raw && ctx.raw.l === 0) {
              return theme === "light"
                ? "rgba(0,0,0,0.5)"
                : "rgba(255,255,255,0.2)";
            }
            return "rgba(0,0,0,0.5)";
          },
          borderWidth: (ctx) => {
            if (groupBySector && ctx.raw && ctx.raw.l === 0) return 2;
            return 1;
          },
          hoverBorderWidth: 3,
          hoverBorderColor: theme === "light" ? "#0f172a" : "white",
          spacing: groupBySector ? 1 : 0,
          captions: groupBySector
            ? {
                display: true,
                color: theme === "light" ? "#0f172a" : "white",
                font: { size: 12, weight: "bold", family: "sans-serif" },
                padding: 4,
              }
            : false,
          labels: {
            display: true,
            formatter: getLabelsFormatter,
            color: theme === "light" ? "#0f172a" : "white",
            font: (ctx) => {
              if (!ctx.raw || ctx.raw.w === undefined)
                return { size: 10, weight: "bold" };
              let s = 11;
              if (ctx.raw.w < 30 || ctx.raw.h < 25) s = 7;
              else if (ctx.raw.w < 45 || ctx.raw.h < 35) s = 9;
              return { size: s, weight: "bold", family: "sans-serif" };
            },
          },
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: () => "",
            label: (item) => {
              if (!item.raw) return "";

              if (groupBySector && item.raw.l === 0) {
                const sectorName = item.raw.g || "Unknown Sector";
                let totalWeight = 0;
                Object.values(symbolMap).forEach((d) => {
                  if (d.sector === sectorName) {
                    totalWeight += d.weight || 0;
                  }
                });
                return [
                  sectorName,
                  `Total Index Weight: ${totalWeight.toFixed(2)}%`,
                ];
              }

              const sym =
                item.raw.g ||
                (item.raw._data && Array.isArray(item.raw._data)
                  ? item.raw._data[0].symbol
                  : item.raw._data
                    ? item.raw._data.symbol
                    : null);
              const dataObj = symbolMap[sym];
              if (!dataObj || dataObj.change === undefined) return "";

              let mcapStr = "N/A";
              if (dataObj.marketCap) {
                const mcap = dataObj.marketCap;
                if (mcap >= 1e12) mcapStr = (mcap / 1e12).toFixed(3) + "T";
                else if (mcap >= 1e9) mcapStr = (mcap / 1e9).toFixed(3) + "B";
                else if (mcap >= 1e6) mcapStr = (mcap / 1e6).toFixed(3) + "M";
                else mcapStr = mcap.toLocaleString();
              }

              const prefix = dataObj.change > 0 ? "+" : "";
              const lines = [
                `${dataObj.name} - ${dataObj.symbol}`,
                `Day Return: ${prefix}${dataObj.change.toFixed(2)}%`,
                `Market Cap: ${mcapStr}`,
                `Index Weight: ${dataObj.weight.toFixed(4)}%`,
              ];

              if (groupBySector) {
                lines.unshift("────────────────────────");
              }

              return lines;
            },
          },
          titleFont: { size: 14 },
          bodyFont: { size: 13 },
          displayColors: false,
          padding: 10,
        },
      },
      animation: { duration: 0 },
    },
  };

  const chartLegend = (
    <div className="global-heatmap-legend">
      <div className="heatmap-legend-container heatmap-legend-wrapper">
        <div className="heatmap-legend-bar-inner">
          <div className="heatmap-legend-segment-inner heatmap-c1"></div>
          <div className="heatmap-legend-segment-inner heatmap-c2"></div>
          <div className="heatmap-legend-segment-inner heatmap-c3"></div>
          <div className="heatmap-legend-segment-inner heatmap-c4"></div>
          <div className="heatmap-legend-segment-inner heatmap-c5"></div>
          <div className="heatmap-legend-segment-inner heatmap-c6"></div>
          <div className="heatmap-legend-segment-inner heatmap-c7"></div>
        </div>
        <div className="heatmap-legend-labels-inner">
          <div className="heatmap-legend-label">&lt;= -3%</div>
          <div className="heatmap-legend-label">-2%</div>
          <div className="heatmap-legend-label">-1%</div>
          <div className="heatmap-legend-label">0%</div>
          <div className="heatmap-legend-label">1%</div>
          <div className="heatmap-legend-label">2%</div>
          <div className="heatmap-legend-label">&gt;= 3%</div>
        </div>
      </div>
    </div>
  );

  return (
    <GenericChart
      config={config}
      updateTrigger={[data, groupBySector, theme]}
      className="heatmap-chart-wrapper"
      chartTitle={chartTitle}
      chartSubtitlePrice={chartSubtitlePrice}
      chartLegend={chartLegend}
    />
  );
}
