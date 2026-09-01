import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from "chart.js";
import { formatDate } from "../../utils/formatters";
import GenericChart from "../common/GenericChart";
import "./DividendChart.css";

Chart.register(
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
);

function prepareChartData(data) {
  const map = new Map();
  const hist = data.Chart_History;
  
  if (hist?.dividend_dates?.length) {
    hist.dividend_dates.forEach((d, i) =>
      map.set(d, {
        histAmt: hist.dividend_amounts[i],
        projAmt: null,
        ciUpper: null,
        ciLower: null,
        est: false,
      }),
    );
  }

  if (data.Train_Fit_Div_Dates && data.Train_Fit_Div_Amounts) {
    data.Train_Fit_Div_Dates.forEach((d, i) => {
      if (map.has(d)) {
        map.get(d).projAmt = data.Train_Fit_Div_Amounts[i];
      } else {
        map.set(d, {
          histAmt: null,
          projAmt: data.Train_Fit_Div_Amounts[i],
          ciUpper: null,
          ciLower: null,
          est: false,
        });
      }
    });
  }

  (data.Div_Future_Dates || []).forEach((d, i) => {
    map.set(d, {
      histAmt: null,
      projAmt: data.Div_Future_Amounts[i],
      ciUpper: data.Div_Future_Upper[i],
      ciLower: data.Div_Future_Lower[i],
      est: true,
    });
  });

  return Array.from(map.entries()).sort(
    (a, b) =>
      new Date(typeof a[0] === "string" ? a[0].replaceAll("-", "/") : a[0]) -
      new Date(typeof b[0] === "string" ? b[0].replaceAll("-", "/") : b[0]),
  );
}

/**
 * Displays historical and projected dividend payouts using a composite bar chart.
 * Also renders confidence intervals (expected range) for future estimates.
 */
export default function DividendChart({ data, theme }) {
  if (!data || !data.Chart_History?.dividend_dates?.length) return null;

  const isDark = theme === "dark";
  const colors = {
    brandRGB: isDark ? "168, 85, 247" : "16, 185, 129",
    history: isDark ? "#ffffff" : "#000000",
    grid: isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)",
    text: isDark ? "rgba(255, 255, 255, 0.5)" : "rgba(0, 0, 0, 0.5)",
  };

  const sorted = prepareChartData(data);
  const finalLabels = sorted.map((i) =>
    i[1].est ? `${formatDate(i[0])} (Est.)` : formatDate(i[0]),
  );
  const histData = sorted.map((i) => i[1].histAmt);
  const projData = sorted.map((i) => i[1].projAmt);
  const ciUpper = sorted.map((i) => i[1].ciUpper);
  const ciLower = sorted.map((i) => i[1].ciLower);
  const floatingCIBounds = ciUpper.map((u, i) =>
    u !== null ? [ciLower[i], u] : null,
  );

  const config = {
    type: "bar",
    data: {
      labels: finalLabels,
      datasets: [
        {
          label: "Expected Range",
          data: floatingCIBounds,
          backgroundColor: `rgba(${colors.brandRGB}, 0.3)`,
          grouped: false,
          barPercentage: 0.8,
          categoryPercentage: 0.8,
          borderRadius: 4,
          borderSkipped: false,
          order: 3,
        },
        {
          label: "Historical Payout",
          data: histData,
          backgroundColor: colors.history,
          grouped: false,
          barPercentage: 0.8,
          categoryPercentage: 0.8,
          borderRadius: 4,
          order: 2,
        },
        {
          label: "Projected Payout",
          data: projData,
          backgroundColor: `rgba(${colors.brandRGB}, 0.8)`,
          grouped: false,
          barPercentage: 0.4,
          categoryPercentage: 0.8,
          borderRadius: 4,
          order: 1,
        },
      ],
    },
    options: {
      color: colors.text,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: colors.text,
            maxRotation: 45,
            minRotation: 45,
            font: { size: 10 },
          },
        },
        y: {
          grid: { color: colors.grid },
          ticks: {
            color: colors.text,
            font: { size: 11 },
            callback: (v) => `$${v.toFixed(2)}`,
          },
        },
      },
      plugins: {
        title: { display: false },
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const amount = ctx.parsed.y;
              if (amount === null && ctx.datasetIndex !== 0) return null;

              if (ctx.datasetIndex === 0) {
                const val = ctx.raw;
                if (val?.length === 2 && val[0] !== null && val[1] !== null) {
                  return `Expected Range: $${val[0].toFixed(2)} – $${val[1].toFixed(2)}`;
                }
                return null;
              }

              const isHistorical = ctx.datasetIndex === 1;
              const isProjected = ctx.datasetIndex === 2;
              if (isHistorical)
                return `Historical Dividend Payout: $${amount.toFixed(2)}`;
              if (isProjected) {
                return `Projected Dividend Payout: $${amount.toFixed(2)}`;
              }
            },
          },
        },
      },
    },
  };

  const hasProjected = projData.some((p) => p !== null);

  const chartTitle = (
    <h2>
      {hasProjected
        ? "Dividend Payout History & Forecast Trends with Expected Range"
        : "Dividend Payout History"}
    </h2>
  );

  let chartSubtitlePrice = null;
  if (data.Chart_History?.dividend_amounts?.length > 0) {
    const divs = data.Chart_History.dividend_amounts;
    const latestDiv = divs[divs.length - 1];
    let changeEl = null;
    if (divs.length > 1) {
      const prevDiv = divs[divs.length - 2];
      const diff = latestDiv - prevDiv;
      const pct = prevDiv !== 0 ? (diff / prevDiv) * 100 : 0;
      const isPos = diff >= 0;
      const sign = isPos ? "+" : "";
      changeEl = (
        <span className={`benchmark-change ${isPos ? "positive" : "negative"}`}>
          {sign}
          {pct.toFixed(2)}%
        </span>
      );
    }
    chartSubtitlePrice = (
      <div className="benchmark-price-row dividend-price-row">
        <span className="benchmark-price">
          Most Recent Dividend Payout: ${latestDiv.toFixed(2)}
        </span>
        {changeEl}
      </div>
    );
  }

  const chartLegend = hasProjected ? (
    <div className="dividend-legend-container">
      <div className="dividend-legend-item">
        <div className="legend-swatch-hist"></div>
        <span>Historical Payout</span>
      </div>
      {(data.Train_Fit_Div_Dates || data.Div_Future_Dates) && (
        <div className="dividend-legend-item">
          <div className="legend-swatch-proj"></div>
          <span>Projected Payout</span>
        </div>
      )}
      {data.Div_Future_Dates && data.Div_Future_Dates.length > 0 && (
        <div className="dividend-legend-item">
          <div className="legend-swatch-range"></div>
          <span>Expected Range</span>
        </div>
      )}
    </div>
  ) : (
    <div className="dividend-legend-container">
      <div className="dividend-legend-item">
        <div className="legend-swatch-hist"></div>
        <span>Historical Payout</span>
      </div>
    </div>
  );

  return (
    <div className="dividend-chart-container">
      <GenericChart
        config={config}
        updateTrigger={[data, theme]}
        className="chart-box"
        wrapperStyle={{ height: "500px" }}
        wrapperClassName="dividend-chart-wrapper"
        canvasId="dividendChart"
        chartTitle={chartTitle}
        chartSubtitlePrice={chartSubtitlePrice}
        chartLegend={chartLegend}
      />
    </div>
  );
}
