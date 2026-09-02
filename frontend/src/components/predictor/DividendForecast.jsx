import HorizonCard from "../cards/HorizonCard";
import DividendChart from "../charts/DividendChart";
import PredictorTable from "../tables/PredictorTable";
import EmptyStateCard from "../common/EmptyStateCard";
import { normalizeDate } from "../../utils/formatters";
import "./DividendForecast.css";

// Combines historical and forecasted dividend data into a unified array for the table
function buildDivRows(data) {
  const dMap = new Map();

  if (data.Chart_History?.dividend_dates?.length > 0) {
    data.Chart_History.dividend_dates.forEach((d, i) => {
      const k = normalizeDate(d);
      dMap.set(k, {
        date: k,
        hist: data.Chart_History.dividend_amounts[i],
        proj: null,
        lower: null,
        upper: null,
      });
    });
  }
  if (data.Train_Fit_Div_Dates) {
    data.Train_Fit_Div_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!dMap.has(k))
        dMap.set(k, {
          date: k,
          hist: null,
          proj: null,
          lower: null,
          upper: null,
        });
      dMap.get(k).proj = data.Train_Fit_Div_Amounts[i];
    });
  }
  if (data.Div_Future_Dates) {
    data.Div_Future_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!dMap.has(k))
        dMap.set(k, {
          date: k,
          hist: null,
          proj: null,
          lower: null,
          upper: null,
        });
      dMap.get(k).proj = data.Div_Future_Amounts[i];
      dMap.get(k).lower = data.Div_Future_Lower[i];
      dMap.get(k).upper = data.Div_Future_Upper[i];
    });
  }

  return Array.from(dMap.values()).sort(
    (a, b) =>
      new Date(
        typeof b.date === "string" ? b.date.replaceAll("-", "/") : b.date,
      ) -
      new Date(typeof a.date === "string" ? a.date.replaceAll("-", "/") : a.date),
  );
}

function RecentPayoutSubtitle({ data }) {
  if (data.Chart_History?.dividend_amounts?.length > 0) {
    const divs = data.Chart_History.dividend_amounts;
    const latestDiv = divs.at(-1);
    let changeEl = null;
    if (divs.length > 1) {
      const prevDiv = divs.at(-2);
      const diff = latestDiv - prevDiv;
      const pct = prevDiv !== 0 ? (diff / prevDiv) * 100 : 0;
      const isPos = diff >= 0;
      const sign = isPos ? "+" : "";
      changeEl = (
        <span
          className={`benchmark-change ${isPos ? "positive" : "negative"}`}
        >
          {sign}
          {pct.toFixed(2)}%
        </span>
      );
    }
    return (
      <div className="recent-payout-container">
        <span>
          Most Recent Dividend Payout: ${latestDiv.toFixed(2)}
        </span>
        {changeEl}
      </div>
    );
  }
  return "N/A";
}

// Displays horizon forecast cards, dividend chart, and data table for the predicted asset
export default function DividendForecast({ data, theme }) {
  const hasDividends =
    data.Chart_History?.dividend_dates &&
    data.Chart_History.dividend_dates.length > 0;

  const divRows = buildDivRows(data);
  const f = data.Div_Forecasts;
  const dates = data.Div_Future_Dates || [];

  const targetDates = [
    dates[0],
    dates.length > 1 ? dates[1] : "N/A",
    dates.length > 2 ? dates[2] : "N/A",
    dates.length > 3 ? dates[3] : "N/A",
    dates.length > 4 ? dates[4] : "N/A",
  ];

  const dirLabel = "Predicted Direction (vs. Last Payout)";

  const horizonConfig =
    f && Object.keys(f).length > 0
      ? [
          {
            title: "Next Projected Payout",
            dateStr: targetDates[0],
            metrics: f.Next_Payout,
          },
          {
            title: "2nd Projected Payout",
            dateStr: targetDates[1],
            metrics: f.Payout_2,
          },
          {
            title: "3rd Projected Payout",
            dateStr: targetDates[2],
            metrics: f.Payout_3,
          },
          {
            title: "4th Projected Payout",
            dateStr: targetDates[3],
            metrics: f.Payout_4,
          },
          {
            title: "5th Projected Payout",
            dateStr: targetDates[4],
            metrics: f.Payout_5,
          },
        ]
      : [];

  let topContent;
  if (!hasDividends) {
    topContent = <EmptyStateCard message="This publicly traded asset does not currently pay dividends." />;
  } else if (!f || Object.keys(f).length === 0) {
    topContent = <EmptyStateCard message="Not enough historical dividend data to generate reliable forecasts." />;
  } else {
    topContent = (
      <div className="dividend-forecast-cards">
        {horizonConfig.map((c) => (
          <HorizonCard
            key={c.title}
            title={c.title}
            dateStr={c.dateStr}
            direction={c.metrics.Direction}
            dirConf={c.metrics.Direction_Confidence}
            amtTitle="Forecasted Payout"
            amt={c.metrics.Amount}
            amtLower={c.metrics.Amount_Lower}
            amtUpper={c.metrics.Amount_Upper}
            dirLabel={dirLabel}
          />
        ))}
      </div>
    );
  }

  return (
    <>
      {topContent}

      {hasDividends ? (
        <DividendChart data={data} theme={theme} />
      ) : (
        <div className="dividend-chart-container">
          <div className="chart-box dividend-chart-box" id="dividendChartBox">
            <div id="noDividendOverlay" className="no-dividend-overlay">
              <div className="no-dividend-overlay-inner">
                <span className="no-dividend-text">NO DIVIDEND DATA</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {hasDividends && (
        <PredictorTable
          title="Dividend Payout History & Forecast Data with Expected Range"
          subtitle={<RecentPayoutSubtitle data={data} />}
          dateHeader="Ex-Dividend Date"
          histHeader="Historical Payout"
          projHeader="Projected Payout"
          rows={divRows}
        />
      )}
    </>
  );
}
