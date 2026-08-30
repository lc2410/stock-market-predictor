import { useState, useCallback, useEffect } from "react";
import HorizonCard from "../cards/HorizonCard";
import PriceChart from "../charts/PriceChart";
import NavSlider from "../charts/NavSlider";
import PredictorTable from "../tables/PredictorTable";
import EmptyStateCard from "../common/EmptyStateCard";
import { normalizeDate } from "../../utils/formatters";
import "./PriceForecast.css";

// Combines historical and forecasted price data into a unified array for the table
function buildPriceRows(data) {
  const pMap = new Map();

  if (data.Chart_History?.dates?.length > 0) {
    data.Chart_History.dates.forEach((d, i) => {
      const k = normalizeDate(d);
      pMap.set(k, {
        date: k,
        hist: data.Chart_History.prices[i],
        proj: null,
        lower: null,
        upper: null,
      });
    });
  }
  if (data.Train_Fit_Dates) {
    data.Train_Fit_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!pMap.has(k))
        pMap.set(k, {
          date: k,
          hist: null,
          proj: null,
          lower: null,
          upper: null,
        });
      const price = data.Train_Fit_Prices[i];
      if (price !== undefined && price !== null) pMap.get(k).proj = price;
    });
  }
  if (data.Chart_Future_Dates) {
    data.Chart_Future_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!pMap.has(k))
        pMap.set(k, {
          date: k,
          hist: null,
          proj: null,
          lower: null,
          upper: null,
        });
      pMap.get(k).proj = data.Chart_Future_Prices[i];
      pMap.get(k).lower = data.Chart_Future_Lower[i];
      pMap.get(k).upper = data.Chart_Future_Upper[i];
    });
  }

  return Array.from(pMap.values()).sort(
    (a, b) =>
      new Date(
        typeof b.date === "string" ? b.date.replace(/-/g, "/") : b.date,
      ) -
      new Date(typeof a.date === "string" ? a.date.replace(/-/g, "/") : a.date),
  );
}

// Determines initial chart view bounds based on mobile view and available data range
function buildViewState(data) {
  const allDates = [...data.Chart_History.dates, ...data.Chart_Future_Dates];
  const absMin = new Date(
    typeof allDates[0] === "string"
      ? allDates[0].replace(/-/g, "/")
      : allDates[0],
  ).getTime();
  let absMax = new Date(
    typeof allDates[allDates.length - 1] === "string"
      ? allDates[allDates.length - 1].replace(/-/g, "/")
      : allDates[allDates.length - 1],
  ).getTime();
  if (!data.Chart_Future_Dates || data.Chart_Future_Dates.length === 0) {
    absMax += 14 * 24 * 60 * 60 * 1000;
  }

  const isMobile = typeof window !== "undefined" && window.innerWidth <= 768;
  const defaultDays = isMobile ? 180 : 365;
  const todayStr =
    data.Chart_History.dates[data.Chart_History.dates.length - 1];
  const todayTime = new Date(
    typeof todayStr === "string" ? todayStr.replace(/-/g, "/") : todayStr,
  ).getTime();

  const initialMin = Math.max(absMin, todayTime - defaultDays * 86400000);
  const initialMax = Math.min(absMax, todayTime + defaultDays * 86400000);

  return {
    min: initialMin,
    max: initialMax,
    absoluteMin: absMin,
    absoluteMax: absMax,
    activeRange: isMobile ? "6M" : "1Y",
  };
}

// Displays horizon forecast cards, interactive price chart, and data table for the predicted asset
export default function PriceForecast({ data, theme }) {
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" && window.innerWidth <= 768,
  );
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      if (mobile !== isMobile) {
        setIsMobile(mobile);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isMobile]);
  const [viewState, setViewState] = useState(() => buildViewState(data));

  const handleViewChange = useCallback((newView) => {
    setViewState((prev) => ({ ...prev, ...newView }));
  }, []);

  const handleReset = useCallback(() => {
    const isMobile = typeof window !== "undefined" && window.innerWidth <= 768;
    const defaultDays = isMobile ? 180 : 365;
    const todayStr =
      data.Chart_History.dates[data.Chart_History.dates.length - 1];
    const todayTime = new Date(
      typeof todayStr === "string" ? todayStr.replace(/-/g, "/") : todayStr,
    ).getTime();

    setViewState((prev) => {
      const initialMin = Math.max(
        prev.absoluteMin,
        todayTime - defaultDays * 86400000,
      );
      const initialMax = Math.min(
        prev.absoluteMax,
        todayTime + defaultDays * 86400000,
      );
      return {
        ...prev,
        min: initialMin,
        max: initialMax,
        activeRange: isMobile ? "6M" : "1Y",
      };
    });
  }, [data]);

  useEffect(() => {
    handleReset();
  }, [isMobile, handleReset]);

  const f = data.Price_Forecasts;
  const isCrypto = data.Chart_Future_Dates.length === 365;
  const fd = data.Chart_Future_Dates;

  const targetIndices = [
    0,
    isCrypto ? 6 : 4,
    isCrypto ? 29 : 20,
    isCrypto ? 89 : 62,
    isCrypto ? 179 : 125,
    isCrypto ? 269 : 188,
    fd.length - 1,
  ];
  const targetDates = targetIndices.map((i) => fd[i]);

  const priceRows = buildPriceRows(data);

  const horizonConfig =
    f && Object.keys(f).length > 0
      ? [
          {
            title: "Next-Day Metrics",
            dateStr: targetDates[0],
            metrics: f.Next_Day,
          },
          {
            title: "Next-Week Metrics",
            dateStr: targetDates[1],
            metrics: f.Next_Week,
          },
          {
            title: "Next-Month Metrics",
            dateStr: targetDates[2],
            metrics: f.Next_Month,
          },
          {
            title: "Next-3-Months Metrics",
            dateStr: targetDates[3],
            metrics: f.Next_3_Months,
          },
          {
            title: "Next-6-Months Metrics",
            dateStr: targetDates[4],
            metrics: f.Next_6_Months,
          },
          {
            title: "Next-9-Months Metrics",
            dateStr: targetDates[5],
            metrics: f.Next_9_Months,
          },
          {
            title: "Next-Year Metrics",
            dateStr: targetDates[6],
            metrics: f.Next_Year,
          },
        ]
      : [];

  return (
    <>
      {!f || Object.keys(f).length === 0 ? (
        <EmptyStateCard message="Not enough closed stock price data to generate reliable forecasts." />
      ) : (
        <div className="price-forecast-cards">
          {horizonConfig.map((c, i) => (
            <HorizonCard
              key={i}
              title={c.title}
              dateStr={c.dateStr}
              direction={c.metrics.Direction}
              dirConf={c.metrics.Direction_Confidence}
              amtTitle="Forecasted Close"
              amt={c.metrics.Amount}
              amtLower={c.metrics.Amount_Lower}
              amtUpper={c.metrics.Amount_Upper}
            />
          ))}
        </div>
      )}

      {data.Chart_History && (
        <>
          <PriceChart data={data} theme={theme} viewState={viewState} />
          <NavSlider
            data={data}
            theme={theme}
            viewState={viewState}
            onViewChange={handleViewChange}
            onReset={handleReset}
          />
        </>
      )}

      <PredictorTable
        title="Closed Stock Price History & Forecast Data with Expected Range"
        subtitle={(() => {
          if (data.Chart_History?.prices?.length > 0) {
            const prices = data.Chart_History.prices;
            const latestPrice = prices[prices.length - 1];
            let changeEl = null;
            if (prices.length > 1) {
              const prevPrice = prices[prices.length - 2];
              const diff = latestPrice - prevPrice;
              const pct = (diff / prevPrice) * 100;
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
              <div className="recent-price-container">
                <span>Most Recent Closed Price: ${latestPrice.toFixed(2)}</span>
                {changeEl}
              </div>
            );
          }
          return "N/A";
        })()}
        dateHeader="Trading Date"
        histHeader="Historical Price"
        projHeader="Projected Price"
        rows={priceRows}
      />
    </>
  );
}
