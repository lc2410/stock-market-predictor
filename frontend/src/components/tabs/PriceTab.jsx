import { useState, useCallback } from 'react';
import HorizonCard from '../cards/HorizonCard';
import PriceChart from '../charts/PriceChart';
import NavSlider from '../charts/NavSlider';
import DataTable from '../table/DataTable';
import { normalizeDate } from '../../utils/formatters';

function buildPriceRows(data) {
  const pMap = new Map();

  if (data.Chart_History?.dates?.length > 0) {
    data.Chart_History.dates.forEach((d, i) => {
      const k = normalizeDate(d);
      pMap.set(k, { date: k, hist: data.Chart_History.prices[i], proj: null, lower: null, upper: null });
    });
  }
  if (data.Train_Fit_Dates) {
    data.Train_Fit_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!pMap.has(k)) pMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
      const price = data.Train_Fit_Prices[i];
      if (price !== undefined && price !== null) pMap.get(k).proj = price;
    });
  }
  if (data.Chart_Future_Dates) {
    data.Chart_Future_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!pMap.has(k)) pMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
      pMap.get(k).proj = data.Chart_Future_Prices[i];
      pMap.get(k).lower = data.Chart_Future_Lower[i];
      pMap.get(k).upper = data.Chart_Future_Upper[i];
    });
  }

  return Array.from(pMap.values()).sort((a, b) => new Date(b.date) - new Date(a.date));
}

function buildViewState(data) {
  const allDates = [...data.Chart_History.dates, ...data.Chart_Future_Dates];
  const absMin = new Date(allDates[0]).getTime();
  let absMax = new Date(allDates[allDates.length - 1]).getTime();
  if (!data.Chart_Future_Dates || data.Chart_Future_Dates.length === 0) {
    absMax += 14 * 24 * 60 * 60 * 1000;
  }
  return { min: absMin, max: absMax, absoluteMin: absMin, absoluteMax: absMax };
}

export default function PriceTab({ data, theme }) {
  const [viewState, setViewState] = useState(() => buildViewState(data));

  const handleViewChange = useCallback(({ min, max }) => {
    setViewState((prev) => ({ ...prev, min, max }));
  }, []);

  const handleReset = useCallback(() => {
    setViewState((prev) => ({ ...prev, min: prev.absoluteMin, max: prev.absoluteMax }));
  }, []);

  const f = data.Price_Forecasts;
  const isCrypto = data.Chart_Future_Dates.length === 365;
  const fd = data.Chart_Future_Dates;
  const d1 = fd[0];
  const d2 = fd[isCrypto ? 6 : 4];
  const d3 = fd[isCrypto ? 29 : 20];
  const d4 = fd[isCrypto ? 89 : 62];
  const d5 = fd[isCrypto ? 179 : 125];
  const d6 = fd[isCrypto ? 269 : 188];
  const d7 = fd[fd.length - 1];

  const priceRows = buildPriceRows(data);

  return (
    <>
      {!f || Object.keys(f).length === 0 ? (
        <div className="metric-card" style={{ textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '20px', marginBottom: '32px' }}>
          Not enough closed stock price data to generate reliable forecasts.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px', marginBottom: '32px' }}>
          <HorizonCard title="Next-Day Metrics" dateStr={d1} direction={f.Next_Day.Direction} dirConf={f.Next_Day.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_Day.Amount} amtLower={f.Next_Day.Amount_Lower} amtUpper={f.Next_Day.Amount_Upper} />
          <HorizonCard title="Next-Week Metrics" dateStr={d2} direction={f.Next_Week.Direction} dirConf={f.Next_Week.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_Week.Amount} amtLower={f.Next_Week.Amount_Lower} amtUpper={f.Next_Week.Amount_Upper} />
          <HorizonCard title="Next-Month Metrics" dateStr={d3} direction={f.Next_Month.Direction} dirConf={f.Next_Month.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_Month.Amount} amtLower={f.Next_Month.Amount_Lower} amtUpper={f.Next_Month.Amount_Upper} />
          <HorizonCard title="Next-3-Months Metrics" dateStr={d4} direction={f.Next_3_Months.Direction} dirConf={f.Next_3_Months.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_3_Months.Amount} amtLower={f.Next_3_Months.Amount_Lower} amtUpper={f.Next_3_Months.Amount_Upper} />
          <HorizonCard title="Next-6-Months Metrics" dateStr={d5} direction={f.Next_6_Months.Direction} dirConf={f.Next_6_Months.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_6_Months.Amount} amtLower={f.Next_6_Months.Amount_Lower} amtUpper={f.Next_6_Months.Amount_Upper} />
          <HorizonCard title="Next-9-Months Metrics" dateStr={d6} direction={f.Next_9_Months.Direction} dirConf={f.Next_9_Months.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_9_Months.Amount} amtLower={f.Next_9_Months.Amount_Lower} amtUpper={f.Next_9_Months.Amount_Upper} />
          <HorizonCard title="Next-Year Metrics" dateStr={d7} direction={f.Next_Year.Direction} dirConf={f.Next_Year.Direction_Confidence} amtTitle="Forecasted Close" amt={f.Next_Year.Amount} amtLower={f.Next_Year.Amount_Lower} amtUpper={f.Next_Year.Amount_Upper} />
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

      <DataTable
        title="Closed Stock Price History & Forecast Data with Expected Range"
        dateHeader="Trading Date"
        histHeader="Historical Price"
        projHeader="Projected Price"
        rows={priceRows}
      />
    </>
  );
}
