import HorizonCard from '../cards/HorizonCard';
import DividendChart from '../charts/DividendChart';
import DataTable from '../table/DataTable';
import { normalizeDate } from '../../utils/formatters';

function buildDivRows(data) {
  const dMap = new Map();

  if (data.Chart_History?.dividend_dates?.length > 0) {
    data.Chart_History.dividend_dates.forEach((d, i) => {
      const k = normalizeDate(d);
      dMap.set(k, { date: k, hist: data.Chart_History.dividend_amounts[i], proj: null, lower: null, upper: null });
    });
  }
  if (data.Train_Fit_Div_Dates) {
    data.Train_Fit_Div_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!dMap.has(k)) dMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
      dMap.get(k).proj = data.Train_Fit_Div_Amounts[i];
    });
  }
  if (data.Div_Future_Dates) {
    data.Div_Future_Dates.forEach((d, i) => {
      const k = normalizeDate(d);
      if (!dMap.has(k)) dMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
      dMap.get(k).proj = data.Div_Future_Amounts[i];
      dMap.get(k).lower = data.Div_Future_Lower[i];
      dMap.get(k).upper = data.Div_Future_Upper[i];
    });
  }

  return Array.from(dMap.values()).sort((a, b) => new Date(b.date) - new Date(a.date));
}

export default function DividendTab({ data, theme }) {
  const hasDividends =
    data.Chart_History?.dividend_dates && data.Chart_History.dividend_dates.length > 0;

  const divRows = buildDivRows(data);
  const f = data.Div_Forecasts;
  const dates = data.Div_Future_Dates || [];
  const d1 = dates[0];
  const d2 = dates.length > 1 ? dates[1] : 'N/A';
  const d3 = dates.length > 2 ? dates[2] : 'N/A';
  const d4 = dates.length > 3 ? dates[3] : 'N/A';
  const d5 = dates.length > 4 ? dates[4] : 'N/A';

  const dirLabel = 'Predicted Direction (vs. Last Payout)';

  return (
    <>
      {!hasDividends ? (
        <div className="metric-card" style={{ textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '20px', marginBottom: '32px' }}>
          This publicly traded asset does not currently pay dividends.
        </div>
      ) : !f || Object.keys(f).length === 0 ? (
        <div className="metric-card" style={{ textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '20px', marginBottom: '32px' }}>
          Not enough historical dividend data to generate reliable forecasts.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px', marginBottom: '32px' }}>
          <HorizonCard title="Next Projected Payout" dateStr={d1} direction={f.Next_Payout.Direction} dirConf={f.Next_Payout.Direction_Confidence} amtTitle="Forecasted Payout" amt={f.Next_Payout.Amount} amtLower={f.Next_Payout.Amount_Lower} amtUpper={f.Next_Payout.Amount_Upper} dirLabel={dirLabel} />
          <HorizonCard title="2nd Projected Payout" dateStr={d2} direction={f.Payout_2.Direction} dirConf={f.Payout_2.Direction_Confidence} amtTitle="Forecasted Payout" amt={f.Payout_2.Amount} amtLower={f.Payout_2.Amount_Lower} amtUpper={f.Payout_2.Amount_Upper} dirLabel={dirLabel} />
          <HorizonCard title="3rd Projected Payout" dateStr={d3} direction={f.Payout_3.Direction} dirConf={f.Payout_3.Direction_Confidence} amtTitle="Forecasted Payout" amt={f.Payout_3.Amount} amtLower={f.Payout_3.Amount_Lower} amtUpper={f.Payout_3.Amount_Upper} dirLabel={dirLabel} />
          <HorizonCard title="4th Projected Payout" dateStr={d4} direction={f.Payout_4.Direction} dirConf={f.Payout_4.Direction_Confidence} amtTitle="Forecasted Payout" amt={f.Payout_4.Amount} amtLower={f.Payout_4.Amount_Lower} amtUpper={f.Payout_4.Amount_Upper} dirLabel={dirLabel} />
          <HorizonCard title="5th Projected Payout" dateStr={d5} direction={f.Payout_5.Direction} dirConf={f.Payout_5.Direction_Confidence} amtTitle="Forecasted Payout" amt={f.Payout_5.Amount} amtLower={f.Payout_5.Amount_Lower} amtUpper={f.Payout_5.Amount_Upper} dirLabel={dirLabel} />
        </div>
      )}

      {/* Dividend chart always renders the box — shows "NO DIVIDEND DATA" overlay if no dividends */}
      {hasDividends ? (
        <DividendChart data={data} theme={theme} />
      ) : (
        <div style={{ marginBottom: '24px' }}>
          <div className="chart-box" id="dividendChartBox" style={{ position: 'relative' }}>
            <div
              id="noDividendOverlay"
              style={{
                position: 'absolute',
                inset: 0,
                background: 'var(--card-bg)',
                backdropFilter: 'blur(4px)',
                borderRadius: '16px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  pointerEvents: 'none',
                }}
              >
                <span
                  style={{
                    fontSize: '36px',
                    fontWeight: 800,
                    color: 'var(--table-border)',
                    letterSpacing: '2px',
                    whiteSpace: 'nowrap',
                    transform: 'rotate(-15deg)',
                    fontFamily: 'Inter, sans-serif',
                    userSelect: 'none',
                  }}
                >
                  NO DIVIDEND DATA
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {hasDividends && (
        <DataTable
          title="Dividend Payout History & Forecast Data with Expected Range"
          dateHeader="Ex-Dividend Date"
          histHeader="Historical Payout"
          projHeader="Projected Payout"
          rows={divRows}
        />
      )}
    </>
  );
}
