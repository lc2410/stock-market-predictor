import { useEffect, useRef } from 'react';
import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';
import { getThemeColors, formatDate } from '../../utils/formatters';

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

/**
 * Dividend payout history + forecast bar chart.
 */
export default function DividendChart({ data, theme }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const hist = data.Chart_History;
    if (!hist?.dividend_dates?.length) return;

    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }

    const colors = getThemeColors();
    const map = new Map();

    hist.dividend_dates.forEach((d, i) =>
      map.set(d, {
        histAmt: hist.dividend_amounts[i],
        projAmt: null,
        ciUpper: null,
        ciLower: null,
        est: false,
      })
    );

    if (data.Train_Fit_Div_Dates && data.Train_Fit_Div_Amounts) {
      data.Train_Fit_Div_Dates.forEach((d, i) => {
        if (map.has(d)) {
          map.get(d).projAmt = data.Train_Fit_Div_Amounts[i];
        } else {
          map.set(d, { histAmt: null, projAmt: data.Train_Fit_Div_Amounts[i], ciUpper: null, ciLower: null, est: false });
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

    const sorted = Array.from(map.entries()).sort((a, b) => new Date(a[0]) - new Date(b[0]));
    const finalLabels = sorted.map((i) =>
      i[1].est ? `${formatDate(i[0])} (Est.)` : formatDate(i[0])
    );
    const histData = sorted.map((i) => i[1].histAmt);
    const projData = sorted.map((i) => i[1].projAmt);
    const ciUpper = sorted.map((i) => i[1].ciUpper);
    const ciLower = sorted.map((i) => i[1].ciLower);
    const floatingCIBounds = ciUpper.map((u, i) => (u !== null ? [ciLower[i], u] : null));

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: finalLabels,
        datasets: [
          {
            label: 'Expected Range',
            data: floatingCIBounds,
            backgroundColor: `rgba(${colors.brandRGB}, 0.15)`,
            grouped: false,
            barPercentage: 0.8,
            categoryPercentage: 0.8,
            borderRadius: 4,
            borderSkipped: false,
            order: 3,
          },
          {
            label: 'Historical Payout',
            data: histData,
            backgroundColor: colors.history,
            grouped: false,
            barPercentage: 0.8,
            categoryPercentage: 0.8,
            borderRadius: 4,
            order: 2,
          },
          {
            label: 'Projected Payout',
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
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: colors.text, maxRotation: 45, minRotation: 45, font: { size: 10 } },
          },
          y: {
            grid: { color: colors.grid },
            ticks: { color: colors.text, font: { size: 11 }, callback: (v) => `$${v.toFixed(2)}` },
          },
        },
        plugins: {
          title: {
            display: true,
            text: projData.some((p) => p !== null)
              ? 'Dividend Payout History & Forecast Trends with Expected Range'
              : 'Dividend Payout History',
            color: colors.text,
            font: { size: 14, weight: '600' },
            padding: { bottom: 16 },
          },
          legend: {
            display: projData.some((p) => p !== null),
            labels: {
              color: colors.text,
              usePointStyle: true,
              generateLabels: () => {
                const items = [
                  { text: 'Historical Payout', fillStyle: colors.history, strokeStyle: 'transparent', fontColor: colors.text },
                ];
                if (data.Train_Fit_Div_Dates || data.Div_Future_Dates) {
                  items.push({
                    text: 'Projected Payout',
                    fillStyle: `rgba(${colors.brandRGB}, 0.8)`,
                    strokeStyle: 'transparent',
                    fontColor: colors.text,
                  });
                }
                if (data.Div_Future_Dates && data.Div_Future_Dates.length) {
                  items.push({
                    text: 'Expected Range',
                    fillStyle: `rgba(${colors.brandRGB}, 0.15)`,
                    strokeStyle: 'transparent',
                    fontColor: colors.text,
                  });
                }
                return items;
              },
            },
          },
          tooltip: {
            filter: (tooltipItem) => tooltipItem.datasetIndex !== 0,
            callbacks: {
              label: (ctx) => {
                const amount = ctx.parsed.y;
                if (amount === null) return null;
                const i = ctx.dataIndex;
                const isHistorical = ctx.datasetIndex === 1;
                const isProjected = ctx.datasetIndex === 2;
                if (isHistorical) return `Historical Dividend Payout: $${amount.toFixed(2)}`;
                if (isProjected) {
                  if (ciUpper[i] !== null && ciUpper[i] !== undefined) {
                    return [
                      `Projected Dividend Payout: $${amount.toFixed(2)}`,
                      `Expected Range: $${ciLower[i].toFixed(2)} – $${ciUpper[i].toFixed(2)}`,
                    ];
                  }
                  return `Projected Dividend Payout: $${amount.toFixed(2)}`;
                }
              },
            },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [data, theme]);

  return (
    <div style={{ marginBottom: '24px' }}>
      <div className="chart-box" id="dividendChartBox" style={{ position: 'relative' }}>
        <canvas ref={canvasRef} id="dividendChart" />
      </div>
    </div>
  );
}
