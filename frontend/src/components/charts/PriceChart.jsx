import { useEffect, useRef } from 'react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
  Filler,
  Title,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import annotationPlugin from 'chartjs-plugin-annotation';
import { getThemeColors } from '../../utils/formatters';

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
  Filler,
  Title,
  annotationPlugin
);

/**
 * Main price history + forecast line chart.
 * Props: data (full API result), viewState ({ min, max }), onChartReady (ref callback)
 */
export default function PriceChart({ data, theme, viewState, onChartReady }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    // Destroy previous instance
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }

    const colors = getThemeColors();
    const hist = data.Chart_History;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const labelTextColor = isDark ? '#0f172a' : '#ffffff';

    // Build unified price map for chart coords
    const historyMap = new Map();
    hist.dates.forEach((d, i) => historyMap.set(d, hist.prices[i]));
    const historyCoords = Array.from(historyMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x)
    );

    const anchorDate = historyCoords[historyCoords.length - 1].x;
    const unifiedMap = new Map();

    if (data.Train_Fit_Dates) {
      data.Train_Fit_Dates.forEach((d, i) => {
        if (d !== anchorDate) unifiedMap.set(d, data.Train_Fit_Prices[i]);
      });
    }

    // Anchor the projected line to the most recent historical close
    const projectedToday = data.Train_Fit_Prices?.length
      ? data.Train_Fit_Prices[data.Train_Fit_Prices.length - 1]
      : historyCoords[historyCoords.length - 1].y;
    unifiedMap.set(anchorDate, projectedToday);

    data.Chart_Future_Dates.forEach((d, i) => unifiedMap.set(d, data.Chart_Future_Prices[i]));
    const unifiedCoords = Array.from(unifiedMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x)
    );

    // Expected range bounds expand from anchor
    const upperCoords = [
      { x: anchorDate, y: projectedToday },
      ...data.Chart_Future_Dates.map((d, i) => ({ x: d, y: data.Chart_Future_Upper[i] })),
    ];
    const lowerCoords = [
      { x: anchorDate, y: projectedToday },
      ...data.Chart_Future_Dates.map((d, i) => ({ x: d, y: data.Chart_Future_Lower[i] })),
    ];

    const ctx = canvasRef.current.getContext('2d');

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Historical Stock Prices',
            data: historyCoords,
            backgroundColor: colors.history,
            borderColor: colors.history,
            borderWidth: 1.5,
            pointRadius: 2,
            order: 1,
          },
          {
            label: 'Projected Stock Prices',
            data: unifiedCoords,
            borderColor: `rgba(${colors.brandRGB}, 1)`,
            backgroundColor: `rgba(${colors.brandRGB}, 0.4)`,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2,
            order: 0,
          },
          {
            label: 'Upper Bound',
            data: upperCoords,
            backgroundColor: `rgba(${colors.brandRGB}, 0.15)`,
            borderColor: 'transparent',
            pointRadius: 0,
            pointHoverRadius: 0,
            pointHitRadius: 0,
            fill: '+1',
            tension: 0.3,
            order: 2,
          },
          {
            label: 'Lower Bound',
            data: lowerCoords,
            borderColor: 'transparent',
            pointRadius: 0,
            pointHoverRadius: 0,
            pointHitRadius: 0,
            fill: false,
            tension: 0.3,
            order: 2,
          },
        ],
      },
      options: {
        color: colors.text,
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { intersect: false, mode: 'x' },
        scales: {
          x: {
            type: 'time',
            min: viewState.min,
            max: viewState.max,
            time: { unit: 'month', tooltipFormat: 'MMM d, yyyy' },
            grid: { color: colors.grid },
            ticks: { color: colors.text, maxRotation: 45, minRotation: 45, font: { size: 11 } },
          },
          y: {
            grid: { color: colors.grid },
            ticks: {
              color: colors.text,
              font: { size: 11 },
              callback: (v) => `$${v.toLocaleString()}`,
            },
          },
        },
        plugins: {
          title: {
            display: true,
            text:
              data.Chart_Future_Dates && data.Chart_Future_Dates.length
                ? 'Closed Stock Price History & Forecast Trends with Expected Range'
                : 'Closed Stock Price History',
            color: colors.text,
            font: { size: 14, weight: '600' },
            padding: { bottom: 16 },
          },
          legend: {
            display: (data.Chart_Future_Dates && data.Chart_Future_Dates.length) > 0,
            labels: {
              color: colors.text,
              usePointStyle: true,
              generateLabels: () => {
                const items = [
                  {
                    text: 'Historical Stock Prices',
                    fillStyle: colors.history,
                    strokeStyle: 'transparent',
                    fontColor: colors.text,
                  },
                  {
                    text: 'Projected Stock Prices',
                    fillStyle: `rgba(${colors.brandRGB}, 0.4)`,
                    strokeStyle: `rgba(${colors.brandRGB}, 1)`,
                    fontColor: colors.text,
                  },
                ];
                if (data.Chart_Future_Dates && data.Chart_Future_Dates.length) {
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
          annotation: {
            annotations: {
              todayLine: {
                type: 'line',
                xMin: anchorDate,
                xMax: anchorDate,
                borderColor: colors.text,
                borderDash: [5, 4],
                label: {
                  display: true,
                  content: 'Today',
                  position: 'start',
                  font: { size: 10 },
                  backgroundColor: colors.text,
                  color: labelTextColor,
                },
              },
            },
          },
          tooltip: {
            filter: function (tooltipItem, currentIndex, tooltipItems) {
              const label = tooltipItem.dataset.label;
              const pointDate = tooltipItem.raw.x;
              const hoverDate = tooltipItems[0].raw.x;
              if (label.includes('Bound')) return false;
              if (pointDate !== hoverDate) return false;
              for (let i = 0; i < currentIndex; i++) {
                if (tooltipItems[i].datasetIndex === tooltipItem.datasetIndex) return false;
              }
              return true;
            },
            callbacks: {
              label: (ctx) => {
                const price = ctx.parsed.y.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                });
                if (ctx.dataset.label !== 'Projected Stock Prices') {
                  return `${ctx.dataset.label}: $${price}`;
                }
                const hoverDate = ctx.raw.x;
                const ciIndex = data.Chart_Future_Dates.indexOf(hoverDate);
                if (ciIndex !== -1) {
                  const lo = data.Chart_Future_Lower[ciIndex].toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  });
                  const hi = data.Chart_Future_Upper[ciIndex].toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  });
                  return [`Projected Stock Price: $${price}`, `Expected Range: $${lo} – $${hi}`];
                }
                return `Projected Stock Price: $${price}`;
              },
            },
          },
        },
      },
    });

    if (onChartReady) onChartReady(chartRef.current);

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
    // Re-create the chart whenever data OR theme changes so color palettes stay
    // in sync with the current light/dark mode.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, theme]);

  // Update x-axis range when viewState changes without re-mounting the chart
  useEffect(() => {
    if (!chartRef.current || !viewState) return;
    chartRef.current.options.scales.x.min = viewState.min;
    chartRef.current.options.scales.x.max = viewState.max;
    chartRef.current.update('none');
  }, [viewState]);

  return (
    <div className="chart-box" style={{ position: 'relative', marginTop: '32px', marginBottom: '16px' }}>
      <canvas ref={canvasRef} id="priceChart" />
    </div>
  );
}
