import { useEffect, useRef } from "react";
import { Chart } from "chart.js";
import "./GenericChart.css";

// A reusable wrapper around Chart.js that handles chart initialization, updates, and cleanup
export default function GenericChart({
  config,
  className = "chart-box",
  wrapperStyle = {},
  canvasId,
  onChartReady,
  updateTrigger = [],
  chartTitle,
  chartSubtitlePrice,
  chartLegend,
}) {
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !config) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.data = config.data;
      chartInstanceRef.current.options = config.options;
      chartInstanceRef.current.update();
    } else {
      const ctx = canvasRef.current.getContext("2d");
      chartInstanceRef.current = new Chart(ctx, config);
    }

    if (onChartReady) {
      onChartReady(chartInstanceRef.current);
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, updateTrigger);

  return (
    <div
      className={`generic-chart-outer-container ${className}`}
      style={wrapperStyle}
    >
      {chartTitle && <div className="generic-chart-title">{chartTitle}</div>}
      {chartSubtitlePrice && (
        <div className="generic-chart-subtitle-price">{chartSubtitlePrice}</div>
      )}
      {chartLegend && (
        <div className="generic-chart-legend-container">{chartLegend}</div>
      )}
      <div className="generic-chart-canvas-wrapper">
        <canvas ref={canvasRef} id={canvasId} />
      </div>
    </div>
  );
}
