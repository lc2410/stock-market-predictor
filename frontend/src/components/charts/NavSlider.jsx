import { useEffect, useRef, useCallback } from "react";
import "./NavSlider.css";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
} from "chart.js";
import "chartjs-adapter-date-fns";
import GenericChart from "../common/GenericChart";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
);

/**
 * An interactive, draggable timeline slider for brushing/zooming the main charts.
 * Displays a miniature overview chart of both historical and projected data.
 */
export default function NavSlider({
  data,
  theme,
  viewState,
  onViewChange,
  onReset,
}) {
  const wrapperRef = useRef(null);
  const leftRef = useRef(null);
  const rightRef = useRef(null);
  const hLRef = useRef(null);
  const hRRef = useRef(null);

  const viewStateRef = useRef(viewState);
  useEffect(() => {
    viewStateRef.current = viewState;
  }, [viewState]);

  const isDark = theme === "dark";
  const colors = {
    brandRGB: isDark ? "168, 85, 247" : "16, 185, 129",
    history: isDark ? "#ffffff" : "#000000",
    grid: isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)",
    text: isDark ? "rgba(255, 255, 255, 0.5)" : "rgba(0, 0, 0, 0.5)",
  };
  const hist = data?.Chart_History;

  let config = null;
  if (data && hist) {
    const historyMap = new Map();
    hist.dates.forEach((d, i) => historyMap.set(d, hist.prices[i]));
    const historyCoords = Array.from(historyMap, ([x, y]) => ({ x, y })).sort(
      (a, b) =>
        new Date(typeof a.x === "string" ? a.x.replace(/-/g, "/") : a.x) -
        new Date(typeof b.x === "string" ? b.x.replace(/-/g, "/") : b.x),
    );

    const anchorDate = historyCoords[historyCoords.length - 1].x;
    const unifiedMap = new Map();
    if (data.Train_Fit_Dates) {
      data.Train_Fit_Dates.forEach((d, i) => {
        if (d !== anchorDate) unifiedMap.set(d, data.Train_Fit_Prices[i]);
      });
    }
    const projectedToday = data.Train_Fit_Prices?.length
      ? data.Train_Fit_Prices[data.Train_Fit_Prices.length - 1]
      : historyCoords[historyCoords.length - 1].y;
    unifiedMap.set(anchorDate, projectedToday);
    (data.Chart_Future_Dates || []).forEach((d, i) =>
      unifiedMap.set(d, data.Chart_Future_Prices[i]),
    );
    
    // Combine historical and projected price data into a single continuous line for the overview chart
    const unifiedCoords = Array.from(unifiedMap, ([x, y]) => ({ x, y })).sort(
      (a, b) =>
        new Date(typeof a.x === "string" ? a.x.replace(/-/g, "/") : a.x) -
        new Date(typeof b.x === "string" ? b.x.replace(/-/g, "/") : b.x),
    );

    config = {
      type: "line",
      data: {
        datasets: [
          {
            data: historyCoords,
            backgroundColor: colors.history,
            borderColor: colors.history,
            borderWidth: 1.5,
            pointRadius: 0,
            order: 1,
          },
          {
            data: unifiedCoords,
            borderColor: `rgba(${colors.brandRGB}, 1)`,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.1,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 10, bottom: 10 } },
        scales: { x: { type: "time", display: false }, y: { display: false } },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    };
  }

  useEffect(() => {
    if (!viewState || !wrapperRef.current) return;

    const wrapper = wrapperRef.current;
    const left = leftRef.current;
    const right = rightRef.current;
    const hL = hLRef.current;
    const hR = hRRef.current;

    const minTs = viewState.absoluteMin;
    const maxTs = viewState.absoluteMax;
    const MIN_WINDOW = 86400000 * 7;
    let dragMode = null;
    let startX = 0;
    let startMin = 0;
    let startMax = 0;

    // Handle slider drag events to pan the timeline or resize the visible window
    const onMove = (e) => {
      if (!dragMode) return;
      const cx = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      const w = wrapper.getBoundingClientRect().width;
      const dTs = ((cx - startX) / w) * (maxTs - minTs);
      const span = startMax - startMin;
      let newMin = viewStateRef.current.min;
      let newMax = viewStateRef.current.max;

      if (dragMode === "pan") {
        newMin = Math.max(minTs, Math.min(startMin + dTs, maxTs - span));
        newMax = newMin + span;
      } else if (dragMode === "left") {
        newMin = Math.max(
          minTs,
          Math.min(startMin + dTs, startMax - MIN_WINDOW),
        );
      } else if (dragMode === "right") {
        newMax = Math.min(
          maxTs,
          Math.max(startMax + dTs, startMin + MIN_WINDOW),
        );
      }

      onViewChange({ min: newMin, max: newMax, activeRange: null });
      const lPx = ((newMin - minTs) / (maxTs - minTs)) * w;
      const rPx = ((newMax - minTs) / (maxTs - minTs)) * w;
      left.style.width = `${lPx}px`;
      right.style.width = `${w - rPx}px`;
      hL.style.left = `${lPx - 7}px`;
      hR.style.left = `${rPx - 7}px`;
    };

    const onStart = (e, mode) => {
      dragMode = mode;
      startX = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      startMin = viewStateRef.current.min;
      startMax = viewStateRef.current.max;
      if (mode === "left") hL.classList.add("nav-handle-active");
      if (mode === "right") hR.classList.add("nav-handle-active");
      if (mode === "pan") wrapper.style.cursor = "grabbing";
      e.preventDefault();
    };

    const onEnd = () => {
      dragMode = null;
      hL.classList.remove("nav-handle-active");
      hR.classList.remove("nav-handle-active");
      wrapper.style.cursor = "ew-resize";
    };

    const onStartL = (e) => onStart(e, "left");
    const onStartR = (e) => onStart(e, "right");
    const onStartPan = (e) => {
      if (e.target !== hL && e.target !== hR) onStart(e, "pan");
    };

    hL.addEventListener("mousedown", onStartL);
    hR.addEventListener("mousedown", onStartR);
    wrapper.addEventListener("mousedown", onStartPan);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onEnd);

    // Touch events for mobile
    hL.addEventListener("touchstart", onStartL, { passive: false });
    hR.addEventListener("touchstart", onStartR, { passive: false });
    wrapper.addEventListener("touchstart", onStartPan, { passive: false });
    document.addEventListener("touchmove", onMove, { passive: false });
    document.addEventListener("touchend", onEnd);

    const resizeObserver = new ResizeObserver(() => {
      const w = wrapper.getBoundingClientRect().width;
      if (w === 0) return;
      const vs = viewStateRef.current;
      const lPx = ((vs.min - minTs) / (maxTs - minTs)) * w;
      const rPx = ((vs.max - minTs) / (maxTs - minTs)) * w;
      left.style.width = `${lPx}px`;
      right.style.width = `${w - rPx}px`;
      hL.style.left = `${lPx - 7}px`;
      hR.style.left = `${rPx - 7}px`;
    });
    resizeObserver.observe(wrapper);

    return () => {
      hL.removeEventListener("mousedown", onStartL);
      hR.removeEventListener("mousedown", onStartR);
      wrapper.removeEventListener("mousedown", onStartPan);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onEnd);

      hL.removeEventListener("touchstart", onStartL);
      hR.removeEventListener("touchstart", onStartR);
      wrapper.removeEventListener("touchstart", onStartPan);
      document.removeEventListener("touchmove", onMove);
      document.removeEventListener("touchend", onEnd);

      resizeObserver.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, theme, viewState?.absoluteMin, viewState?.absoluteMax]);

  useEffect(() => {
    if (!wrapperRef.current || !viewState) return;
    const w = wrapperRef.current.getBoundingClientRect().width;
    if (w === 0) return;
    const minTs = viewState.absoluteMin;
    const maxTs = viewState.absoluteMax;
    const lPx = ((viewState.min - minTs) / (maxTs - minTs)) * w;
    const rPx = ((viewState.max - minTs) / (maxTs - minTs)) * w;
    leftRef.current.style.width = `${lPx}px`;
    rightRef.current.style.width = `${w - rPx}px`;
    hLRef.current.style.left = `${lPx - 7}px`;
    hRRef.current.style.left = `${rPx - 7}px`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    viewState?.min,
    viewState?.max,
    viewState?.absoluteMin,
    viewState?.absoluteMax,
  ]);

  const handleReset = useCallback(() => {
    onReset();
  }, [onReset]);

  const GripDots = () => (
    <div className="nav-slider-grip-dots">
      <div className="nav-slider-grip-dot" />
      <div className="nav-slider-grip-dot" />
    </div>
  );

  return (
    <div className="nav-slider-container">
      <div className="nav-slider-controls nav-slider-controls-row">
        <div className="nav-slider-ranges nav-slider-ranges-row">
          {[
            { label: "1W", days: 7 },
            { label: "1M", days: 30 },
            { label: "3M", days: 90 },
            { label: "6M", days: 180 },
            { label: "9M", days: 270 },
            { label: "1Y", days: 365 },
          ].map((r) => (
            <button
              key={r.label}
              className={`glass-btn-small ${viewState.activeRange === r.label ? "active-range" : ""}`}
              onClick={() => {
                if (!viewState || !data?.Chart_History) return;
                const anchorDateTs = new Date(
                  data.Chart_History.dates[
                    data.Chart_History.dates.length - 1
                  ].replace(/-/g, "/"),
                ).getTime();
                const newMin = Math.max(
                  viewState.absoluteMin,
                  anchorDateTs - r.days * 86400000,
                );
                const newMax = Math.min(
                  viewState.absoluteMax,
                  anchorDateTs + r.days * 86400000,
                );
                onViewChange({
                  min: newMin,
                  max: newMax,
                  activeRange: r.label,
                });
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
        <button
          id="navResetBtn"
          className="glass-btn-small"
          onClick={handleReset}
        >
          ↺ Reset Timeline
        </button>
      </div>
      <div ref={wrapperRef} id="navWrapper" className="nav-wrapper-custom">
        <div className="nav-slider-chart-container">
          {config && (
            <GenericChart
              config={config}
              updateTrigger={[data, theme]}
              className=""
              wrapperStyle={{ width: "100%", height: "100%" }}
              canvasId="navChart"
            />
          )}
          {/* Render Notches */}
          {(() => {
            if (!viewState || !data?.Chart_History) return null;
            const anchorTs = new Date(
              data.Chart_History.dates[
                data.Chart_History.dates.length - 1
              ].replace(/-/g, "/"),
            ).getTime();
            const minTs = viewState.absoluteMin;
            const maxTs = viewState.absoluteMax;
            const span = maxTs - minTs;
            if (span <= 0) return null;
            const days = [7, 30, 90, 180, 270, 365];
            const notches = [];
            days.forEach((d) => {
              const leftTs = anchorTs - d * 86400000;
              const rightTs = anchorTs + d * 86400000;
              if (leftTs >= minTs)
                notches.push(((leftTs - minTs) / span) * 100);
              if (rightTs <= maxTs)
                notches.push(((rightTs - minTs) / span) * 100);
            });
            notches.push(((anchorTs - minTs) / span) * 100); // center notch
            return notches.map((pct, i) => (
              <div
                key={i}
                className="nav-slider-notch"
                style={{ left: `${pct}%` }}
              />
            ));
          })()}
        </div>
        <div
          ref={leftRef}
          id="navLeft"
          className="nav-overlay-custom nav-slider-overlay-left"
        />
        <div
          ref={rightRef}
          id="navRight"
          className="nav-overlay-custom nav-slider-overlay-right"
        />
        <div ref={hLRef} id="navHandleL" className="nav-handle-custom">
          <GripDots />
        </div>
        <div ref={hRRef} id="navHandleR" className="nav-handle-custom">
          <GripDots />
        </div>
      </div>
    </div>
  );
}
