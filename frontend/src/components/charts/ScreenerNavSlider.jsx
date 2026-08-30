import { useEffect, useRef } from "react";
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

// Interactive timeline slider for controlling the visible time range on the main charts
export default function ScreenerNavSlider({
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

  const isDark = theme === "dark";
  const colors = {
    brandRGB: isDark ? "168, 85, 247" : "16, 185, 129",
    history: isDark ? "#ffffff" : "#000000",
  };

  let config = null;
  if (data && data.dates && data.history) {
    const coords = data.dates.map((d, i) => ({
      x: new Date(d.replace(/-/g, "/")).valueOf(),
      y: data.history[i],
    }));

    config = {
      type: "line",
      data: {
        datasets: [
          {
            data: coords,
            backgroundColor: `rgba(${colors.brandRGB}, 0.5)`,
            borderColor: `rgba(${colors.brandRGB}, 1)`,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.1,
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

  const viewStateRef = useRef(viewState);
  useEffect(() => {
    viewStateRef.current = viewState;
  }, [viewState]);

  useEffect(() => {
    if (
      !wrapperRef.current ||
      !leftRef.current ||
      !rightRef.current ||
      !hLRef.current ||
      !hRRef.current
    )
      return;
    const wrapper = wrapperRef.current;
    const hL = hLRef.current;
    const hR = hRRef.current;

    const minTs = viewStateRef.current?.absoluteMin;
    const maxTs = viewStateRef.current?.absoluteMax;
    if (!minTs || !maxTs) return;

    let dragMode = null;
    let startX = 0;
    let startMin = viewStateRef.current.min;
    let startMax = viewStateRef.current.max;

    const getW = () => wrapper.getBoundingClientRect().width;

    const onMove = (e) => {
      if (!dragMode) return;
      const cx = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      const w = getW();
      const dTs = ((cx - startX) / w) * (maxTs - minTs);
      const span = startMax - startMin;
      const MIN_WINDOW = 86400000 * 7; // 7 days

      // Calculate new time boundaries based on drag direction (pan entire window, or adjust left/right edges)
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
    };

    const onStartPan = (e) => {
      dragMode = "pan";
      startX = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      startMin = viewStateRef.current.min;
      startMax = viewStateRef.current.max;
      e.preventDefault();
    };
    const onStartL = (e) => {
      dragMode = "left";
      startX = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      startMin = viewStateRef.current.min;
      startMax = viewStateRef.current.max;
      e.preventDefault();
      e.stopPropagation();
      hL.classList.add("nav-handle-active");
    };
    const onStartR = (e) => {
      dragMode = "right";
      startX = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      startMin = viewStateRef.current.min;
      startMax = viewStateRef.current.max;
      e.preventDefault();
      e.stopPropagation();
      hR.classList.add("nav-handle-active");
    };
    const onEnd = () => {
      dragMode = null;
      hL.classList.remove("nav-handle-active");
      hR.classList.remove("nav-handle-active");
    };

    hL.addEventListener("mousedown", onStartL);
    hR.addEventListener("mousedown", onStartR);
    hL.addEventListener("touchstart", onStartL);
    hR.addEventListener("touchstart", onStartR);
    wrapper.addEventListener("mousedown", onStartPan);
    wrapper.addEventListener("touchstart", onStartPan);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onEnd);
    document.addEventListener("touchmove", onMove);
    document.addEventListener("touchend", onEnd);

    return () => {
      hL.removeEventListener("mousedown", onStartL);
      hR.removeEventListener("mousedown", onStartR);
      hL.removeEventListener("touchstart", onStartL);
      hR.removeEventListener("touchstart", onStartR);
      wrapper.removeEventListener("mousedown", onStartPan);
      wrapper.removeEventListener("touchstart", onStartPan);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onEnd);
      document.removeEventListener("touchmove", onMove);
      document.removeEventListener("touchend", onEnd);
    };
  }, [viewState?.absoluteMin, viewState?.absoluteMax, onViewChange]);

  useEffect(() => {
    if (!wrapperRef.current || !viewState || !viewState.absoluteMax) return;
    const minTs = viewState.absoluteMin;
    const maxTs = viewState.absoluteMax;
    const lPct = ((viewState.min - minTs) / (maxTs - minTs)) * 100;
    const rPct = ((viewState.max - minTs) / (maxTs - minTs)) * 100;

    if (leftRef.current) leftRef.current.style.width = `${lPct}%`;
    if (rightRef.current) rightRef.current.style.width = `${100 - rPct}%`;
    if (hLRef.current) hLRef.current.style.left = `calc(${lPct}% - 7px)`;
    if (hRRef.current) hRRef.current.style.left = `calc(${rPct}% - 7px)`;
  }, [viewState]);

  if (!viewState || !config) return null;

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
                if (!viewState) return;
                const anchorTs = viewState.absoluteMax;
                const newMin = Math.max(
                  viewState.absoluteMin,
                  anchorTs - r.days * 86400000,
                );
                onViewChange({
                  min: newMin,
                  max: anchorTs,
                  activeRange: r.label,
                });
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
        <button className="glass-btn-small" onClick={onReset}>
          ↺ Reset Timeline
        </button>
      </div>
      <div ref={wrapperRef} className="nav-wrapper-custom">
        <div className="nav-slider-chart-container">
          <GenericChart
            config={config}
            updateTrigger={[data, theme]}
            className=""
            wrapperStyle={{ width: "100%", height: "100%" }}
          />
          {/* Render Notches */}
          {(() => {
            if (!viewState) return null;
            const anchorTs = viewState.absoluteMax;
            const minTs = viewState.absoluteMin;
            const span = anchorTs - minTs;
            if (span <= 0) return null;
            const days = [7, 30, 90, 180, 270, 365];
            const notches = [];
            days.forEach((d) => {
              const leftTs = anchorTs - d * 86400000;
              if (leftTs >= minTs)
                notches.push(((leftTs - minTs) / span) * 100);
            });
            notches.push(100); // end notch
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
          className="nav-overlay-custom nav-slider-overlay-left"
        />
        <div
          ref={rightRef}
          className="nav-overlay-custom nav-slider-overlay-right"
        />
        <div ref={hLRef} className="nav-handle-custom" style={{ left: "-7px" }}>
          <div className="nav-slider-grip-dots">
            <div className="nav-slider-grip-dot" />
            <div className="nav-slider-grip-dot" />
          </div>
        </div>
        <div
          ref={hRRef}
          className="nav-handle-custom"
          style={{ right: "-7px" }}
        >
          <div className="nav-slider-grip-dots">
            <div className="nav-slider-grip-dot" />
            <div className="nav-slider-grip-dot" />
          </div>
        </div>
      </div>
    </div>
  );
}
