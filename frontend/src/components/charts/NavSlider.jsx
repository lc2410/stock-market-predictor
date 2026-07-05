import { useEffect, useRef, useCallback } from 'react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { getThemeColors } from '../../utils/formatters';

Chart.register(LineController, LineElement, PointElement, LinearScale, TimeScale);

/**
 * Mini navigation chart + draggable range handles.
 * Uses useRef for imperative DOM/Canvas access and useEffect for cleanup.
 */
export default function NavSlider({ data, theme, viewState, onViewChange, onReset }) {
  const navCanvasRef = useRef(null);
  const navChartRef = useRef(null);
  const wrapperRef = useRef(null);
  const leftRef = useRef(null);
  const rightRef = useRef(null);
  const hLRef = useRef(null);
  const hRRef = useRef(null);

  const viewStateRef = useRef(viewState);
  useEffect(() => {
    viewStateRef.current = viewState;
  }, [viewState]);

  // Render the mini nav chart when data changes
  useEffect(() => {
    if (!data || !navCanvasRef.current) return;

    if (navChartRef.current) {
      navChartRef.current.destroy();
      navChartRef.current = null;
    }

    const colors = getThemeColors();
    const hist = data.Chart_History;
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
    const projectedToday = data.Train_Fit_Prices?.length
      ? data.Train_Fit_Prices[data.Train_Fit_Prices.length - 1]
      : historyCoords[historyCoords.length - 1].y;
    unifiedMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) => unifiedMap.set(d, data.Chart_Future_Prices[i]));
    const unifiedCoords = Array.from(unifiedMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x)
    );

    const ctx = navCanvasRef.current.getContext('2d');
    navChartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          { data: historyCoords, backgroundColor: colors.history, pointRadius: 1, order: 1 },
          {
            data: unifiedCoords,
            borderColor: `rgba(${colors.brandRGB}, 1)`,
            borderWidth: 1.5,
            pointRadius: 0,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 10, bottom: 10 } },
        scales: { x: { type: 'time', display: false }, y: { display: false } },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });

    return () => {
      if (navChartRef.current) {
        navChartRef.current.destroy();
        navChartRef.current = null;
      }
    };
    // Re-create nav chart when data OR theme changes.
  }, [data, theme]);

  // Setup drag-handle slider for panning and zooming the main price chart
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

    const onMove = (e) => {
      if (!dragMode) return;
      const cx = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      const w = wrapper.getBoundingClientRect().width;
      const dTs = ((cx - startX) / w) * (maxTs - minTs);
      const span = startMax - startMin;
      let newMin = viewStateRef.current.min;
      let newMax = viewStateRef.current.max;

      if (dragMode === 'pan') {
        newMin = Math.max(minTs, Math.min(startMin + dTs, maxTs - span));
        newMax = newMin + span;
      } else if (dragMode === 'left') {
        newMin = Math.max(minTs, Math.min(startMin + dTs, startMax - MIN_WINDOW));
      } else if (dragMode === 'right') {
        newMax = Math.min(maxTs, Math.max(startMax + dTs, startMin + MIN_WINDOW));
      }

      onViewChange({ min: newMin, max: newMax });
      // The updateUI will happen automatically via the viewState update -> DOM sync,
      // but doing it synchronously keeps the drag perfectly smooth.
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
      if (mode === 'left') hL.classList.add('nav-handle-active');
      if (mode === 'right') hR.classList.add('nav-handle-active');
      if (mode === 'pan') wrapper.style.cursor = 'grabbing';
      e.preventDefault();
    };

    const onEnd = () => {
      dragMode = null;
      hL.classList.remove('nav-handle-active');
      hR.classList.remove('nav-handle-active');
      wrapper.style.cursor = 'ew-resize';
    };

    const onStartL = (e) => onStart(e, 'left');
    const onStartR = (e) => onStart(e, 'right');
    const onStartPan = (e) => {
      if (e.target !== hL && e.target !== hR) onStart(e, 'pan');
    };

    hL.addEventListener('mousedown', onStartL);
    hR.addEventListener('mousedown', onStartR);
    wrapper.addEventListener('mousedown', onStartPan);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onEnd);

    const resizeObserver = new ResizeObserver(() => {
      // The separate viewState sync useEffect handles changes based on viewState,
      // but if the window resizes, we need to recalculate widths based on current viewStateRef
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
      hL.removeEventListener('mousedown', onStartL);
      hR.removeEventListener('mousedown', onStartR);
      wrapper.removeEventListener('mousedown', onStartPan);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onEnd);
      resizeObserver.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, theme, viewState.absoluteMin, viewState.absoluteMax]);

  // Keep slider handles in sync when viewState changes externally (e.g. from Reset button)
  useEffect(() => {
    if (!wrapperRef.current) return;
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
  }, [viewState.min, viewState.max, viewState.absoluteMin, viewState.absoluteMax]);

  const handleReset = useCallback(() => {
    onReset();
  }, [onReset]);

  // Handle grip dots inside handles
  const GripDots = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', pointerEvents: 'none' }}>
      <div style={{ width: '2px', height: '10px', background: 'rgba(255,255,255,0.8)', borderRadius: '1px' }} />
      <div style={{ width: '2px', height: '10px', background: 'rgba(255,255,255,0.8)', borderRadius: '1px' }} />
    </div>
  );

  return (
    <div style={{ position: 'relative', width: '100%', marginBottom: '40px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button id="navResetBtn" className="glass-btn-small" onClick={handleReset}>
          ↺ Reset Timeline
        </button>
      </div>
      <div ref={wrapperRef} id="navWrapper" className="nav-wrapper-custom">
        <canvas ref={navCanvasRef} id="navChart" style={{ width: '100%', height: '100%', display: 'block', borderRadius: '6px' }} />
        <div ref={leftRef} id="navLeft" className="nav-overlay-custom" style={{ left: 0, borderRadius: '6px 0 0 6px' }} />
        <div ref={rightRef} id="navRight" className="nav-overlay-custom" style={{ right: 0, borderRadius: '0 6px 6px 0' }} />
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
