// Register required Chart.js plugins
Chart.register(Chart.Filler);

document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predictBtn');
  const tickerInput = document.getElementById('tickerInput');
  const resultContainer = document.getElementById('resultContainer');
  const errorContainer = document.getElementById('errorContainer');
  const loader = document.getElementById('loader');

  let priceChartInstance = null;
  let navChartInstance = null;
  let dividendChartInstance = null;

  const handlePrediction = async () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
      showError('Please enter a ticker symbol.');
      return;
    }
    clearMessages();
    loader.style.display = 'block';
    try {
      const response = await fetch(`/predict/${ticker}`);
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || 'An unknown error occurred.');
      displayResult(data);
    } catch (error) {
      showError(error.message);
    } finally {
      loader.style.display = 'none';
    }
  };

  predictBtn.addEventListener('click', handlePrediction);
  tickerInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') handlePrediction();
  });

  // ─────────────────────────────────────────────────────────────────────────
  // displayResult — builds the HTML shell then hands off to renderCharts
  // ─────────────────────────────────────────────────────────────────────────
  function displayResult(data) {
    let html = `
            <h3>Forecast for ${data.Ticker}</h3>
            <div class="results-grid">
                <h4 class="grid-subtitle">Next-Day Price Forecast</h4>
                <div class="result-item"><strong>Next Trading Day:</strong></div>
                <div class="result-item">${data.Next_Trading_Day}</div>
                <div class="result-item"><strong>Price Direction:</strong></div>
                <div class="result-item ${data.Price_Predicted.toLowerCase()}">${data.Price_Predicted}</div>
                <div class="result-item"><strong>Price Confidence:</strong></div>
                <div class="result-item">${data['Price_Confidence (%)']}%</div>
                <div class="result-item"><strong>Forecasted Close:</strong></div>
                <div class="result-item">$${data.Forecasted_Close.toFixed(2)}</div>
                <div class="separator"></div>
                <h4 class="grid-subtitle">Long-Term Price Projection</h4>
                <div class="result-item"><strong>1 Week (${data.Extended_Forecasts['1_Week'].Date}):</strong></div>
                <div class="result-item">$${data.Extended_Forecasts['1_Week'].Price.toFixed(2)}</div>
                <div class="result-item"><strong>1 Month (${data.Extended_Forecasts['1_Month'].Date}):</strong></div>
                <div class="result-item">$${data.Extended_Forecasts['1_Month'].Price.toFixed(2)}</div>
                <div class="result-item"><strong>1 Year (${data.Extended_Forecasts['1_Year'].Date}):</strong></div>
                <div class="result-item">$${data.Extended_Forecasts['1_Year'].Price.toFixed(2)}</div>
                <div class="separator"></div>
                <h4 class="grid-subtitle">Dividend Forecast</h4>
                ${
                  data.Next_Dividend_Date === 'N/A'
                    ? `
                <div class="result-item" style="grid-column:1/-1;color:#6b7280;font-style:italic;font-size:14px;padding:6px 0;">
                    This company does not pay dividends to its stakeholders.
                </div>`
                    : `
                <div class="result-item"><strong>Next Dividend Date:</strong></div>
                <div class="result-item">${data.Next_Dividend_Date}</div>
                <div class="result-item"><strong>Dividend Direction:</strong></div>
                <div class="result-item ${data.Div_Predicted.toLowerCase()}">${data.Div_Predicted}</div>
                <div class="result-item"><strong>Dividend Confidence:</strong></div>
                <div class="result-item">${data['Div_Confidence (%)'] === 'N/A' ? 'N/A' : data['Div_Confidence (%)'] + '%'}</div>
                <div class="result-item"><strong>Forecasted Dividend:</strong></div>
                <div class="result-item">${typeof data.Forecasted_Dividend === 'number' ? '$' + data.Forecasted_Dividend.toFixed(2) : 'N/A'}</div>`
                }
            </div>
            <div class="charts-wrapper">
                <div class="chart-box" style="position:relative;">
                    <canvas id="priceChart"></canvas>
                </div>
                <!-- Navigator: mini overview chart + draggable selection window -->
                <div style="position:relative;width:100%;margin-top:-8px;margin-bottom:32px;">
                    <!-- Reset button sits above the navigator bar -->
                    <div style="display:flex;justify-content:flex-end;margin-bottom:4px;">
                        <button id="navResetBtn" style="
                            padding:3px 10px;font-size:11px;font-weight:600;
                            background:#f1f5f9;color:#374151;border:1px solid #d1d5db;
                            border-radius:4px;cursor:pointer;
                        " title="Reset to full range">↺ Reset</button>
                    </div>
                    <div id="navWrapper" style="
                        position:relative;width:100%;height:72px;
                        background:#f0f2f5;border-radius:6px;
                        border:1px solid #e5e7eb;cursor:ew-resize;user-select:none;
                        overflow:visible;
                    ">
                        <canvas id="navChart" style="width:100%;height:100%;display:block;border-radius:6px;overflow:hidden;"></canvas>
                        <!-- Grey overlays for unselected regions -->
                        <div id="navLeft"  style="position:absolute;top:0;left:0;height:100%;background:rgba(180,190,200,0.45);pointer-events:none;border-radius:6px 0 0 6px;"></div>
                        <div id="navRight" style="position:absolute;top:0;right:0;height:100%;background:rgba(180,190,200,0.45);pointer-events:none;border-radius:0 6px 6px 0;"></div>
                        <!-- Handles: wider (14px), taller than the bar, with grip lines -->
                        <div id="navHandleL" style="
                            position:absolute;top:-4px;width:14px;height:calc(100% + 8px);
                            background:rgba(37,99,235,0.75);cursor:ew-resize;
                            border-radius:4px;display:flex;align-items:center;justify-content:center;
                            transition:background 0.15s;box-shadow:0 1px 4px rgba(0,0,0,0.2);
                        ">
                            <!-- three grip lines -->
                            <div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;">
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                            </div>
                        </div>
                        <div id="navHandleR" style="
                            position:absolute;top:-4px;width:14px;height:calc(100% + 8px);
                            background:rgba(37,99,235,0.75);cursor:ew-resize;
                            border-radius:4px;display:flex;align-items:center;justify-content:center;
                            transition:background 0.15s;box-shadow:0 1px 4px rgba(0,0,0,0.2);
                        ">
                            <div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;">
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                                <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="chart-box" id="dividendChartBox" style="position:relative;">
                    <canvas id="dividendChart"></canvas>
                    <div id="noDividendOverlay" style="
                        display:none;position:absolute;inset:0;
                        background:#f3f4f6;border-radius:10px;
                        overflow:hidden;
                    ">
                        <div style="
                            width:100%;height:100%;display:flex;
                            align-items:center;justify-content:center;
                        ">
                            <p style="
                                margin:0;font-size:22px;font-weight:700;
                                color:rgba(107,114,128,0.45);text-align:center;
                                transform:rotate(-15deg);line-height:1.4;
                                user-select:none;pointer-events:none;
                                font-family:Inter, sans-serif;
                            ">This company does not pay<br>dividends to its stakeholders.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    resultContainer.innerHTML = html;
    if (data.Chart_History) setTimeout(() => renderCharts(data), 150);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // renderCharts — main chart + navigator + dividend chart
  // ─────────────────────────────────────────────────────────────────────────
  function renderCharts(data) {
    const histData = data.Chart_History;

    if (priceChartInstance) priceChartInstance.destroy();
    if (navChartInstance) navChartInstance.destroy();
    if (dividendChartInstance) dividendChartInstance.destroy();

    // ── Build coordinate arrays ───────────────────────────────────────────

    const historyCoords = histData.dates.map((d, i) => ({
      x: d,
      y: histData.prices[i],
    }));
    const anchorDate = histData.dates[histData.dates.length - 1];
    const anchorPrice = histData.prices[histData.prices.length - 1];
    const histStartDate = histData.dates[0];

    const trainFitCoords = (data.Train_Fit_Dates || [])
      .map((d, i) => ({ x: d, y: data.Train_Fit_Prices[i] }))
      .filter((pt) => pt.x >= histStartDate);

    const unifiedLineCoords = [
      ...trainFitCoords,
      { x: anchorDate, y: anchorPrice },
      ...data.Chart_Future_Dates.map((d, i) => ({
        x: d,
        y: data.Chart_Future_Prices[i],
      })),
    ];

    const upperCoords = [
      { x: anchorDate, y: anchorPrice },
      ...data.Chart_Future_Dates.map((d, i) => ({
        x: d,
        y: data.Chart_Future_Upper[i],
      })),
    ];
    const lowerCoords = [
      { x: anchorDate, y: anchorPrice },
      ...data.Chart_Future_Dates.map((d, i) => ({
        x: d,
        y: data.Chart_Future_Lower[i],
      })),
    ];

    // All dates across the full range (history + forecast), used for navigator
    const allDates = [...histData.dates, ...data.Chart_Future_Dates];
    const minTs = new Date(allDates[0]).getTime();
    const maxTs = new Date(allDates[allDates.length - 1]).getTime();

    // Initial view: show from histStartDate to end of forecast
    let viewMin = minTs;
    let viewMax = maxTs;

    // ── Main Price Chart ──────────────────────────────────────────────────

    const ctxPrice = document.getElementById('priceChart').getContext('2d');
    priceChartInstance = new Chart(ctxPrice, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Historical',
            data: historyCoords,
            borderColor: 'rgba(0,0,0,0)',
            backgroundColor: '#111827',
            pointRadius: 2,
            pointHoverRadius: 4,
            showLine: false,
            order: 1,
          },
          {
            label: 'Prediction Forecast',
            data: unifiedLineCoords,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37,99,235,0.4)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            fill: false,
            tension: 0.2,
            order: 0,
          },
          {
            label: 'Upper Bound',
            data: upperCoords,
            borderColor: 'transparent',
            backgroundColor: 'rgba(37,99,235,0.15)',
            pointRadius: 0,
            fill: '+1',
            tension: 0.3,
            order: 2,
          },
          {
            label: 'Lower Bound',
            data: lowerCoords,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.3,
            order: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { intersect: false, mode: 'nearest' },
        scales: {
          x: {
            type: 'time',
            min: viewMin,
            max: viewMax,
            time: {
              unit: 'month',
              displayFormats: { month: 'MMM yyyy' },
              tooltipFormat: 'MMM d, yyyy',
            },
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: { maxRotation: 45, minRotation: 45, font: { size: 11 } },
          },
          y: {
            min: 0,
            grid: { color: 'rgba(0,0,0,0.05)' },
            ticks: {
              font: { size: 11 },
              callback: (v) => `$${v.toLocaleString()}`,
            },
          },
        },
        plugins: {
          title: {
            display: true,
            text: 'Price Forecast with 95% Confidence Interval',
            font: { size: 13, weight: '600' },
            padding: { bottom: 16 },
          },
          legend: {
            labels: {
              filter: (item) => !item.text.includes('Bound'),
              usePointStyle: false,
            },
            onClick: function (e, legendItem, legend) {
              const chart = legend.chart;
              const meta = chart.getDatasetMeta(legendItem.datasetIndex);
              meta.hidden = !meta.hidden;
              if (legendItem.text === 'Prediction Forecast') {
                chart.data.datasets.forEach((ds, i) => {
                  if (ds.label === 'Upper Bound' || ds.label === 'Lower Bound')
                    chart.getDatasetMeta(i).hidden = meta.hidden;
                });
              }
              chart.update();
            },
          },
          annotation: {
            annotations: {
              todayLine: {
                type: 'line',
                xMin: anchorDate,
                xMax: anchorDate,
                borderColor: 'rgba(100,100,100,0.5)',
                borderWidth: 1.5,
                borderDash: [5, 4],
                label: {
                  display: true,
                  content: 'Today',
                  position: 'start',
                  font: { size: 10 },
                  color: '#666',
                },
              },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label.includes('Bound')) return null;
                if (
                  ctx.dataset.label === 'Historical' &&
                  new Date(ctx.parsed.x) >= new Date(anchorDate)
                )
                  return null;
                const lbl =
                  ctx.dataset.label === 'Prediction Forecast'
                    ? 'Forecast'
                    : ctx.dataset.label;
                return `${lbl}: $${ctx.parsed.y.toFixed(2)}`;
              },
            },
          },
        },
      },
    });

    // ── Navigator (mini overview) chart ───────────────────────────────────

    const ctxNav = document.getElementById('navChart').getContext('2d');
    navChartInstance = new Chart(ctxNav, {
      type: 'line',
      data: {
        datasets: [
          {
            data: historyCoords,
            borderColor: 'rgba(0,0,0,0)',
            backgroundColor: '#374151',
            pointRadius: 1,
            showLine: false,
            order: 1,
          },
          {
            data: unifiedLineCoords,
            borderColor: '#2563eb',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.2,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'time',
            time: { unit: 'year', displayFormats: { year: 'yyyy' } },
            grid: { display: false },
            ticks: { display: false },
          },
          y: { display: false, min: 0 },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
          annotation: { annotations: {} },
        },
      },
    });

    // ── Navigator drag logic ──────────────────────────────────────────────
    // The selection window is defined by [viewMin, viewMax] in ms timestamps.
    // Dragging the window pans; dragging the handles resizes it.

    const navWrapper = document.getElementById('navWrapper');
    const navLeft = document.getElementById('navLeft');
    const navRight = document.getElementById('navRight');
    const handleL = document.getElementById('navHandleL');
    const handleR = document.getElementById('navHandleR');

    function tsToFrac(ts) {
      return (ts - minTs) / (maxTs - minTs);
    }
    function fracToTs(frac) {
      return minTs + frac * (maxTs - minTs);
    }

    function updateOverlays() {
      const W = navWrapper.getBoundingClientRect().width;
      const leftPx = tsToFrac(viewMin) * W;
      const rightPx = tsToFrac(viewMax) * W;
      const winPx = rightPx - leftPx;

      navLeft.style.width = `${leftPx}px`;
      navRight.style.width = `${W - rightPx}px`;
      handleL.style.left = `${leftPx - 7}px`; // centre 14px handle on edge
      handleR.style.left = `${rightPx - 7}px`;
    }

    function applyViewToMainChart() {
      priceChartInstance.options.scales.x.min = viewMin;
      priceChartInstance.options.scales.x.max = viewMax;
      priceChartInstance.update('none');
    }

    // Hover feedback: brighten handle on hover
    [handleL, handleR].forEach((h) => {
      h.addEventListener('mouseenter', () => {
        h.style.background = 'rgba(37,99,235,1)';
        h.style.boxShadow = '0 2px 8px rgba(37,99,235,0.4)';
      });
      h.addEventListener('mouseleave', () => {
        if (dragMode !== 'left' && dragMode !== 'right') {
          h.style.background = 'rgba(37,99,235,0.75)';
          h.style.boxShadow = '0 1px 4px rgba(0,0,0,0.2)';
        }
      });
    });

    // Reset button — restores full range
    document.getElementById('navResetBtn').addEventListener('click', () => {
      viewMin = minTs;
      viewMax = maxTs;
      updateOverlays();
      applyViewToMainChart();
    });

    // Minimum window: 7 days in ms
    const MIN_WINDOW = 7 * 24 * 3600 * 1000;

    let dragMode = null; // 'pan' | 'left' | 'right'
    let dragStartX = 0;
    let dragStartMin = 0;
    let dragStartMax = 0;

    function onDragStart(e, mode) {
      dragMode = mode;
      dragStartX = e.clientX ?? e.touches[0].clientX;
      dragStartMin = viewMin;
      dragStartMax = viewMax;
      // Active state: fully opaque while dragging
      if (mode === 'left') {
        handleL.style.background = 'rgba(29,78,216,1)';
        handleL.style.cursor = 'grabbing';
      }
      if (mode === 'right') {
        handleR.style.background = 'rgba(29,78,216,1)';
        handleR.style.cursor = 'grabbing';
      }
      if (mode === 'pan') navWrapper.style.cursor = 'grabbing';
      e.preventDefault();
    }

    handleL.addEventListener('mousedown', (e) => onDragStart(e, 'left'));
    handleR.addEventListener('mousedown', (e) => onDragStart(e, 'right'));
    navWrapper.addEventListener('mousedown', (e) => {
      // Only start pan if not clicking a handle
      if (e.target === handleL || e.target === handleR) return;
      onDragStart(e, 'pan');
    });

    // Touch equivalents
    handleL.addEventListener('touchstart', (e) => onDragStart(e, 'left'), {
      passive: false,
    });
    handleR.addEventListener('touchstart', (e) => onDragStart(e, 'right'), {
      passive: false,
    });
    navWrapper.addEventListener(
      'touchstart',
      (e) => {
        if (e.target === handleL || e.target === handleR) return;
        onDragStart(e, 'pan');
      },
      { passive: false },
    );

    function onDragMove(e) {
      if (!dragMode) return;
      const clientX = e.clientX ?? e.touches[0].clientX;
      const W = navWrapper.getBoundingClientRect().width;
      const dxFrac = (clientX - dragStartX) / W;
      const dxTs = dxFrac * (maxTs - minTs);
      const span = dragStartMax - dragStartMin;

      if (dragMode === 'pan') {
        let newMin = dragStartMin + dxTs;
        let newMax = dragStartMax + dxTs;
        // Clamp to full range
        if (newMin < minTs) {
          newMin = minTs;
          newMax = minTs + span;
        }
        if (newMax > maxTs) {
          newMax = maxTs;
          newMin = maxTs - span;
        }
        viewMin = newMin;
        viewMax = newMax;
      } else if (dragMode === 'left') {
        viewMin = Math.min(dragStartMin + dxTs, viewMax - MIN_WINDOW);
        viewMin = Math.max(viewMin, minTs);
      } else if (dragMode === 'right') {
        viewMax = Math.max(dragStartMax + dxTs, viewMin + MIN_WINDOW);
        viewMax = Math.min(viewMax, maxTs);
      }

      updateOverlays();
      applyViewToMainChart();
      e.preventDefault();
    }

    function onDragEnd() {
      // Restore handle appearance
      handleL.style.background = 'rgba(37,99,235,0.75)';
      handleL.style.cursor = 'ew-resize';
      handleR.style.background = 'rgba(37,99,235,0.75)';
      handleR.style.cursor = 'ew-resize';
      navWrapper.style.cursor = 'ew-resize';
      dragMode = null;
    }

    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
    document.addEventListener('touchmove', onDragMove, { passive: false });
    document.addEventListener('touchend', onDragEnd);

    // Initialise overlays after chart renders so getBoundingClientRect is valid
    setTimeout(updateOverlays, 200);

    // ── Dividend Chart ────────────────────────────────────────────────────

    const noDividend =
      data.Next_Dividend_Date === 'N/A' || !histData.dividend_dates.length;

    if (noDividend) {
      // Hide the canvas, show the greyed-out overlay instead
      document.getElementById('dividendChart').style.display = 'none';
      document.getElementById('noDividendOverlay').style.display = 'block';
    } else {
      const divLabels = [...histData.dividend_dates];
      const divAmounts = [...histData.dividend_amounts];
      const bgColors = Array(divAmounts.length).fill('#16a34a');

      if (
        typeof data.Forecasted_Dividend === 'number' &&
        data.Next_Dividend_Date !== 'N/A'
      ) {
        divLabels.push(data.Next_Dividend_Date + ' (Est)');
        divAmounts.push(data.Forecasted_Dividend);
        bgColors.push('#93c5fd');
      }

      const ctxDiv = document.getElementById('dividendChart').getContext('2d');
      dividendChartInstance = new Chart(ctxDiv, {
        type: 'bar',
        data: {
          labels: divLabels,
          datasets: [
            {
              label: 'Dividend Payout ($)',
              data: divAmounts,
              backgroundColor: bgColors,
              borderRadius: 5,
              borderSkipped: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              grid: { display: false },
              ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } },
            },
            y: {
              grid: { color: 'rgba(0,0,0,0.05)' },
              ticks: {
                font: { size: 11 },
                callback: (v) => `$${v.toFixed(2)}`,
              },
            },
          },
          plugins: {
            title: {
              display: true,
              text: 'Dividend History & Next Projected Payout',
              font: { size: 14, weight: '600' },
              padding: { bottom: 16 },
            },
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Dividend: $${ctx.parsed.y.toFixed(4)}`,
              },
            },
          },
        },
      });
    }
  }

  function showError(message) {
    errorContainer.textContent = `Error: ${message}`;
  }
  function clearMessages() {
    resultContainer.innerHTML = '';
    errorContainer.textContent = '';
  }
});
