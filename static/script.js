Chart.register(Chart.Filler);

document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predictBtn');
  const tickerInput = document.getElementById('tickerInput');
  const clearSearchBtn = document.getElementById('clearSearchBtn');
  const autocompleteResults = document.getElementById('autocompleteResults');
  const resultContainer = document.getElementById('resultContainer');
  const errorContainer = document.getElementById('errorContainer');
  const loader = document.getElementById('loader');

  let priceChartInstance = null;
  let navChartInstance = null;
  let dividendChartInstance = null;

  let debounceTimer;
  let isSearching = false; 
  let latestSearchId = 0;  

  tickerInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    const currentSearchId = ++latestSearchId; 
    
    clearSearchBtn.style.display = query.length > 0 ? 'block' : 'none';

    clearTimeout(debounceTimer);
    if (!query || isSearching) {
      autocompleteResults.innerHTML = '';
      return;
    }
    
    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/search/${query}`);
        const data = await res.json();
        
        if (isSearching || currentSearchId !== latestSearchId) return;
        
        if (data.length > 0) {
          autocompleteResults.innerHTML = data.map(item => `
            <div class="autocomplete-item" data-symbol="${item.symbol}">
              <span class="ac-sym">${item.symbol}</span>
              <span class="ac-name">${item.name || ''}</span>
            </div>
          `).join('');
        } else {
          autocompleteResults.innerHTML = '<div class="autocomplete-item" style="color: #999;">No results found</div>';
        }
      } catch (err) {
        console.error("Search failed:", err);
      }
    }, 300);
  });

  clearSearchBtn.addEventListener('click', () => {
    tickerInput.value = '';
    clearSearchBtn.style.display = 'none';
    autocompleteResults.innerHTML = '';
    latestSearchId++; 
    
    if (priceChartInstance) priceChartInstance.destroy();
    if (navChartInstance) navChartInstance.destroy();
    if (dividendChartInstance) dividendChartInstance.destroy();
    
    clearMessages();
    tickerInput.focus();
  });

  autocompleteResults.addEventListener('click', (e) => {
    const item = e.target.closest('.autocomplete-item');
    if (item && item.getAttribute('data-symbol')) {
      tickerInput.value = item.getAttribute('data-symbol');
      autocompleteResults.innerHTML = '';
      handlePrediction();
    }
  });

  document.addEventListener('click', (e) => {
    if (e.target !== tickerInput && !autocompleteResults.contains(e.target) && e.target !== clearSearchBtn) {
      autocompleteResults.innerHTML = '';
    }
  });

  function setLoadingState(isLoading) {
    tickerInput.disabled = isLoading;
    predictBtn.disabled = isLoading;
    predictBtn.style.cursor = isLoading ? 'not-allowed' : '';
    clearSearchBtn.style.display = isLoading ? 'none' : (tickerInput.value.length > 0 ? 'block' : 'none');
    
    if (isLoading) {
        loader.style.display = 'block';
    } else {
        loader.style.display = 'none';
    }
  }

  const handlePrediction = async () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
      showError('Please enter a ticker symbol.');
      return;
    }
    
    clearTimeout(debounceTimer);
    latestSearchId++; 
    autocompleteResults.innerHTML = '';
    isSearching = true;

    clearMessages();
    setLoadingState(true);
    
    try {
      const response = await fetch(`/predict/${ticker}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'An unknown error occurred.');
      displayResult(data);
    } catch (error) {
      showError(error.message);
    } finally {
      setLoadingState(false);
      isSearching = false; 
    }
  };

  predictBtn.addEventListener('click', handlePrediction);
  tickerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault(); 
        handlePrediction();
    }
  });

  function displayResult(data) {
    const hasDividends = data.Next_Dividend_Date !== 'N/A';
    const divExt = data.Div_Extended_Forecasts || {};

    const divLongTermRows =
      hasDividends && Object.keys(divExt).length
        ? `
      <div class="separator"></div>
      <h4 class="grid-subtitle">Long-Term Dividend Projection</h4>
      <div class="result-item"><strong>2nd Payout (${divExt['2_Payouts']?.Date ?? 'N/A'}):</strong></div>
      <div class="result-item">${divExt['2_Payouts'] ? '$' + divExt['2_Payouts'].Amount.toFixed(2) : 'N/A'}</div>
      <div class="result-item"><strong>3rd Payout (${divExt['3_Payouts']?.Date ?? 'N/A'}):</strong></div>
      <div class="result-item">${divExt['3_Payouts'] ? '$' + divExt['3_Payouts'].Amount.toFixed(2) : 'N/A'}</div>
      <div class="result-item"><strong>4th Payout (${divExt['4_Payouts']?.Date ?? 'N/A'}):</strong></div>
      <div class="result-item">${divExt['4_Payouts'] ? '$' + divExt['4_Payouts'].Amount.toFixed(2) : 'N/A'}</div>`
        : '';

    const dividendSection = !hasDividends
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
      <div class="result-item">${typeof data.Forecasted_Dividend === 'number' ? '$' + data.Forecasted_Dividend.toFixed(2) : 'N/A'}</div>
      ${divLongTermRows}`;

    const histData = data.Chart_History;
    let priceTableRows = '';
    let divTableRows = '';

    if (histData) {
      for (let i = histData.dates.length - 1; i >= 0; i--) {
        priceTableRows += `
          <tr>
            <td style="text-align:left;">${histData.dates[i]}</td>
            <td>$${histData.prices[i].toFixed(2)}</td>
          </tr>`;
      }

      if (histData.dividend_dates && histData.dividend_dates.length) {
        for (let i = histData.dividend_dates.length - 1; i >= 0; i--) {
          divTableRows += `
            <tr>
              <td style="text-align:left;">${histData.dividend_dates[i]}</td>
              <td>$${histData.dividend_amounts[i].toFixed(2)}</td>
            </tr>`;
        }
      }
    }

    const divTableSection = hasDividends && divTableRows
      ? `<div class="separator" style="margin-top:32px;"></div>
         <h4 class="grid-subtitle" style="text-align:left;">Recent Dividend Payouts (Last 12)</h4>
         <div class="table-wrapper">
           <table class="history-table">
             <thead><tr>
               <th style="text-align:left;">Ex-Dividend Date</th>
               <th>Amount Per Share</th>
             </tr></thead>
             <tbody>${divTableRows}</tbody>
           </table>
         </div>`
      : '';

    resultContainer.innerHTML = `
      <h3>Forecast for ${data.Ticker}</h3>

      <h2 class="section-heading">Price Forecast</h2>

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
      </div>

      <div class="separator" style="margin-top:24px;"></div>
      <h4 class="grid-subtitle" style="text-align:left;">Historical Price Data (Trailing Year)</h4>
      <div class="table-wrapper">
        <table class="history-table">
          <thead><tr>
            <th style="text-align:left;">Date</th>
            <th>Close Price</th>
          </tr></thead>
          <tbody>${priceTableRows}</tbody>
        </table>
      </div>

      <div class="separator" style="margin-top:32px;"></div>
      
      <div class="chart-box" style="position:relative;margin-top:24px;margin-bottom:16px;">
        <canvas id="priceChart"></canvas>
      </div>
      <div style="position:relative;width:100%;margin-bottom:40px;display:flex;flex-direction:column;gap:12px;">
        <div style="display:flex;justify-content:flex-end;">
          <button id="navResetBtn" style="
            padding:6px 14px;font-size:12px;font-weight:600;
            background:#f1f5f9;color:#374151;border:1px solid #d1d5db;
            border-radius:4px;cursor:pointer;transition:background 0.2s;
          " onmouseover="this.style.background='#e2e8f0'" onmouseout="this.style.background='#f1f5f9'" title="Reset to full range">↺ Reset</button>
        </div>
        <div id="navWrapper" style="
          position:relative;width:100%;height:72px;
          background:#f0f2f5;border-radius:6px;
          border:1px solid #e5e7eb;cursor:ew-resize;user-select:none;overflow:visible;
        ">
          <canvas id="navChart" style="width:100%;height:100%;display:block;border-radius:6px;overflow:hidden;"></canvas>
          <div id="navLeft"  style="position:absolute;top:0;left:0;height:100%;background:rgba(180,190,200,0.45);pointer-events:none;border-radius:6px 0 0 6px;"></div>
          <div id="navRight" style="position:absolute;top:0;right:0;height:100%;background:rgba(180,190,200,0.45);pointer-events:none;border-radius:0 6px 6px 0;"></div>
          <div id="navHandleL" style="
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

      <h2 class="section-heading">Dividend Forecast</h2>

      <div class="results-grid">
        <h4 class="grid-subtitle">Next Dividend Forecast</h4>
        ${dividendSection}
      </div>

      ${divTableSection}

      <div class="separator" style="margin-top:32px;"></div>
      <div style="margin-top:24px;">
        <div class="chart-box" id="dividendChartBox" style="position:relative;">
          <canvas id="dividendChart"></canvas>
          <div id="noDividendOverlay" style="
            display:none;position:absolute;inset:0;
            background:#f3f4f6;border-radius:10px;overflow:hidden;
          ">
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
              <span style="
                font-size:42px;font-weight:800;color:rgba(156,163,175,0.2);
                letter-spacing:3px;white-space:nowrap;transform:rotate(-25deg);
                font-family:Inter,sans-serif;user-select:none;
              ">NO DIVIDEND DATA</span>
            </div>
          </div>
        </div>
      </div>
    `;

    if (data.Chart_History) setTimeout(() => renderCharts(data), 150);
  }

  function renderCharts(data) {
    const histData = data.Chart_History;

    if (priceChartInstance) priceChartInstance.destroy();
    if (navChartInstance) navChartInstance.destroy();
    if (dividendChartInstance) dividendChartInstance.destroy();

    const historyMap = new Map();
    histData.dates.forEach((d, i) => historyMap.set(d, histData.prices[i]));
    
    const historyCoords = Array.from(historyMap, ([x, y]) => ({ x, y }))
                               .sort((a, b) => new Date(a.x) - new Date(b.x));

    const anchorDate = historyCoords[historyCoords.length - 1].x;
    const anchorPrice = historyCoords[historyCoords.length - 1].y;

    const projectedToday = data.Train_Fit_Prices && data.Train_Fit_Prices.length > 0
        ? data.Train_Fit_Prices[data.Train_Fit_Prices.length - 1]
        : anchorPrice;

    const unifiedMap = new Map();
    if (data.Train_Fit_Dates) {
        data.Train_Fit_Dates.forEach((d, i) => {
            if (d >= historyCoords[0].x && d !== anchorDate) {
                unifiedMap.set(d, data.Train_Fit_Prices[i]);
            }
        });
    }
    unifiedMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) => unifiedMap.set(d, data.Chart_Future_Prices[i]));
    
    const unifiedLineCoords = Array.from(unifiedMap, ([x, y]) => ({ x, y }))
                                   .sort((a, b) => new Date(a.x) - new Date(b.x));

    const upperMap = new Map();
    upperMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) => upperMap.set(d, data.Chart_Future_Upper[i]));
    const upperCoords = Array.from(upperMap, ([x, y]) => ({ x, y })).sort((a, b) => new Date(a.x) - new Date(b.x));

    const lowerMap = new Map();
    lowerMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) => lowerMap.set(d, data.Chart_Future_Lower[i]));
    const lowerCoords = Array.from(lowerMap, ([x, y]) => ({ x, y })).sort((a, b) => new Date(a.x) - new Date(b.x));

    const allDates = [...histData.dates, ...data.Chart_Future_Dates];
    const minTs = new Date(allDates[0]).getTime();
    const maxTs = new Date(allDates[allDates.length - 1]).getTime();
    let viewMin = minTs;
    let viewMax = maxTs;

    const ctxPrice = document.getElementById('priceChart').getContext('2d');
    priceChartInstance = new Chart(ctxPrice, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Historical Stock Prices',
            data: historyCoords,
            borderColor: 'rgba(0,0,0,0)',
            backgroundColor: '#111827',
            pointRadius: 2,
            pointHoverRadius: 4,
            showLine: false,
            order: 1,
          },
          {
            label: 'Projected Stock Prices',
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
            pointHoverRadius: 0,
            hitRadius: 0,
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
            pointHoverRadius: 0,
            hitRadius: 0,
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
        interaction: { intersect: false, mode: 'x' },
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
            text: 'Stock Price History & Forecast with 95% Confidence Interval',
            font: { size: 13, weight: '600' },
            padding: { bottom: 16 },
          },
          legend: {
            labels: {
              filter: (item) => !item.text.includes('Bound'),
              usePointStyle: false,
              sort: (a, b) => {
                if (a.text === 'Historical Stock Prices') return -1;
                if (b.text === 'Historical Stock Prices') return 1;
                return 0;
              }
            },
            onClick: function (e, legendItem, legend) {
              const chart = legend.chart;
              const meta = chart.getDatasetMeta(legendItem.datasetIndex);
              meta.hidden = !meta.hidden;
              if (legendItem.text === 'Projected Stock Prices') {
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
                  color: '#ffffff',
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
                if (tooltipItems[i].datasetIndex === tooltipItem.datasetIndex)
                  return false;
              }

              return true;
            },
            callbacks: {
              label: (ctx) => {
                return `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`;
              },
            },
          },
        },
      },
    });

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

    const navWrapper = document.getElementById('navWrapper');
    const navLeft = document.getElementById('navLeft');
    const navRight = document.getElementById('navRight');
    const handleL = document.getElementById('navHandleL');
    const handleR = document.getElementById('navHandleR');

    const MIN_WINDOW = 7 * 24 * 3600 * 1000;
    let dragMode = null;
    let dragStartX = 0;
    let dragStartMin = 0;
    let dragStartMax = 0;

    function tsToFrac(ts) {
      return (ts - minTs) / (maxTs - minTs);
    }

    function updateOverlays() {
      const W = navWrapper.getBoundingClientRect().width;
      const leftPx = tsToFrac(viewMin) * W;
      const rightPx = tsToFrac(viewMax) * W;
      navLeft.style.width = `${leftPx}px`;
      navRight.style.width = `${W - rightPx}px`;
      handleL.style.left = `${leftPx - 7}px`;
      handleR.style.left = `${rightPx - 7}px`;
    }

    function applyViewToMainChart() {
      priceChartInstance.options.scales.x.min = viewMin;
      priceChartInstance.options.scales.x.max = viewMax;
      priceChartInstance.update('none');
    }

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

    document.getElementById('navResetBtn').addEventListener('click', () => {
      viewMin = minTs;
      viewMax = maxTs;
      updateOverlays();
      applyViewToMainChart();
    });

    function onDragStart(e, mode) {
      dragMode = mode;
      dragStartX = e.clientX ?? e.touches[0].clientX;
      dragStartMin = viewMin;
      dragStartMax = viewMax;
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
      if (e.target === handleL || e.target === handleR) return;
      onDragStart(e, 'pan');
    });
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
      const dxTs = ((clientX - dragStartX) / W) * (maxTs - minTs);
      const span = dragStartMax - dragStartMin;

      if (dragMode === 'pan') {
        let newMin = dragStartMin + dxTs;
        let newMax = dragStartMax + dxTs;
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
        viewMin = Math.max(
          Math.min(dragStartMin + dxTs, viewMax - MIN_WINDOW),
          minTs,
        );
      } else if (dragMode === 'right') {
        viewMax = Math.min(
          Math.max(dragStartMax + dxTs, viewMin + MIN_WINDOW),
          maxTs,
        );
      }

      updateOverlays();
      applyViewToMainChart();
      e.preventDefault();
    }

    function onDragEnd() {
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

    setTimeout(updateOverlays, 200);

    const noDividend =
      data.Next_Dividend_Date === 'N/A' || !histData.dividend_dates.length;

    if (noDividend) {
      document.getElementById('dividendChart').style.display = 'none';
      document.getElementById('noDividendOverlay').style.display = 'block';
    } else {
      const divLabelsMap = new Map();
      
      histData.dividend_dates.forEach((d, i) => {
          divLabelsMap.set(d, {
              amount: histData.dividend_amounts[i],
              bgColor: '#111827',
              upper: null,
              lower: null,
              isEst: false
          });
      });

      const futureDates = data.Div_Future_Dates || [];
      const futureAmounts = data.Div_Future_Amounts || [];
      const futureUpper = data.Div_Future_Upper || [];
      const futureLower = data.Div_Future_Lower || [];

      futureDates.forEach((d, i) => {
          divLabelsMap.set(d, {
              amount: futureAmounts[i],
              bgColor: '#93c5fd',
              upper: futureUpper[i],
              lower: futureLower[i],
              isEst: true
          });
      });

      const sortedDivKeys = Array.from(divLabelsMap.keys()).sort((a, b) => new Date(a) - new Date(b));
      
      const finalDivLabels = [];
      const finalDivAmounts = [];
      const finalBgColors = [];
      const ciUpper = [];
      const ciLower = [];

      sortedDivKeys.forEach(dateKey => {
          const entry = divLabelsMap.get(dateKey);
          finalDivLabels.push(entry.isEst ? `${dateKey} (Est.)` : dateKey);
          finalDivAmounts.push(entry.amount);
          finalBgColors.push(entry.bgColor);
          ciUpper.push(entry.upper);
          ciLower.push(entry.lower);
      });

      const ctxDiv = document.getElementById('dividendChart').getContext('2d');
      dividendChartInstance = new Chart(ctxDiv, {
        type: 'bar',
        data: {
          labels: finalDivLabels,
          datasets: [
            {
              label: 'Dividend Payout ($)',
              data: finalDivAmounts,
              backgroundColor: finalBgColors,
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
              text: 'Dividend History & Forecast with 95% Confidence Interval',
              font: { size: 14, weight: '600' },
              padding: { bottom: 16 },
            },
            legend: {
              display: true,
              labels: {
                usePointStyle: true,
                generateLabels: () => {
                  const items = [
                    {
                      text: 'Historical Dividend Payout',
                      fillStyle: '#111827',
                      strokeStyle: 'transparent',
                    },
                  ];
                  if (futureDates.length) {
                    items.push({
                      text: 'Projected Dividend Payout',
                      fillStyle: '#93c5fd',
                      strokeStyle: 'transparent',
                    });
                  }
                  return items;
                },
              },
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const amount = ctx.parsed.y;
                  const i = ctx.dataIndex;
                  if (ciUpper[i] !== null && ciUpper[i] !== undefined) {
                    return [
                      `Projected Dividend Payout: $${amount.toFixed(2)}`,
                      `95% CI: $${ciLower[i].toFixed(2)} – $${ciUpper[i].toFixed(2)}`,
                    ];
                  }
                  return `Historical Dividend Payout: $${amount.toFixed(2)}`;
                },
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