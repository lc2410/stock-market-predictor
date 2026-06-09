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

  // Store fetched payload to allow seamless theme toggling without recalling the API
  window.lastFetchedData = null;

  // --- Theme Controller ---
  const themeToggleBtn = document.getElementById('themeToggle');
  const themeIcon = themeToggleBtn.querySelector('.theme-icon');
  const themeLabel = themeToggleBtn.querySelector('.theme-label');
  let currentTheme = localStorage.getItem('theme') || 'light';

  document.documentElement.setAttribute('data-theme', currentTheme);
  themeIcon.textContent = currentTheme === 'light' ? '🌙' : '☀️';
  themeLabel.textContent = currentTheme === 'light' ? 'Dark' : 'Light';

  themeToggleBtn.addEventListener('click', () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    themeIcon.textContent = currentTheme === 'light' ? '🌙' : '☀️';
    themeLabel.textContent = currentTheme === 'light' ? 'Dark' : 'Light';

    // Re-render chart to inject new CSS variables
    if (window.lastFetchedData && window.lastFetchedData.Chart_History) {
      renderCharts(window.lastFetchedData);
    }
  });

  // --- Search & Autocomplete Controller ---
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

        // Prevent race conditions if the user typed during the fetch
        if (isSearching || currentSearchId !== latestSearchId) return;

        if (data.length > 0) {
          autocompleteResults.innerHTML = data
            .map(
              (item) => `
            <div class="autocomplete-item" data-symbol="${item.symbol}">
              <span class="ac-sym">${item.symbol}</span>
              <span class="ac-name">${item.name || ''}</span>
            </div>
          `,
            )
            .join('');
        } else {
          autocompleteResults.innerHTML =
            '<div class="autocomplete-item" style="color: #999;">No results found</div>';
        }
      } catch (err) {
        console.error('Search failed:', err);
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
    window.lastFetchedData = null;
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
    if (
      e.target !== tickerInput &&
      !autocompleteResults.contains(e.target) &&
      e.target !== clearSearchBtn
    ) {
      autocompleteResults.innerHTML = '';
    }
  });

  function setLoadingState(isLoading) {
    tickerInput.disabled = isLoading;
    predictBtn.disabled = isLoading;
    predictBtn.style.cursor = isLoading ? 'not-allowed' : '';
    clearSearchBtn.style.display = isLoading
      ? 'none'
      : tickerInput.value.length > 0
        ? 'block'
        : 'none';
    loader.style.display = isLoading ? 'block' : 'none';
  }

  // --- API Invocation ---
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
      if (!response.ok)
        throw new Error(data.error || 'An unknown error occurred.');

      window.lastFetchedData = data;
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

  // --- DOM Rendering ---
  function displayResult(data) {
    const hasDividends = data.Next_Dividend_Date !== 'N/A';
    const divExt = data.Div_Extended_Forecasts || {};

    const divLongTermRows =
      hasDividends && Object.keys(divExt).length
        ? `
      <h3 class="subsection-heading">Long-Term Projections</h3>
      <div class="dashboard-grid">
        <div class="metric-card">
          <span class="metric-label">2nd Payout (${divExt['2_Payouts']?.Date ?? 'N/A'})</span>
          <span class="metric-value">${divExt['2_Payouts'] ? '$' + divExt['2_Payouts'].Amount.toFixed(2) : 'N/A'}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">3rd Payout (${divExt['3_Payouts']?.Date ?? 'N/A'})</span>
          <span class="metric-value">${divExt['3_Payouts'] ? '$' + divExt['3_Payouts'].Amount.toFixed(2) : 'N/A'}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">4th Payout (${divExt['4_Payouts']?.Date ?? 'N/A'})</span>
          <span class="metric-value">${divExt['4_Payouts'] ? '$' + divExt['4_Payouts'].Amount.toFixed(2) : 'N/A'}</span>
        </div>
      </div>`
        : '';

    const dividendSection = !hasDividends
      ? `
      <div class="metric-card" style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); font-style: italic;">
        This company does not currently pay dividends to its stakeholders.
      </div>`
      : `
      <div class="dashboard-grid">
        <div class="metric-card">
          <span class="metric-label">Next Dividend Date</span>
          <span class="metric-value">${data.Next_Dividend_Date}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Direction</span>
          <span class="metric-value ${data.Div_Predicted.toLowerCase()}">${data.Div_Predicted}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Confidence</span>
          <span class="metric-value">${data['Div_Confidence (%)'] === 'N/A' ? 'N/A' : data['Div_Confidence (%)'] + '%'}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Forecasted Dividend</span>
          <span class="metric-value">${typeof data.Forecasted_Dividend === 'number' ? '$' + data.Forecasted_Dividend.toFixed(2) : 'N/A'}</span>
        </div>
      </div>
      ${divLongTermRows}`;

    const histData = data.Chart_History;
    let priceTableRows = '';
    let divTableRows = '';

    if (histData) {
      for (let i = histData.dates.length - 1; i >= 0; i--) {
        priceTableRows += `
          <tr>
            <td>${histData.dates[i]}</td>
            <td><strong style="color:var(--brand-primary);">$${histData.prices[i].toFixed(2)}</strong></td>
          </tr>`;
      }
      if (histData.dividend_dates && histData.dividend_dates.length) {
        for (let i = histData.dividend_dates.length - 1; i >= 0; i--) {
          divTableRows += `
            <tr>
              <td>${histData.dividend_dates[i]}</td>
              <td><strong style="color:var(--brand-primary);">$${histData.dividend_amounts[i].toFixed(2)}</strong></td>
            </tr>`;
        }
      }
    }

    const divTableSection =
      hasDividends && divTableRows
        ? `
         <h3 class="subsection-heading">Recent Dividend Payouts (Last 12)</h3>
         <div class="table-wrapper">
           <table class="glass-table">
             <thead><tr>
               <th>Ex-Dividend Date</th>
               <th>Amount Per Share</th>
             </tr></thead>
             <tbody>${divTableRows}</tbody>
           </table>
         </div>`
        : '';

    resultContainer.innerHTML = `
      <h2 class="section-heading" style="margin-top: 10px;">${data.Company_Name} <span style="color:var(--text-muted);font-weight:600;">(${data.Ticker})</span></h2>

      <h3 class="subsection-heading" style="margin-top: 0; padding-bottom: 8px; border-bottom: 2px solid var(--outline-border);">Price Forecast</h3>

      <h3 class="subsection-heading" style="margin-top: 20px;">Next-Day Metrics</h3>
      <div class="dashboard-grid">
        <div class="metric-card">
          <span class="metric-label">Next Trading Day</span>
          <span class="metric-value">${data.Next_Trading_Day}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Direction</span>
          <span class="metric-value ${data.Price_Predicted.toLowerCase()}">${data.Price_Predicted}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Confidence</span>
          <span class="metric-value">${data['Price_Confidence (%)']}%</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Forecasted Close</span>
          <span class="metric-value">$${data.Forecasted_Close.toFixed(2)}</span>
        </div>
      </div>

      <h3 class="subsection-heading">Long-Term Projections</h3>
      <div class="dashboard-grid">
        <div class="metric-card">
          <span class="metric-label">1 Week (${data.Extended_Forecasts['1_Week'].Date})</span>
          <span class="metric-value">$${data.Extended_Forecasts['1_Week'].Price.toFixed(2)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">1 Month (${data.Extended_Forecasts['1_Month'].Date})</span>
          <span class="metric-value">$${data.Extended_Forecasts['1_Month'].Price.toFixed(2)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">1 Year (${data.Extended_Forecasts['1_Year'].Date})</span>
          <span class="metric-value">$${data.Extended_Forecasts['1_Year'].Price.toFixed(2)}</span>
        </div>
      </div>

      <div class="chart-box" style="position:relative;margin-top:32px;margin-bottom:16px;">
        <canvas id="priceChart"></canvas>
      </div>
      
      <div style="position:relative;width:100%;margin-bottom:40px;display:flex;flex-direction:column;gap:12px;">
        <div style="display:flex;justify-content:flex-end;">
          <button id="navResetBtn" class="glass-btn-small" title="Reset to full range">↺ Reset Timeline</button>
        </div>
        <div id="navWrapper" class="nav-wrapper-custom">
          <canvas id="navChart" style="width:100%;height:100%;display:block;border-radius:6px;overflow:hidden;"></canvas>
          <div id="navLeft" class="nav-overlay-custom" style="left:0; border-radius:6px 0 0 6px;"></div>
          <div id="navRight" class="nav-overlay-custom" style="right:0; border-radius:0 6px 6px 0;"></div>
          
          <div id="navHandleL" class="nav-handle-custom">
            <div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;">
              <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
              <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
            </div>
          </div>
          <div id="navHandleR" class="nav-handle-custom">
            <div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;">
              <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
              <div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div>
            </div>
          </div>
        </div>
      </div>

      <h3 class="subsection-heading">Historical Price Data (Trailing Year)</h3>
      <div class="table-wrapper">
        <table class="glass-table">
          <thead><tr>
            <th>Trading Date</th>
            <th>Close Price</th>
          </tr></thead>
          <tbody>${priceTableRows}</tbody>
        </table>
      </div>

      <h3 class="subsection-heading" style="margin-top: 56px; padding-bottom: 8px; border-bottom: 2px solid var(--outline-border);">Dividend Forecast</h3>
      <h3 class="subsection-heading" style="margin-top: 20px;">Next-Day Metrics</h3>
      ${dividendSection}
      ${divTableSection}

      <div style="margin-top:24px;">
        <div class="chart-box" id="dividendChartBox" style="position:relative;">
          <canvas id="dividendChart"></canvas>
          <div id="noDividendOverlay" style="display:none;position:absolute;inset:0;background:var(--card-bg);backdrop-filter:blur(4px);border-radius:16px;overflow:hidden;">
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
              <span style="font-size:36px;font-weight:800;color:var(--table-border);letter-spacing:2px;white-space:nowrap;transform:rotate(-15deg);font-family:Inter,sans-serif;user-select:none;">
                NO DIVIDEND DATA
              </span>
            </div>
          </div>
        </div>
      </div>
    `;

    if (data.Chart_History) setTimeout(() => renderCharts(data), 150);
  }

  // --- Chart Setup & Lifecycle ---
  function renderCharts(data) {
    const histData = data.Chart_History;

    if (priceChartInstance) priceChartInstance.destroy();
    if (navChartInstance) navChartInstance.destroy();
    if (dividendChartInstance) dividendChartInstance.destroy();

    const style = getComputedStyle(document.body);
    const brandRGB = style.getPropertyValue('--brand-rgb').trim();
    const chartHistoryColor = style.getPropertyValue('--chart-history').trim();
    const chartGridColor = style.getPropertyValue('--chart-grid').trim();
    const textMainColor = style.getPropertyValue('--text-main').trim();

    const historyMap = new Map();
    histData.dates.forEach((d, i) => historyMap.set(d, histData.prices[i]));

    const historyCoords = Array.from(historyMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x),
    );

    const anchorDate = historyCoords[historyCoords.length - 1].x;
    const anchorPrice = historyCoords[historyCoords.length - 1].y;

    const projectedToday =
      data.Train_Fit_Prices && data.Train_Fit_Prices.length > 0
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
    data.Chart_Future_Dates.forEach((d, i) =>
      unifiedMap.set(d, data.Chart_Future_Prices[i]),
    );

    const unifiedLineCoords = Array.from(unifiedMap, ([x, y]) => ({
      x,
      y,
    })).sort((a, b) => new Date(a.x) - new Date(b.x));

    const upperMap = new Map();
    upperMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) =>
      upperMap.set(d, data.Chart_Future_Upper[i]),
    );
    const upperCoords = Array.from(upperMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x),
    );

    const lowerMap = new Map();
    lowerMap.set(anchorDate, projectedToday);
    data.Chart_Future_Dates.forEach((d, i) =>
      lowerMap.set(d, data.Chart_Future_Lower[i]),
    );
    const lowerCoords = Array.from(lowerMap, ([x, y]) => ({ x, y })).sort(
      (a, b) => new Date(a.x) - new Date(b.x),
    );

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
            backgroundColor: chartHistoryColor,
            pointRadius: 2,
            pointHoverRadius: 4,
            showLine: false,
            order: 1,
          },
          {
            label: 'Projected Stock Prices',
            data: unifiedLineCoords,
            borderColor: `rgba(${brandRGB}, 1)`,
            backgroundColor: `rgba(${brandRGB}, 0.4)`,
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
            backgroundColor: `rgba(${brandRGB}, 0.15)`,
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
        color: textMainColor,
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
            grid: { color: chartGridColor },
            ticks: {
              color: textMainColor,
              maxRotation: 45,
              minRotation: 45,
              font: { size: 11 },
            },
          },
          y: {
            min: 0,
            grid: { color: chartGridColor },
            ticks: {
              color: textMainColor,
              font: { size: 11 },
              callback: (v) => `$${v.toLocaleString()}`,
            },
          },
        },
        plugins: {
          title: {
            display: true,
            text: 'Stock Price History & Forecast with 95% Confidence Interval',
            color: textMainColor,
            font: { size: 13, weight: '600' },
            padding: { bottom: 16 },
          },
          legend: {
            labels: {
              color: textMainColor,
              filter: (item) => !item.text.includes('Bound'),
              usePointStyle: false,
              sort: (a, b) => (a.text === 'Historical Stock Prices' ? -1 : 1),
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
                borderColor: textMainColor,
                borderWidth: 1.5,
                borderDash: [5, 4],
                label: {
                  display: true,
                  content: 'Today',
                  position: 'start',
                  font: { size: 10 },
                  backgroundColor: textMainColor,
                  color: currentTheme === 'light' ? '#ffffff' : '#000000',
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
                const price = ctx.parsed.y.toFixed(2);
                if (ctx.dataset.label !== 'Projected Stock Prices') {
                  return `${ctx.dataset.label}: $${price}`;
                }
                const hoverDate = ctx.raw.x;
                const ciIndex = data.Chart_Future_Dates.indexOf(hoverDate);
                if (ciIndex !== -1) {
                  const lo = data.Chart_Future_Lower[ciIndex].toFixed(2);
                  const hi = data.Chart_Future_Upper[ciIndex].toFixed(2);
                  return [
                    `Projected Stock Price: $${price}`,
                    `95% CI: $${lo} \u2013 $${hi}`,
                  ];
                }
                return `Projected Stock Price: $${price}`;
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
            backgroundColor: chartHistoryColor,
            pointRadius: 1,
            showLine: false,
            order: 1,
          },
          {
            data: unifiedLineCoords,
            borderColor: `rgba(${brandRGB}, 1)`,
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
        layout: {
            padding: { top: 10, bottom: 10 }
        },
        scales: {
          x: {
            type: 'time',
            time: { unit: 'year' },
            grid: { display: false },
            border: { display: false },
            ticks: { display: false },
          },
          y: { display: false },
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
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
      const wrapperWidth = navWrapper.getBoundingClientRect().width;
      const leftPx = tsToFrac(viewMin) * wrapperWidth;
      const rightPx = tsToFrac(viewMax) * wrapperWidth;
      navLeft.style.width = `${leftPx}px`;
      navRight.style.width = `${wrapperWidth - rightPx}px`;
      handleL.style.left = `${leftPx - 7}px`;
      handleR.style.left = `${rightPx - 7}px`;
    }

    function applyViewToMainChart() {
      priceChartInstance.options.scales.x.min = viewMin;
      priceChartInstance.options.scales.x.max = viewMax;
      priceChartInstance.update('none');
    }

    function onDragStart(e, mode) {
      dragMode = mode;
      dragStartX = e.clientX ?? e.touches[0].clientX;
      dragStartMin = viewMin;
      dragStartMax = viewMax;
      if (mode === 'left') handleL.classList.add('nav-handle-active');
      if (mode === 'right') handleR.classList.add('nav-handle-active');
      if (mode === 'pan') navWrapper.style.cursor = 'grabbing';
      e.preventDefault();
    }

    handleL.addEventListener('mousedown', (e) => onDragStart(e, 'left'));
    handleR.addEventListener('mousedown', (e) => onDragStart(e, 'right'));
    navWrapper.addEventListener('mousedown', (e) => {
      if (e.target !== handleL && e.target !== handleR) onDragStart(e, 'pan');
    });

    function onDragMove(e) {
      if (!dragMode) return;
      const clientX = e.clientX ?? e.touches[0].clientX;
      const wrapperWidth = navWrapper.getBoundingClientRect().width;
      const deltaTs = ((clientX - dragStartX) / wrapperWidth) * (maxTs - minTs);
      const span = dragStartMax - dragStartMin;

      if (dragMode === 'pan') {
        let newMin = dragStartMin + deltaTs;
        let newMax = dragStartMax + deltaTs;
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
          Math.min(dragStartMin + deltaTs, viewMax - MIN_WINDOW),
          minTs,
        );
      } else if (dragMode === 'right') {
        viewMax = Math.min(
          Math.max(dragStartMax + deltaTs, viewMin + MIN_WINDOW),
          maxTs,
        );
      }

      updateOverlays();
      applyViewToMainChart();
      e.preventDefault();
    }

    function onDragEnd() {
      handleL.classList.remove('nav-handle-active');
      handleR.classList.remove('nav-handle-active');
      navWrapper.style.cursor = 'ew-resize';
      dragMode = null;
    }

    // Bind resize observer so navigator maintains alignment
    if (window.navListeners) {
      document.removeEventListener('mousemove', window.navListeners.onDragMove);
      document.removeEventListener('mouseup', window.navListeners.onDragEnd);
      if (window.navListeners.resizeObserver)
        window.navListeners.resizeObserver.disconnect();
    }

    const resizeObserver = new ResizeObserver(() => {
      window.requestAnimationFrame(() => {
        if (navWrapper.getBoundingClientRect().width > 0) {
          updateOverlays();
          applyViewToMainChart();
        }
      });
    });

    window.navListeners = { onDragMove, onDragEnd, resizeObserver };
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
    resizeObserver.observe(navWrapper);

    document.getElementById('navResetBtn').addEventListener('click', () => {
      viewMin = minTs;
      viewMax = maxTs;
      updateOverlays();
      applyViewToMainChart();
    });

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
          bgColor: chartHistoryColor,
          upper: null,
          lower: null,
          isEst: false,
        });
      });

      const futureDates = data.Div_Future_Dates || [];
      const futureAmounts = data.Div_Future_Amounts || [];
      const futureUpper = data.Div_Future_Upper || [];
      const futureLower = data.Div_Future_Lower || [];

      futureDates.forEach((d, i) => {
        divLabelsMap.set(d, {
          amount: futureAmounts[i],
          bgColor: `rgba(${brandRGB}, 0.7)`,
          upper: futureUpper[i],
          lower: futureLower[i],
          isEst: true,
        });
      });

      const sortedDivKeys = Array.from(divLabelsMap.keys()).sort(
        (a, b) => new Date(a) - new Date(b),
      );
      const finalDivLabels = [];
      const finalDivAmounts = [];
      const finalBgColors = [];
      const ciUpper = [];
      const ciLower = [];

      sortedDivKeys.forEach((dateKey) => {
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
            },
          ],
        },
        options: {
          color: textMainColor,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              grid: { display: false },
              ticks: {
                color: textMainColor,
                maxRotation: 45,
                minRotation: 45,
                font: { size: 10 },
              },
            },
            y: {
              grid: { color: chartGridColor },
              ticks: {
                color: textMainColor,
                font: { size: 11 },
                callback: (v) => `$${v.toFixed(2)}`,
              },
            },
          },
          plugins: {
            title: {
              display: true,
              text: 'Dividend History & Forecast with 95% Confidence Interval',
              color: textMainColor,
              font: { size: 14, weight: '600' },
              padding: { bottom: 16 },
            },
            legend: {
              display: true,
              labels: {
                color: textMainColor,
                usePointStyle: true,
                generateLabels: () => {
                  const items = [
                    {
                      text: 'Historical Dividend Payout',
                      fillStyle: chartHistoryColor,
                      strokeStyle: 'transparent',
                      fontColor: textMainColor,
                    },
                  ];
                  if (futureDates.length)
                    items.push({
                      text: 'Projected Dividend Payout',
                      fillStyle: `rgba(${brandRGB}, 0.7)`,
                      strokeStyle: 'transparent',
                      fontColor: textMainColor,
                    });
                  return items;
                },
              },
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const amount = ctx.parsed.y;
                  const i = ctx.dataIndex;
                  if (ciUpper[i] !== null && ciUpper[i] !== undefined)
                    return [
                      `Projected Dividend Payout: $${amount.toFixed(2)}`,
                      `95% CI: $${ciLower[i].toFixed(2)} – $${ciUpper[i].toFixed(2)}`,
                    ];
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
