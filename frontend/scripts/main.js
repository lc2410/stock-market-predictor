Chart.register(Chart.Filler);

// utility functions
const Utils = {
  formatDate: (dateStr) => {
    if (!dateStr || dateStr === 'N/A') return 'N/A';
    const parts = dateStr.split('-');
    if (parts.length !== 3) return dateStr;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${months[parseInt(parts[1], 10) - 1]} ${parseInt(parts[2], 10)}, ${parts[0]}`;
  },
  formatMoney: (val) => typeof val === 'number' ? `$${val.toFixed(2)}` : 'N/A',
  getThemeColors: () => {
    const style = getComputedStyle(document.body);
    return {
      brandRGB: style.getPropertyValue('--brand-rgb').trim(),
      history: style.getPropertyValue('--chart-history').trim(),
      grid: style.getPropertyValue('--chart-grid').trim(),
      text: style.getPropertyValue('--text-main').trim(),
    };
  }
};

const State = {
  lastFetchedData: null,
  isSearching: false,
  latestSearchId: 0,
};

const Elements = {}; // Populated on DOMContentLoaded

// theme manager
const ThemeManager = {
  currentTheme: localStorage.getItem('theme') || 'light',
  
  init() {
    this.applyTheme(this.currentTheme);
    Elements.themeToggleBtn.addEventListener('click', () => this.toggle());
  },
  
  toggle() {
    this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    this.applyTheme(this.currentTheme);
    localStorage.setItem('theme', this.currentTheme);

    if (State.lastFetchedData && State.lastFetchedData.Chart_History) {
      ChartManager.renderAll(State.lastFetchedData);
    }
  },
  
  applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    Elements.themeIcon.textContent = theme === 'light' ? '🌙' : '☀️';
    Elements.themeLabel.textContent = theme === 'light' ? 'Dark' : 'Light';
  }
};

// search manager
const SearchManager = {
  debounceTimer: null,

  init() {
    Elements.tickerInput.addEventListener('input', (e) => this.handleInput(e.target.value.trim()));
    Elements.clearSearchBtn.addEventListener('click', () => this.clearSearch());
    Elements.autocompleteResults.addEventListener('click', (e) => this.handleSelection(e));
    
    // Close autocomplete on outside click
    document.addEventListener('click', (e) => {
      if (e.target !== Elements.tickerInput && !Elements.autocompleteResults.contains(e.target) && e.target !== Elements.clearSearchBtn) {
        Elements.autocompleteResults.innerHTML = '';
      }
    });
  },

  handleInput(query) {
    const currentSearchId = ++State.latestSearchId;
    Elements.clearSearchBtn.style.display = query.length > 0 ? 'block' : 'none';
    clearTimeout(this.debounceTimer);

    if (!query || State.isSearching) {
      Elements.autocompleteResults.innerHTML = '';
      return;
    }

    this.debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/search/${query}`);
        const data = await res.json();
        
        if (State.isSearching || currentSearchId !== State.latestSearchId) return;

        if (data.length > 0) {
          Elements.autocompleteResults.innerHTML = data.map(item => `
            <div class="autocomplete-item" data-symbol="${item.symbol}">
              <span class="ac-sym">${item.symbol}</span><span class="ac-name">${item.name || ''}</span>
            </div>
          `).join('');
        } else {
          Elements.autocompleteResults.innerHTML = '<div class="autocomplete-item" style="color: #999;">No results found</div>';
        }
      } catch (err) { console.error('Search failed:', err); }
    }, 300);
  },

  handleSelection(e) {
    const item = e.target.closest('.autocomplete-item');
    if (item && item.getAttribute('data-symbol')) {
      Elements.tickerInput.value = item.getAttribute('data-symbol');
      Elements.autocompleteResults.innerHTML = '';
      App.fetchPrediction();
    }
  },

  clearSearch() {
    Elements.tickerInput.value = '';
    Elements.clearSearchBtn.style.display = 'none';
    Elements.autocompleteResults.innerHTML = '';
    State.latestSearchId++;
    
    ChartManager.destroyAll();
    ChartManager.viewState = { min: 0, max: 0, absoluteMin: 0, absoluteMax: 0 };
    UIManager.clearDisplay();
    State.lastFetchedData = null;
    Elements.tickerInput.focus();
  }
};

// ui manager
const UIManager = {
  setLoading(isLoading) {
    Elements.tickerInput.disabled = isLoading;
    Elements.predictBtn.disabled = isLoading;
    Elements.predictBtn.style.cursor = isLoading ? 'not-allowed' : '';
    Elements.clearSearchBtn.style.display = isLoading ? 'none' : (Elements.tickerInput.value.length > 0 ? 'block' : 'none');
    Elements.loader.style.display = isLoading ? 'block' : 'none';
    if(isLoading) this.clearDisplay();
  },

  showError(message) {
    Elements.errorContainer.textContent = `Error: ${message}`;
    Elements.errorContainer.style.display = 'block';
  },

  clearDisplay() {
    Elements.resultContainer.innerHTML = '';
    Elements.errorContainer.textContent = '';
    Elements.errorContainer.style.display = 'none';
  },

  renderDashboard(data) {
    const hasDividends = data.Chart_History?.dividend_dates && data.Chart_History.dividend_dates.length > 0;
    
    // Standardizes incoming timestamps to YYYY-MM-DD for reliable Map indexing
    const normalizeDate = (isoString) => {
        if (!isoString) return '';
        return new Date(isoString).toISOString().split('T')[0];
    };

    // Build Unified Price Rows
    const pMap = new Map();

    if (data.Chart_History?.dates && data.Chart_History.dates.length > 0) {
      data.Chart_History.dates.forEach((d, i) => {
        const k = normalizeDate(d);
        pMap.set(k, { date: k, hist: data.Chart_History.prices[i], proj: null, lower: null, upper: null });
      });
    }

    if (data.Train_Fit_Dates) {
      data.Train_Fit_Dates.forEach((d, i) => {
        const k = normalizeDate(d);
        if (!pMap.has(k)) pMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
        
        const price = data.Train_Fit_Prices[i];
        if (price !== undefined && price !== null) pMap.get(k).proj = price;
      });
    }

    if (data.Chart_Future_Dates) {
      data.Chart_Future_Dates.forEach((d, i) => {
        const k = normalizeDate(d);
        if (!pMap.has(k)) pMap.set(k, { date: k, hist: null, proj: null, lower: null, upper: null });
        
        pMap.get(k).proj = data.Chart_Future_Prices[i];
        pMap.get(k).lower = data.Chart_Future_Lower[i];
        pMap.get(k).upper = data.Chart_Future_Upper[i];
      });
    }
    
    const priceRows = Array.from(pMap.values()).sort((a, b) => new Date(b.date) - new Date(a.date));

    // Build Unified Dividend Rows
    const dMap = new Map();

    if (data.Chart_History?.dividend_dates && data.Chart_History.dividend_dates.length > 0) {
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
    
    const divRows = Array.from(dMap.values()).sort((a, b) => new Date(b.date) - new Date(a.date));

    const getGradeColor = (grade) => grade.includes('A') || grade.includes('B') ? 'up' : (grade.includes('D') || grade.includes('F') ? 'down' : '');
    const getSentimentColor = (sentiment) => sentiment === 'Bullish' ? 'up' : (sentiment === 'Bearish' ? 'down' : '');

    // Render Tabbed UI skeleton structure
    Elements.resultContainer.innerHTML = `
      <h2 class="section-heading" style="margin-top: 10px; border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
        ${data.Company_Name} <span style="color:var(--text-muted);font-weight:600;">(${data.Ticker})</span>
      </h2>
      
      <div class="tabs-container">
        <button class="tab-button active" data-tab="sentiment">Sentiment Analysis</button>
        <button class="tab-button" data-tab="price">Price Forecast</button>
        <button class="tab-button" data-tab="dividend">Dividend Forecast</button>
      </div>

      <!-- Tab Content Panels -->
      <div id="tab-sentiment" class="tab-content active">
        <div class="dashboard-grid" style="margin-top: 10px; margin-bottom: 16px;">
          ${this._card("Grade", data.Stock_Grade, getGradeColor(data.Stock_Grade))}
          ${this._card("General Sentiment", data.News_Sentiment, getSentimentColor(data.News_Sentiment))}
        </div>
        <div style="background: rgba(var(--brand-rgb), 0.05); border: 1px solid rgba(var(--brand-rgb), 0.2); border-left: 4px solid rgba(var(--brand-rgb), 1); padding: 16px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; line-height: 1.6;">
          <strong style="color: var(--text-main); display: block; margin-bottom: 6px;">AI Sentiment Reasoning:</strong>
          ${this._buildReasoningHTML(data.AI_Reasoning)}
        </div>
      </div>

      <div id="tab-price" class="tab-content">
        ${this._buildPriceMetrics(data)}
        ${this._buildPriceChartsHTML()}
        ${this._buildUnifiedTable("Closed Stock Price History & Forecast Data with Expected Range", "Trading Date", "Historical Price", "Projected Price", priceRows)}
      </div>

      <div id="tab-dividend" class="tab-content">
        ${this._buildDividendMetrics(data, hasDividends)}
        ${this._buildDividendChartHTML(hasDividends)}
        ${hasDividends ? this._buildUnifiedTable("Dividend Payout History & Forecast Data with Expected Range", "Ex-Dividend Date", "Historical Payout", "Projected Payout", divRows) : ''}
      </div>
    `;

    // Initialize Interactive Tab Controllers
    const tabButtons = Elements.resultContainer.querySelectorAll('.tab-button');
    const tabContents = Elements.resultContainer.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        const targetTab = button.getAttribute('data-tab');

        // Toggle Active Button Styling state
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        // Toggle Visibility Panels
        tabContents.forEach(content => content.classList.remove('active'));
        const activeContent = Elements.resultContainer.querySelector(`#tab-${targetTab}`);
        activeContent.classList.add('active');

        // Force Chart engine layout checks upon switching panels to guarantee smooth layout shifts
        if (targetTab === 'price' && ChartManager.instances.price) {
          ChartManager.instances.price.resize();
          ChartManager.instances.nav.resize();
        } else if (targetTab === 'dividend' && ChartManager.instances.dividend) {
          ChartManager.instances.dividend.resize();
        }
      });
    });

    if (data.Chart_History) setTimeout(() => ChartManager.renderAll(data), 150);
  },

  _buildReasoningHTML(reasoning) {
    if (!reasoning || typeof reasoning === 'string') {
      return `<span style="color: var(--text-muted);">${reasoning || 'No recent data available.'}</span>`;
    }

    let html = '';

    // Render News Sentiment
    if (reasoning.news) {
      const generateNewsHTML = (h) => {
        // Safe string escaping for HTML injection
        const safeTitle = (h.title || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        const safeSummary = (h.summary || 'No summary available.').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        const safePublisher = (h.publisher || 'Unknown Publisher').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        const safeUrl = (h.url || '#').replace(/"/g, '&quot;');

        return `<li style="margin-bottom: 6px;">
            <span class="news-item-link" 
                  onclick="NewsModal.open('${safeTitle}', '${safeSummary}', '${safePublisher}', '${safeUrl}')">
                "${h.title || h}"
            </span>
            <a href="${safeUrl}" target="_blank" class="news-export-inline" title="Read external article">
                <svg class="export-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
            </a>
        </li>`;
      };

      if (reasoning.news.positive && reasoning.news.positive.length > 0) {
        const items = reasoning.news.positive.map(generateNewsHTML).join('');
        html += `<div style="margin-bottom: 12px;"><strong style="color: var(--brand-success);">Positive Press:</strong><ul style="margin-top: 4px; padding-left: 24px; list-style-type: disc;">${items}</ul></div>`;
      }
      if (reasoning.news.negative && reasoning.news.negative.length > 0) {
        const items = reasoning.news.negative.map(generateNewsHTML).join('');
        html += `<div style="margin-bottom: 12px;"><strong style="color: var(--brand-danger);">Negative Press:</strong><ul style="margin-top: 4px; padding-left: 24px; list-style-type: disc;">${items}</ul></div>`;
      }
      if (reasoning.news.neutral) {
        html += `<div style="margin-bottom: 12px;"><span style="color: var(--text-muted);">${reasoning.news.neutral}</span></div>`;
      }
    }

    // Render Fundamental Catalysts
    if (reasoning.fundamentals) {
      if (reasoning.fundamentals.positive && reasoning.fundamentals.positive.length > 0) {
        const items = reasoning.fundamentals.positive.map(f => `<li style="margin-bottom: 6px;">${f}</li>`).join('');
        html += `<div style="margin-top: 16px;"><strong style="color: var(--brand-success);">General Strengths:</strong><ul style="margin-top: 6px; padding-left: 24px; list-style-type: disc;">${items}</ul></div>`;
      }
      if (reasoning.fundamentals.negative && reasoning.fundamentals.negative.length > 0) {
        const items = reasoning.fundamentals.negative.map(f => `<li style="margin-bottom: 6px;">${f}</li>`).join('');
        html += `<div style="margin-top: 16px;"><strong style="color: var(--brand-danger);">General Risks:</strong><ul style="margin-top: 6px; padding-left: 24px; list-style-type: disc;">${items}</ul></div>`;
      }
    }

    // Render ETF Holdings with Progress Bars (Mockup Layout)
    if (reasoning.etf_holdings && reasoning.etf_holdings.length > 0) {
      const items = reasoning.etf_holdings.map((h, index) => {
        const nameDisplay = h.name !== h.symbol ? `${h.name} (${h.symbol})` : h.symbol;
        const pctValue = parseFloat(h.weight) || 0;
        
        return `
          <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 13px;">
              <span style="color: var(--text-main); font-weight: 500;">
                <span style="color: var(--text-muted); margin-right: 4px;">${index + 1}.</span> ${nameDisplay}
              </span>
              <span style="color: var(--text-main); font-weight: 600; font-family: monospace;">${h.weight || '0.00%'}</span>
            </div>
            <div style="width: 100%; height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden;">
              <div style="width: ${pctValue}%; height: 100%; background: var(--brand-primary); border-radius: 3px; transition: width 0.4s ease;"></div>
            </div>
          </div>
        `;
      }).join('');
      
      html += `
        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(var(--brand-rgb), 0.15);">
          <strong style="color: var(--text-main); display: block; margin-bottom: 14px;">Top 10 Fund Holdings by Weight:</strong>
          <div style="display: flex; flex-direction: column;">${items}</div>
        </div>
      `;
    }

    // Render ETF Sectors with Progress Bars (Mockup Layout)
    if (reasoning.etf_sectors && reasoning.etf_sectors.length > 0) {
      const items = reasoning.etf_sectors.map(s => {
        const pctValue = parseFloat(s.weight) || 0;
        
        return `
          <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 13px;">
              <span style="color: var(--text-main); font-weight: 500;">${s.sector}</span>
              <span style="color: var(--text-main); font-weight: 600; font-family: monospace;">${s.weight || '0.00%'}</span>
            </div>
            <div style="width: 100%; height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden;">
              <div style="width: ${pctValue}%; height: 100%; background: linear-gradient(90deg, var(--brand-primary), rgba(var(--brand-rgb), 0.7)); border-radius: 3px; transition: width 0.4s ease;"></div>
            </div>
          </div>
        `;
      }).join('');
      
      html += `
        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(var(--brand-rgb), 0.15);">
          <strong style="color: var(--text-main); display: block; margin-bottom: 14px;">Economic Sector Exposure:</strong>
          <div style="display: flex; flex-direction: column;">${items}</div>
        </div>
      `;
    }

    return html || '<span style="color: var(--text-muted);">No recent data available.</span>';
  },

  _largeCard(title, dateStr, dir, dirConf, amtTitle, amt, amtLower, amtUpper, dirLabel = "Predicted Direction (vs. Last Recorded Price)") {
    const isUp = dir === "Up";
    const dirClass = isUp ? "pill-up" : "pill-down";
    const arrowIcon = isUp ? '↑' : '↓';

    return `
      <div class="premium-horizon-card">
        <div class="horizon-header">
          <div style="display: flex; align-items: center; gap: 8px;">
            <h4 class="horizon-title">${title}</h4>
          </div>
          <div class="horizon-date-badge">${Utils.formatDate(dateStr)}</div>
        </div>
        <div class="horizon-body">
          <div class="stat-box">
            <span class="stat-label">${dirLabel}</span>
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
              <span class="direction-pill ${dirClass}">${arrowIcon} ${dir}</span>
              <span class="conf-badge">${dirConf}% Conf.</span>
            </div>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-box">
            <span class="stat-label">${amtTitle}</span>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">
              <span class="stat-val">${Utils.formatMoney(amt)}</span>
              <span class="conf-badge">Range: ${Utils.formatMoney(amtLower)} &ndash; ${Utils.formatMoney(amtUpper)}</span>
            </div>
          </div>
        </div>
      </div>
    `;
  },

  _buildPriceMetrics(data) {
    const f = data.Price_Forecasts;
    if (!f || Object.keys(f).length === 0) {
      return '<div class="metric-card" style="text-align: center; color: var(--text-muted); font-style: italic; margin-top: 20px; margin-bottom: 32px;">Not enough closed stock price data to generate reliable forecasts.</div>';
    }
    
    const isCrypto = data.Chart_Future_Dates.length === 365;
    const d1 = data.Chart_Future_Dates[0];
    const d2 = data.Chart_Future_Dates[isCrypto ? 6 : 4];
    const d3 = data.Chart_Future_Dates[isCrypto ? 29 : 20];
    const d4 = data.Chart_Future_Dates[isCrypto ? 89 : 62];
    const d5 = data.Chart_Future_Dates[isCrypto ? 179 : 125];
    const d6 = data.Chart_Future_Dates[isCrypto ? 269 : 188];
    const d7 = data.Chart_Future_Dates[data.Chart_Future_Dates.length - 1];

    return `
      <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 20px; margin-bottom: 32px;">
        ${this._largeCard("Next-Day Metrics", d1, f.Next_Day.Direction, f.Next_Day.Direction_Confidence, "Forecasted Close", f.Next_Day.Amount, f.Next_Day.Amount_Lower, f.Next_Day.Amount_Upper)}
        ${this._largeCard("Next-Week Metrics", d2, f.Next_Week.Direction, f.Next_Week.Direction_Confidence, "Forecasted Close", f.Next_Week.Amount, f.Next_Week.Amount_Lower, f.Next_Week.Amount_Upper)}
        ${this._largeCard("Next-Month Metrics", d3, f.Next_Month.Direction, f.Next_Month.Direction_Confidence, "Forecasted Close", f.Next_Month.Amount, f.Next_Month.Amount_Lower, f.Next_Month.Amount_Upper)}
        ${this._largeCard("Next-3-Months Metrics", d4, f.Next_3_Months.Direction, f.Next_3_Months.Direction_Confidence, "Forecasted Close", f.Next_3_Months.Amount, f.Next_3_Months.Amount_Lower, f.Next_3_Months.Amount_Upper)}
        ${this._largeCard("Next-6-Months Metrics", d5, f.Next_6_Months.Direction, f.Next_6_Months.Direction_Confidence, "Forecasted Close", f.Next_6_Months.Amount, f.Next_6_Months.Amount_Lower, f.Next_6_Months.Amount_Upper)}
        ${this._largeCard("Next-9-Months Metrics", d6, f.Next_9_Months.Direction, f.Next_9_Months.Direction_Confidence, "Forecasted Close", f.Next_9_Months.Amount, f.Next_9_Months.Amount_Lower, f.Next_9_Months.Amount_Upper)}
        ${this._largeCard("Next-Year Metrics", d7, f.Next_Year.Direction, f.Next_Year.Direction_Confidence, "Forecasted Close", f.Next_Year.Amount, f.Next_Year.Amount_Lower, f.Next_Year.Amount_Upper)}
      </div>
    `;
  },

  _buildDividendMetrics(data, hasDividends) {
    if (!hasDividends) {
      return `<div class="metric-card" style="text-align: center; color: var(--text-muted); font-style: italic; margin-top: 20px; margin-bottom: 32px;">This publicly traded asset does not currently pay dividends.</div>`;
    }
    
    if (!data.Div_Forecasts || Object.keys(data.Div_Forecasts).length === 0) {
      return `<div class="metric-card" style="text-align: center; color: var(--text-muted); font-style: italic; margin-top: 20px; margin-bottom: 32px;">Not enough historical dividend data to generate reliable forecasts.</div>`;
    }
    
    const f = data.Div_Forecasts;
    const dates = data.Div_Future_Dates;
    const d1 = dates[0];
    const d2 = dates.length > 1 ? dates[1] : 'N/A';
    const d3 = dates.length > 2 ? dates[2] : 'N/A';
    const d4 = dates.length > 3 ? dates[3] : 'N/A';
    const d5 = dates.length > 4 ? dates[4] : 'N/A';

    return `
      <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 20px; margin-bottom: 32px;">
        ${this._largeCard("Next Projected Payout", d1, f.Next_Payout.Direction, f.Next_Payout.Direction_Confidence, "Forecasted Payout", f.Next_Payout.Amount, f.Next_Payout.Amount_Lower, f.Next_Payout.Amount_Upper, "Predicted Direction (vs. Last Payout)")}
        ${this._largeCard("2nd Projected Payout", d2, f.Payout_2.Direction, f.Payout_2.Direction_Confidence, "Forecasted Payout", f.Payout_2.Amount, f.Payout_2.Amount_Lower, f.Payout_2.Amount_Upper, "Predicted Direction (vs. Last Payout)")}
        ${this._largeCard("3rd Projected Payout", d3, f.Payout_3.Direction, f.Payout_3.Direction_Confidence, "Forecasted Payout", f.Payout_3.Amount, f.Payout_3.Amount_Lower, f.Payout_3.Amount_Upper, "Predicted Direction (vs. Last Payout)")}
        ${this._largeCard("4th Projected Payout", d4, f.Payout_4.Direction, f.Payout_4.Direction_Confidence, "Forecasted Payout", f.Payout_4.Amount, f.Payout_4.Amount_Lower, f.Payout_4.Amount_Upper, "Predicted Direction (vs. Last Payout)")}
        ${this._largeCard("5th Projected Payout", d5, f.Payout_5.Direction, f.Payout_5.Direction_Confidence, "Forecasted Payout", f.Payout_5.Amount, f.Payout_5.Amount_Lower, f.Payout_5.Amount_Upper, "Predicted Direction (vs. Last Payout)")}
      </div>
    `;
  },

  _buildUnifiedTable(title, dateHeader, histHeader, projHeader, rows) {
    if (!rows || !rows.length) return '';
    const hasProj = rows.some(r => r.proj !== null && r.proj !== undefined);
    const finalTitle = hasProj ? title : (title.includes("Price") ? "Closed Stock Price History" : "Dividend Payout History");
    
    let html = '';
    for (let r of rows) {
      const histStr = (r.hist !== null && r.hist !== undefined) ? `<strong style="color:var(--chart-history);">${Utils.formatMoney(r.hist)}</strong>` : '&ndash;';
      if (hasProj) {
        const projStr = (r.proj !== null && r.proj !== undefined) ? `<strong style="color:rgba(var(--brand-rgb), 1);">${Utils.formatMoney(r.proj)}</strong>` : '&ndash;';
        const ciStr = (r.lower !== null && r.upper !== null && r.lower !== undefined && r.upper !== undefined) ? `${Utils.formatMoney(r.lower)} &ndash; ${Utils.formatMoney(r.upper)}` : '&ndash;';
        html += `<tr><td>${Utils.formatDate(r.date)}</td><td>${histStr}</td><td>${projStr}</td><td style="color:var(--text-muted);font-size:13px;">${ciStr}</td></tr>`;
      } else {
        html += `<tr><td>${Utils.formatDate(r.date)}</td><td>${histStr}</td></tr>`;
      }
    }
    
    const theadHtml = hasProj ? `<th>${dateHeader}</th><th>${histHeader}</th><th>${projHeader}</th><th>Expected Range</th>` : `<th>${dateHeader}</th><th>${histHeader}</th>`;
    
    return `
      <h3 class="subsection-heading">${finalTitle}</h3>
      <div class="table-wrapper"><table class="glass-table">
        <thead><tr>${theadHtml}</tr></thead>
        <tbody>${html}</tbody>
      </table></div>`;
  },

  _buildPriceChartsHTML() {
    return `
      <div class="chart-box" style="position:relative;margin-top:32px;margin-bottom:16px;">
        <canvas id="priceChart"></canvas>
      </div>
      <div style="position:relative;width:100%;margin-bottom:40px;display:flex;flex-direction:column;gap:12px;">
        <div style="display:flex;justify-content:flex-end;"><button id="navResetBtn" class="glass-btn-small">↺ Reset Timeline</button></div>
        <div id="navWrapper" class="nav-wrapper-custom">
          <canvas id="navChart" style="width:100%;height:100%;display:block;border-radius:6px;"></canvas>
          <div id="navLeft" class="nav-overlay-custom" style="left:0; border-radius:6px 0 0 6px;"></div>
          <div id="navRight" class="nav-overlay-custom" style="right:0; border-radius:0 6px 6px 0;"></div>
          <div id="navHandleL" class="nav-handle-custom"><div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;"><div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div><div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div></div></div>
          <div id="navHandleR" class="nav-handle-custom"><div style="display:flex;flex-direction:column;gap:3px;pointer-events:none;"><div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div><div style="width:2px;height:10px;background:rgba(255,255,255,0.8);border-radius:1px;"></div></div></div>
        </div>
      </div>`;
  },

  _buildDividendChartHTML(hasDividends) {
    return `
      <div style="margin-bottom:24px;">
        <div class="chart-box" id="dividendChartBox" style="position:relative;">
          <canvas id="dividendChart" style="display:${hasDividends ? 'block' : 'none'}"></canvas>
          ${!hasDividends ? `
          <div id="noDividendOverlay" style="position:absolute;inset:0;background:var(--card-bg);backdrop-filter:blur(4px);border-radius:16px;overflow:hidden;">
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
              <span style="font-size:36px;font-weight:800;color:var(--table-border);letter-spacing:2px;white-space:nowrap;transform:rotate(-15deg);font-family:Inter,sans-serif;user-select:none;">
                NO DIVIDEND DATA
              </span>
            </div>
          </div>` : ''}
        </div>
      </div>`;
  },

  _card(label, value, extraClass = '') {
    return `<div class="metric-card"><span class="metric-label">${label}</span><span class="metric-value ${extraClass}">${value}</span></div>`;
  }
};

// Chart Manager
const ChartManager = {
  instances: { price: null, nav: null, dividend: null },
  viewState: { min: 0, max: 0 },

  destroyAll() {
    if (this.instances.price) this.instances.price.destroy();
    if (this.instances.nav) this.instances.nav.destroy();
    if (this.instances.dividend) this.instances.dividend.destroy();
  },

  renderAll(data) {
    this.destroyAll();
    const colors = Utils.getThemeColors();
    const hist = data.Chart_History;

    // Build Unified Price Maps for Chart Coordinates
    const historyMap = new Map();
    hist.dates.forEach((d, i) => historyMap.set(d, hist.prices[i]));
    const historyCoords = Array.from(historyMap, ([x, y]) => ({ x, y })).sort((a, b) => new Date(a.x) - new Date(b.x));
    
    const anchorDate = historyCoords[historyCoords.length - 1].x;
    const unifiedMap = new Map();
    
    if (data.Train_Fit_Dates) {
      data.Train_Fit_Dates.forEach((d, i) => {
        if (d !== anchorDate) {
          unifiedMap.set(d, data.Train_Fit_Prices[i]);
        }
      });
    }
    
    // Anchor the projected line to the most recent historical close to prevent rendering gaps
    const projectedToday = data.Train_Fit_Prices?.length ? data.Train_Fit_Prices[data.Train_Fit_Prices.length - 1] : historyCoords[historyCoords.length - 1].y;
    unifiedMap.set(anchorDate, projectedToday);
    
    data.Chart_Future_Dates.forEach((d, i) => unifiedMap.set(d, data.Chart_Future_Prices[i]));
    const unifiedCoords = Array.from(unifiedMap, ([x, y]) => ({ x, y })).sort((a, b) => new Date(a.x) - new Date(b.x));

    // Expected Range bounds begin expanding outward from the anchor date
    const upperCoords = [{x: anchorDate, y: projectedToday}, ...data.Chart_Future_Dates.map((d, i) => ({x: d, y: data.Chart_Future_Upper[i]}))];
    const lowerCoords = [{x: anchorDate, y: projectedToday}, ...data.Chart_Future_Dates.map((d, i) => ({x: d, y: data.Chart_Future_Lower[i]}))];

    // Setup absolute timeline boundaries for the navigation slider
    const allDates = [...hist.dates, ...data.Chart_Future_Dates];
    this.viewState.absoluteMin = new Date(allDates[0]).getTime();
    this.viewState.absoluteMax = new Date(allDates[allDates.length - 1]).getTime();
    
    // Add visual right-padding for fallbacks to prevent clipping the final point
    if (!data.Chart_Future_Dates || data.Chart_Future_Dates.length === 0) {
      this.viewState.absoluteMax += 14 * 24 * 60 * 60 * 1000;
    }

    if (this.viewState.min === 0 || this.viewState.max === 0) {
      this.viewState.min = this.viewState.absoluteMin;
      this.viewState.max = this.viewState.absoluteMax;
    }

    this._renderPriceChart(historyCoords, unifiedCoords, upperCoords, lowerCoords, anchorDate, colors, data);
    this._renderNavChart(historyCoords, unifiedCoords, colors);
    this._setupNavSlider();
    
    if (hist.dividend_dates && hist.dividend_dates.length) {
      this._renderDividendChart(data, hist, colors);
    }
  },

  _renderPriceChart(hist, proj, upper, lower, anchorDate, colors, data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const labelTextColor = isDark ? '#0f172a' : '#ffffff';

    this.instances.price = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          { label: 'Historical Stock Prices', data: hist, backgroundColor: colors.history, borderColor: colors.history, borderWidth: 1.5, pointRadius: 2, order: 1 },
          { label: 'Projected Stock Prices', data: proj, borderColor: `rgba(${colors.brandRGB}, 1)`, backgroundColor: `rgba(${colors.brandRGB}, 0.4)`, borderWidth: 2, pointRadius: 0, tension: 0.2, order: 0 },
          { label: 'Upper Bound', data: upper, backgroundColor: `rgba(${colors.brandRGB}, 0.15)`, borderColor: 'transparent', pointRadius: 0, pointHoverRadius: 0, pointHitRadius: 0, fill: '+1', tension: 0.3, order: 2 },
          { label: 'Lower Bound', data: lower, borderColor: 'transparent', pointRadius: 0, pointHoverRadius: 0, pointHitRadius: 0, fill: false, tension: 0.3, order: 2 },
        ]
      },
      options: {
        color: colors.text,
        responsive: true, maintainAspectRatio: false, animation: false, interaction: { intersect: false, mode: 'x' },
        scales: {
          x: { 
            type: 'time', min: this.viewState.min, max: this.viewState.max, 
            time: { unit: 'month', tooltipFormat: 'MMM d, yyyy' }, 
            grid: { color: colors.grid }, 
            ticks: { color: colors.text, maxRotation: 45, minRotation: 45, font: { size: 11 } } 
          },
          y: { 
            grid: { color: colors.grid }, 
            ticks: { color: colors.text, font: { size: 11 }, callback: v => `$${v.toLocaleString()}` } 
          }
        },
        plugins: {
          title: { display: true, text: (data.Chart_Future_Dates && data.Chart_Future_Dates.length) ? 'Closed Stock Price History & Forecast Trends with Expected Range' : 'Closed Stock Price History', color: colors.text, font: { size: 14, weight: '600' }, padding: { bottom: 16 } },
          legend: { 
            display: (data.Chart_Future_Dates && data.Chart_Future_Dates.length) > 0,
            labels: { 
              color: colors.text, 
              usePointStyle: true,
              generateLabels: () => {
                const items = [
                  { text: 'Historical Stock Prices', fillStyle: colors.history, strokeStyle: 'transparent', fontColor: colors.text },
                  { text: 'Projected Stock Prices', fillStyle: `rgba(${colors.brandRGB}, 0.4)`, strokeStyle: `rgba(${colors.brandRGB}, 1)`, fontColor: colors.text }
                ];
                if (data.Chart_Future_Dates && data.Chart_Future_Dates.length) {
                  items.push({ text: 'Expected Range', fillStyle: `rgba(${colors.brandRGB}, 0.15)`, strokeStyle: 'transparent', fontColor: colors.text });
                }
                return items;
              }
            } 
          },
          annotation: { 
            annotations: { 
              todayLine: { 
                type: 'line', xMin: anchorDate, xMax: anchorDate, borderColor: colors.text, borderDash: [5, 4], 
                label: { display: true, content: 'Today', position: 'start', font: { size: 10 }, backgroundColor: colors.text, color: labelTextColor } 
              } 
            } 
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
                const price = ctx.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                if (ctx.dataset.label !== 'Projected Stock Prices') {
                  return `${ctx.dataset.label}: $${price}`;
                }
                const hoverDate = ctx.raw.x;
                const ciIndex = data.Chart_Future_Dates.indexOf(hoverDate);
                if (ciIndex !== -1) {
                  const lo = data.Chart_Future_Lower[ciIndex].toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                  const hi = data.Chart_Future_Upper[ciIndex].toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                  return [
                    `Projected Stock Price: $${price}`,
                    `Expected Range: $${lo} \u2013 $${hi}`,
                  ];
                }
                return `Projected Stock Price: $${price}`;
              }
            }
          }
        }
      }
    });
  },

  _renderNavChart(hist, proj, colors) {
    const ctx = document.getElementById('navChart').getContext('2d');
    this.instances.nav = new Chart(ctx, {
      type: 'line',
      data: { datasets: [{ data: hist, backgroundColor: colors.history, pointRadius: 1, order: 1 }, { data: proj, borderColor: `rgba(${colors.brandRGB}, 1)`, borderWidth: 1.5, pointRadius: 0, order: 0 }] },
      options: { 
        responsive: true, maintainAspectRatio: false, animation: false, 
        layout: { padding: { top: 10, bottom: 10 } },
        scales: { x: { type: 'time', display: false }, y: { display: false } }, 
        plugins: { legend: { display: false }, tooltip: { enabled: false } } 
      }
    });
  },

  _setupNavSlider() {
    const wrapper = document.getElementById('navWrapper');
    const left = document.getElementById('navLeft');
    const right = document.getElementById('navRight');
    const hL = document.getElementById('navHandleL');
    const hR = document.getElementById('navHandleR');
    
    const minTs = this.viewState.absoluteMin;
    const maxTs = this.viewState.absoluteMax;
    const MIN_WINDOW = 86400000 * 7; 
    let dragMode = null, startX = 0, startMin = 0, startMax = 0;

    const updateUI = () => {
      const w = wrapper.getBoundingClientRect().width;
      if (w === 0) return; 
      const lPx = ((this.viewState.min - minTs) / (maxTs - minTs)) * w;
      const rPx = ((this.viewState.max - minTs) / (maxTs - minTs)) * w;
      
      left.style.width = `${lPx}px`; 
      right.style.width = `${w - rPx}px`;
      hL.style.left = `${lPx - 7}px`; 
      hR.style.left = `${rPx - 7}px`;
      
      if (this.instances.price) {
        this.instances.price.options.scales.x.min = this.viewState.min;
        this.instances.price.options.scales.x.max = this.viewState.max;
        this.instances.price.update('none');
      }
    };

    const onMove = (e) => {
      if (!dragMode) return;
      const cx = e.clientX ?? (e.touches ? e.touches[0].clientX : 0);
      const w = wrapper.getBoundingClientRect().width;
      const dTs = ((cx - startX) / w) * (maxTs - minTs);
      const span = startMax - startMin;
      
      if (dragMode === 'pan') {
        this.viewState.min = Math.max(minTs, Math.min(startMin + dTs, maxTs - span));
        this.viewState.max = this.viewState.min + span;
      } else if (dragMode === 'left') {
        this.viewState.min = Math.max(minTs, Math.min(startMin + dTs, this.viewState.max - MIN_WINDOW));
      } else if (dragMode === 'right') {
        this.viewState.max = Math.min(maxTs, Math.max(startMax + dTs, this.viewState.min + MIN_WINDOW));
      }
      updateUI();
    };

    const onStart = (e, mode) => { 
      dragMode = mode; 
      startX = e.clientX ?? (e.touches ? e.touches[0].clientX : 0); 
      startMin = this.viewState.min; 
      startMax = this.viewState.max; 
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
    const onStartPan = (e) => { if (e.target !== hL && e.target !== hR) onStart(e, 'pan'); };

    if (window.navListeners) {
      document.removeEventListener('mousemove', window.navListeners.onMove);
      document.removeEventListener('mouseup', window.navListeners.onEnd);
      
      const oldHL = document.getElementById('navHandleL');
      const oldHR = document.getElementById('navHandleR');
      const oldWrap = document.getElementById('navWrapper');
      
      if (oldHL) oldHL.removeEventListener('mousedown', window.navListeners.onStartL);
      if (oldHR) oldHR.removeEventListener('mousedown', window.navListeners.onStartR);
      if (oldWrap) oldWrap.removeEventListener('mousedown', window.navListeners.onStartPan);

      if (window.navListeners.resizeObserver) window.navListeners.resizeObserver.disconnect();
    }

    hL.addEventListener('mousedown', onStartL);
    hR.addEventListener('mousedown', onStartR);
    wrapper.addEventListener('mousedown', onStartPan);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onEnd);
    
    const resizeObserver = new ResizeObserver(() => {
      window.requestAnimationFrame(() => updateUI());
    });
    resizeObserver.observe(wrapper);

    window.navListeners = { onMove, onEnd, onStartL, onStartR, onStartPan, resizeObserver };

    document.getElementById('navResetBtn').addEventListener('click', () => { 
      this.viewState.min = minTs; 
      this.viewState.max = maxTs; 
      updateUI(); 
    });

    setTimeout(updateUI, 150);
  },

  _renderDividendChart(data, hist, colors) {
    const map = new Map();
    
    hist.dividend_dates.forEach((d, i) => map.set(d, { histAmt: hist.dividend_amounts[i], projAmt: null, ciUpper: null, ciLower: null, est: false }));
    
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
      map.set(d, { histAmt: null, projAmt: data.Div_Future_Amounts[i], ciUpper: data.Div_Future_Upper[i], ciLower: data.Div_Future_Lower[i], est: true });
    });

    const sorted = Array.from(map.entries()).sort((a, b) => new Date(a[0]) - new Date(b[0]));
    const finalLabels = sorted.map(i => i[1].est ? `${Utils.formatDate(i[0])} (Est.)` : Utils.formatDate(i[0]));
    const histData = sorted.map(i => i[1].histAmt);
    const projData = sorted.map(i => i[1].projAmt);
    const ciUpper = sorted.map(i => i[1].ciUpper);
    const ciLower = sorted.map(i => i[1].ciLower);
    
    const floatingCIBounds = ciUpper.map((u, i) => u !== null ? [ciLower[i], u] : null);

    const ctx = document.getElementById('dividendChart').getContext('2d');
    this.instances.dividend = new Chart(ctx, {
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
            order: 3
          },
          { 
            label: 'Historical Payout',
            data: histData, 
            backgroundColor: colors.history, 
            grouped: false,
            barPercentage: 0.8,
            categoryPercentage: 0.8,
            borderRadius: 4,
            order: 2 
          },
          { 
            label: 'Projected Payout',
            data: projData, 
            backgroundColor: `rgba(${colors.brandRGB}, 0.8)`, 
            grouped: false,
            barPercentage: 0.4,
            categoryPercentage: 0.8,
            borderRadius: 4,
            order: 1 
          }
        ]
      },
      options: {
        color: colors.text, responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: { 
          x: { grid: { display: false }, ticks: { color: colors.text, maxRotation: 45, minRotation: 45, font: { size: 10 } } }, 
          y: { grid: { color: colors.grid }, ticks: { color: colors.text, font: { size: 11 }, callback: v => `$${v.toFixed(2)}` } } 
        },
        plugins: {
          title: { display: true, text: projData.some(p => p !== null) ? 'Dividend Payout History & Forecast Trends with Expected Range' : 'Dividend Payout History', color: colors.text, font: { size: 14, weight: '600' }, padding: { bottom: 16 } },
          legend: { 
            display: projData.some(p => p !== null),
            labels: {
              color: colors.text,
              usePointStyle: true,
              generateLabels: () => {
                const items = [
                  { text: 'Historical Payout', fillStyle: colors.history, strokeStyle: 'transparent', fontColor: colors.text }
                ];
                if (data.Train_Fit_Div_Dates || data.Div_Future_Dates) {
                  items.push({ text: 'Projected Payout', fillStyle: `rgba(${colors.brandRGB}, 0.8)`, strokeStyle: 'transparent', fontColor: colors.text });
                }
                if (data.Div_Future_Dates && data.Div_Future_Dates.length) {
                  items.push({ text: 'Expected Range', fillStyle: `rgba(${colors.brandRGB}, 0.15)`, strokeStyle: 'transparent', fontColor: colors.text });
                }
                return items;
              }
            }
          },
          tooltip: {
            filter: function (tooltipItem) {
              return tooltipItem.datasetIndex !== 0; 
            },
            callbacks: {
              label: (ctx) => {
                const amount = ctx.parsed.y;
                if (amount === null) return null; 

                const i = ctx.dataIndex;
                const isHistorical = ctx.datasetIndex === 1;
                const isProjected = ctx.datasetIndex === 2;

                if (isHistorical) {
                  return `Historical Dividend Payout: $${amount.toFixed(2)}`;
                }
                
                if (isProjected) {
                  if (ciUpper[i] !== null && ciUpper[i] !== undefined) {
                    return [
                      `Projected Dividend Payout: $${amount.toFixed(2)}`,
                      `Expected Range: $${ciLower[i].toFixed(2)} \u2013 $${ciUpper[i].toFixed(2)}`
                    ];
                  }
                  return `Projected Dividend Payout: $${amount.toFixed(2)}`;
                }
              }
            }
          }
        }
      }
    });
  }
};

// Primary Application Object
const App = {
  init() {
    // Map DOM elements
    Elements.predictBtn = document.getElementById('predictBtn');
    Elements.tickerInput = document.getElementById('tickerInput');
    Elements.clearSearchBtn = document.getElementById('clearSearchBtn');
    Elements.autocompleteResults = document.getElementById('autocompleteResults');
    Elements.resultContainer = document.getElementById('resultContainer');
    Elements.errorContainer = document.getElementById('errorContainer');
    Elements.loader = document.getElementById('loader');
    Elements.themeToggleBtn = document.getElementById('themeToggle');
    Elements.themeIcon = Elements.themeToggleBtn.querySelector('.theme-icon');
    Elements.themeLabel = Elements.themeToggleBtn.querySelector('.theme-label');

    // Initialize Sub-Systems
    ThemeManager.init();
    SearchManager.init();

    // Bind Primary Events
    Elements.predictBtn.addEventListener('click', () => this.fetchPrediction());
    Elements.tickerInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.fetchPrediction(); } });
  },

  async fetchPrediction() {
    const ticker = Elements.tickerInput.value.trim().toUpperCase();
    if (!ticker) { UIManager.showError('Please enter a ticker symbol.'); return; }

    SearchManager.clearSearch(); // Resets UI/Dropdowns
    Elements.tickerInput.value = ticker; // Put ticker back after clear
    
    State.isSearching = true;
    UIManager.setLoading(true);

    // Reset SSE Progress UI
    const progressFill = document.getElementById('progress-bar-fill');
    const progressPercentage = document.getElementById('progress-percentage');
    const stepsContainer = document.getElementById('progress-steps-container');
    if (progressFill) progressFill.style.width = '0%';
    if (progressPercentage) progressPercentage.textContent = '0%';
    if (stepsContainer) stepsContainer.innerHTML = '';
    
    const timers = {};
    let activeStepElement = null;
    let timerInterval = null;

    try {
      const eventSource = new EventSource(`/predict_stream/${ticker}`);
      
      eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        
        if (data.status === 'error') {
            eventSource.close();
            clearInterval(timerInterval);
            UIManager.showError(data.error || 'An unknown error occurred.');
            UIManager.setLoading(false);
            State.isSearching = false;
            return;
        }

        if (data.progress !== undefined) {
            if (progressFill) progressFill.style.width = `${data.progress}%`;
            if (progressPercentage) progressPercentage.textContent = `${data.progress}%`;
        }

        if (data.status === 'processing' && data.step) {
            if (activeStepElement) {
                // Complete previous step
                activeStepElement.classList.remove('active');
                activeStepElement.classList.add('completed');
                const icon = activeStepElement.querySelector('.step-icon');
                if (icon) {
                    icon.className = 'step-icon';
                    icon.textContent = '✓';
                    icon.style.color = 'var(--brand-success)';
                    icon.style.border = 'none';
                    icon.style.animation = 'none';
                }
            }
            
            // Create new step
            const stepId = `step-${Date.now()}`;
            timers[stepId] = 0.0;
            const stepEl = document.createElement('div');
            stepEl.className = 'progress-step active';
            stepEl.innerHTML = `
                <div class="step-left">
                    <span class="step-icon spinner-icon"></span>
                    <span>${data.step}</span>
                </div>
                <span class="step-timer" id="${stepId}">0.0s</span>
            `;
            if (stepsContainer) stepsContainer.appendChild(stepEl);
            activeStepElement = stepEl;
            
            // Start timer
            clearInterval(timerInterval);
            timerInterval = setInterval(() => {
                timers[stepId] += 0.1;
                const timeEl = document.getElementById(stepId);
                if (timeEl) timeEl.textContent = `${timers[stepId].toFixed(1)}s`;
            }, 100);
        }

        if (data.status === 'complete') {
            eventSource.close();
            clearInterval(timerInterval);
            if (activeStepElement) {
                activeStepElement.classList.remove('active');
                activeStepElement.classList.add('completed');
                const icon = activeStepElement.querySelector('.step-icon');
                if (icon) {
                    icon.className = 'step-icon';
                    icon.textContent = '✓';
                    icon.style.color = 'var(--brand-success)';
                    icon.style.border = 'none';
                    icon.style.animation = 'none';
                }
            }

            // Hold on 100% for a moment so the user sees completion
            setTimeout(() => {
                Elements.loader.classList.add('fade-out');
                
                setTimeout(() => {
                    UIManager.setLoading(false); // Instantly hides it natively
                    Elements.loader.classList.remove('fade-out');
                    
                    State.lastFetchedData = data.result;
                    UIManager.renderDashboard(data.result);
                    
                    // Fade in the new dashboard
                    Elements.resultContainer.classList.add('fade-in');
                    setTimeout(() => {
                        Elements.resultContainer.classList.remove('fade-in');
                    }, 500);

                    State.isSearching = false;
                }, 350); // Wait for fadeOut animation to finish
            }, 800); // 800ms hold on 100%
        }
      };
      
      eventSource.onerror = (err) => {
        console.error("SSE Error", err);
        eventSource.close();
        clearInterval(timerInterval);
        UIManager.showError('Connection to server lost. Please try again.');
        UIManager.setLoading(false);
        State.isSearching = false;
      };

    } catch (error) {
      clearInterval(timerInterval);
      UIManager.showError(error.message);
      UIManager.setLoading(false);
      State.isSearching = false;
    }
  }
};

// News Modal Logic
const NewsModal = {
    init() {
        this.overlay = document.getElementById('newsModal');
        this.titleEl = document.getElementById('modalTitle');
        this.publisherEl = document.getElementById('modalPublisher');
        this.summaryEl = document.getElementById('modalSummary');
        this.linkEl = document.getElementById('modalExternalLink');
        this.closeBtn = document.getElementById('closeModalBtn');
        this.contentBox = this.overlay.querySelector('.modal-content');

        this.closeBtn.addEventListener('click', () => this.close());
        this.overlay.addEventListener('click', (e) => {
            // Close if clicking outside the modal content
            if (e.target === this.overlay) this.close();
        });
    },

    open(title, summary, publisher, url) {
        // Decode escaped characters
        const decode = (str) => str.replace(/&quot;/g, '"').replace(/&#39;/g, "'");

        this.titleEl.textContent = decode(title);
        this.summaryEl.textContent = decode(summary);
        this.publisherEl.textContent = decode(publisher);
        this.linkEl.href = url;

        this.overlay.style.display = 'flex';
        this.overlay.classList.remove('modal-fadeOut');
    },

    close() {
        this.overlay.classList.add('modal-fadeOut');
        // Wait for animation to finish before hiding
        setTimeout(() => {
            this.overlay.style.display = 'none';
        }, 300);
    }
};

// Bootstrap Application
document.addEventListener('DOMContentLoaded', () => {
    App.init();
    NewsModal.init();
});