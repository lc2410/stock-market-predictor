document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predictBtn');
    const tickerInput = document.getElementById('tickerInput');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const loader = document.getElementById('loader');

    // NEW: Global variables to hold chart instances
    let priceChartInstance = null;
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

            if (!response.ok) {
                throw new Error(data.error || 'An unknown error occurred.');
            }
            
            displayResult(data);

        } catch (error) {
            showError(error.message);
        } finally {
            loader.style.display = 'none';
        }
    };

    predictBtn.addEventListener('click', handlePrediction);
    tickerInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') handlePrediction();
    });

    function displayResult(data) {
        // 1. Generate the Text Grid (Your existing code)
        let html = `
            <h3>Forecast for ${data.Ticker}</h3>
            <div class="results-grid">
                <h4 class="grid-subtitle">Price Forecast</h4>
                <div class="result-item"><strong>Next Trading Day:</strong></div>
                <div class="result-item">${data.Next_Trading_Day}</div>
                <div class="result-item"><strong>Price Direction:</strong></div>
                <div class="result-item ${data.Price_Predicted.toLowerCase()}">${data.Price_Predicted}</div>
                <div class="result-item"><strong>Price Confidence:</strong></div>
                <div class="result-item">${data['Price_Confidence (%)']}%</div>
                <div class="result-item"><strong>Forecasted Close:</strong></div>
                <div class="result-item">$${data.Forecasted_Close.toFixed(2)}</div>

                <div class="separator"></div>
                
                <h4 class="grid-subtitle">Dividend Forecast</h4>
                <div class="result-item"><strong>Next Dividend Date:</strong></div>
                <div class="result-item">${data.Next_Dividend_Date}</div>
                <div class="result-item"><strong>Dividend Direction:</strong></div>
                <div class="result-item ${data.Div_Predicted.toLowerCase()}">${data.Div_Predicted}</div>
                <div class="result-item"><strong>Dividend Confidence:</strong></div>
                <div class="result-item">${data['Div_Confidence (%)'] === 'N/A' ? 'N/A' : data['Div_Confidence (%)'] + '%'}</div>
                <div class="result-item"><strong>Forecasted Dividend:</strong></div>
                <div class="result-item">${data.Forecasted_Dividend === 'N/A' ? 'N/A' : '$' + data.Forecasted_Dividend.toFixed(2)}</div>
            </div>
        `;

        // 2. NEW: Append Containers for the Charts
        html += `
            <div class="charts-wrapper" style="margin-top: 40px;">
                <div class="chart-container" style="position: relative; height:400px; margin-bottom: 40px;">
                    <canvas id="priceChart"></canvas>
                </div>
                <div class="chart-container" style="position: relative; height:300px;">
                    <canvas id="dividendChart"></canvas>
                </div>
            </div>
        `;

        resultContainer.innerHTML = html;

        // 3. NEW: Call the function to draw the charts
        if (data.Chart_Data) {
            console.log("Rendering charts with data:", data.Chart_Data);
            renderCharts(data);
        }
    }

    // NEW: Function to render Chart.js graphs
    function renderCharts(data) {
        // --- PREPARE DATA ---
        const cData = data.Chart_Data;
        
        // Destroy old charts if they exist
        if (priceChartInstance) priceChartInstance.destroy();
        if (dividendChartInstance) dividendChartInstance.destroy();

        // --- CHART 1: PRICE (History + Prediction) ---
        const ctxPrice = document.getElementById('priceChart').getContext('2d');
        priceChartInstance = new Chart(ctxPrice, {
            type: 'line',
            data: {
                labels: cData.dates, // Includes history + next trading day
                datasets: [{
                    label: 'Stock Price',
                    data: cData.prices,
                    borderColor: '#2563eb', // Blue
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0, 
                    fill: true,
                    // Dotted line for the final prediction segment
                    segment: {
                        borderDash: ctx => ctx.p0DataIndex === cData.prices.length - 2 ? [6, 6] : undefined,
                        borderColor: ctx => ctx.p0DataIndex === cData.prices.length - 2 ? '#dc2626' : '#2563eb'
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    title: { display: true, text: '6-Month Price History & Next Day Forecast' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.parsed.y.toFixed(2);
                                if (context.dataIndex === context.dataset.data.length - 1) {
                                    return `Prediction: $${label}`;
                                }
                                return `$${label}`;
                            }
                        }
                    }
                }
            }
        });

        // --- CHART 2: DIVIDENDS (History + Prediction) ---
        // We manually combine the history list with the single prediction value
        const divLabels = [...cData.dividend_dates];
        const divAmounts = [...cData.dividend_amounts];
        const bgColors = Array(divAmounts.length).fill('#16a34a'); // All Green

        // Only add prediction if it's valid (not "N/A")
        if (data.Forecasted_Dividend !== 'N/A' && data.Next_Dividend_Date !== 'N/A') {
            divLabels.push(data.Next_Dividend_Date + ' (Est)');
            divAmounts.push(data.Forecasted_Dividend);
            bgColors.push('#9ca3af'); // Grey for the prediction
        }

        const ctxDiv = document.getElementById('dividendChart').getContext('2d');
        dividendChartInstance = new Chart(ctxDiv, {
            type: 'bar',
            data: {
                labels: divLabels,
                datasets: [{
                    label: 'Dividend Payout',
                    data: divAmounts,
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: 'Dividend History & Next Projected Payout' }
                }
            }
        });
    }

    function showError(message) {
        errorContainer.textContent = `Error: ${message}`;
    }
    
    function clearMessages() {
        resultContainer.innerHTML = '';
        errorContainer.textContent = '';
    }
});