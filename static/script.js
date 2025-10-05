document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predictBtn');
    const tickerInput = document.getElementById('tickerInput');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const loader = document.getElementById('loader');

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
        const html = `
            <h3>Forecast for ${data.Ticker}</h3>
            <div class="results-grid">
                <h4 class="grid-subtitle">Price Forecast</h4>
                <div class="result-item"><strong>Next Trading Day:</strong></div>
                <div class="result-item">${data.Next_Trading_Day}</div>
                <div class="result-item"><strong>Price Direction:</strong></div>
                <div class="result-item ${data.Price_Predicted.toLowerCase()}">${data.Price_Predicted}</div>
                <div class="result-item"><strong>Price Confidence:</strong></div>
                <div class="result-item">${data['Price_Confidence (%)']}%</div>
                <div class="result-item"><strong>Forecasted Close (per share):</strong></div>
                <div class="result-item">$${data.Forecasted_Close.toFixed(2)}</div>

                <div class="separator"></div>
                
                <h4 class="grid-subtitle">Dividend Forecast</h4>
                <div class="result-item"><strong>Next Dividend Date:</strong></div>
                <div class="result-item">${data.Next_Dividend_Date}</div>
                <div class="result-item"><strong>Dividend Direction:</strong></div>
                <div class="result-item ${data.Div_Predicted.toLowerCase()}">${data.Div_Predicted}</div>
                <div class="result-item"><strong>Dividend Confidence:</strong></div>
                <div class="result-item">${data['Div_Confidence (%)'] === 'N/A' ? 'N/A' : data['Div_Confidence (%)'] + '%'}</div>
                <div class="result-item"><strong>Forecasted Dividend (per share):</strong></div>
                <div class="result-item">${data.Forecasted_Dividend === 'N/A' ? 'N/A' : '$' + data.Forecasted_Dividend.toFixed(2)}</div>
            </div>
        `;
        resultContainer.innerHTML = html;
    }

    function showError(message) {
        errorContainer.textContent = `Error: ${message}`;
    }
    
    function clearMessages() {
        resultContainer.innerHTML = '';
        errorContainer.textContent = '';
    }
});
