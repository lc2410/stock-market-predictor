# Real-Time Stock & Dividend Forecaster

## Overview
Stock trading is a complex process for both short and long-term investors, particularly in highly volatile market conditions. This application is meant to help those who are afraid of getting into the market blindly by delivering real-time market data and predictive insights into publicly traded assets like ETFs, index funds, and/or individual stocks. By combining automated data ingestion, advanced machine learning techniques, and AI sentiment analysis into an intuitive interface, the platform empowers users to confidently evaluate assets and make data-driven investment decisions.

---

## Application Architecture
To ensure enterprise-grade stability, security, and performance, this application relies on a multi-layered architecture running on a dedicated Linux virtual machine (Ubuntu 22.04 LTS).

![Image of Application Architecture](imgs/stock-market-predictor-arch.png)

1. **Client (Front-End):** The user browser loads the HTML, CSS, and JavaScript. The UI is decoupled from the heavy ML processing. It utilizes the native `EventSource` API to maintain an open Server-Sent Events (SSE) connection, ensuring the interface remains highly responsive by streaming a live execution checklist and progress trackers while the models train in the background.
2. **Web Server (Nginx Reverse Proxy):** Nginx acts as the secure front door to the application. It intercepts incoming public HTTP traffic on port 80 and buffers requests to protect the internal server from slow clients or malicious spikes. It safely proxies validated dynamic requests to the internal application layer.
3. **Application Server (Gunicorn WSGI):** Web servers and Python applications speak different protocols. Gunicorn acts as the essential Web Server Gateway Interface (WSGI) translator. It runs as a highly available background `systemd` service on internal port 8000 and manages a pool of worker processes that execute the Flask code in parallel.
4. **API & ML Execution:** The Flask routing layer handles API requests. It proxies Yahoo Finance autocomplete queries to bypass CORS restrictions. It also catches prediction requests to trigger the algorithmic pipeline. This fetches real-time data, dynamically trains the machine learning models, and actively streams the execution progress and final multi-horizon forecast back to the client as structured JSON via SSE.
5. **CI/CD Pipeline:** An automated workflow that triggers on code changes to enforce strict quality control. Before any code reaches the production server, the pipeline executes security vulnerability scans, code linting, automated test suites, and static code analysis, ensuring only stable, validated code is deployed to the Oracle Cloud virtual machine.
6. **Data Source:** The external provider for all live and historical financial data. The backend interfaces with Yahoo Finance to continuously resolve partial ticker symbols for the search autocomplete, and to fetch the extensive historical price and dividend datasets required to train the machine learning models in real-time.

---

## Project File Structure
The repository is organized into cleanly separated domains to maintain strict modularity between the infrastructure, machine learning, API, and client-facing layers.

```text
stock-market-predictor/
├── .github/workflows/
│   └── deploy.yml            # Automated CI/CD pipeline configuration
├── backend/
│   ├── apis/                 # Flask routing and API endpoints
│   ├── models/               # Scikit-learn ML pipeline & data fetching
│   └── tests/                # Pytest unit and integration test suite
├── frontend/
│   ├── scripts/              # Vanilla JS, DOM manipulation, and Chart.js logic
│   ├── styles/               # UI styling and layout
│   ├── tests/                # Playwright End-to-End (E2E) browser tests
│   └── index.html            # Main web interface
├── infra/                    # Terraform Infrastructure as Code (IaC) scripts
├── app.py                    # Flask application bootloader
└── sonar-project.properties  # SonarCloud static analysis configuration
```

---

## Core Features
* **Searching for a Publicly Traded Asset:** A debounced, native search bar that proxies the Yahoo Finance query API through the Flask backend. This provides live, CORS-friendly autocomplete and allows users to search across a vast universe of publicly traded assets, including standard equities, ETFs, Mutual Funds, Cryptocurrencies, Market Indices, and specialized assets like REITs and CEFs.
* **Sentiment Grading Analysis:** The system synthesizes quantitative ML outputs, Wall Street consensus ratings, fundamental metrics (like EPS, Beta, Market Cap, and Yield), and NLP-driven news sentiment to assign an overarching AI Stock Grade (A+ through F) and a General Sentiment (Bullish/Bearish/Neutral). It dynamically adapts its grading criteria based on the asset class, ensuring funds or cryptocurrencies aren't penalized for lacking traditional corporate metrics. For ETFs and Mutual Funds, it also dynamically extracts and visualizes the top 10 holdings and economic sector exposures.
    * **In-App News Reader:** The NLP reasoning block isolates the strongest positive and negative news catalysts driving the asset. Users can choose to instantly open external links directly to the original publisher, or click the headline to trigger a native, glassmorphic modal overlay that presents a clean, localized summary of the article without ever leaving the application.
* **Real-Time Streaming Execution Pipeline:** To ensure the user interface remains responsive while training complex ML models, the application utilizes Server-Sent Events (SSE) to stream a live, step-by-step progress checklist to the frontend, complete with precision micro-timers and smooth CSS transitions.
* **Closed Price Forecasting Summary:** Delivers a clear breakdown of the next trading day's predicted price direction, magnitude, and statistical confidence. This is paired with an interactive Chart.js engine featuring a draggable time-navigator, allowing users to seamlessly pan and zoom across historical market data and view future trajectories bounded by a 95% confidence interval margin of error from as far as 1-year from now. A unified, scrollable data table provides a clean view of trailing-year historical prices alongside the future projections.
* **Dividend Payout Forecasting Summary:** Automatically determines if an asset pays dividends and projects the exact date, direction, and amount of the next payout. It features an interactive bar chart visualizing historical payouts against the projected next five payout cycles in total, complete with explicit 95% margin of error bounds. A dedicated data table organizes these historical and forecasted ex-dividend dates and amounts.
* **Light/Dark Modes:** A native, fully integrated theme manager that allows users to toggle between a clean light mode and a deep dark mode, dynamically updating the CSS variables, UI components, and Chart.js canvases on the fly without requiring a page reload.

---

## The Machine Learning Process
The forecasting engine abandons simple linear models to embrace a holistic, multi-modal approach. By combining dynamic feature engineering, aggressive noise reduction, and a Dual Forecasting Pipeline designed to prevent statistical overfitting, the system simultaneously routes live media data through a specialized NLP Sentiment Analysis layer. This allows the algorithm to synthesize raw price action with real-time market sentiment for a highly calibrated forecast.

### 1. Data Collection & Structuring
Instead of blindly fetching massive datasets, the system uses a **Dynamic Back-Fill Algorithm** via the `yfinance` API. It initially requests the last 5 years of daily price and dividend history. If the asset pays dividends but hasn't reached the minimum threshold of 25 historical payouts required for robust ML training, the system intelligently and iteratively reaches back further in time (up to 30 years) in 5-year chunks until it satisfies the training requirements or hits the company's IPO date. This raw data is then cleaned, timezone-normalized, and prepared for quantitative analysis.

### 2. Feature Engineering
Raw stock prices do not tell the model why the price is moving, and raw dividend amounts lack corporate context. The pipeline engineers a focused set of technical and fundamental features:

**Price Pipeline Features:**
* **Price Action & Momentum:** Calculates immediate Logarithmic Returns, multi-day Lagged Returns, and 10/21-day Rate of Change (ROC) to measure the raw signal and speed of immediate price movements.
* **Quantitative Market Indicators:**
    * **Relative Strength Index (RSI-5 & RSI-14):** Measures short and standard-term speed and change of price movements to signal "overbought" or "oversold" conditions.
    * **MACD Histogram:** Tracks the relationship between short-term and long-term exponential moving averages to quantify shifts in trend direction and momentum acceleration.
    * **Bollinger Bands:** Calculates band width for detecting volatility breakouts and the stock's position relative to the bands for dynamic support/resistance.
    * **Simple Moving Average (SMA) Ratios:** Calculates the ratio of the current price to the 50-day and 200-day SMAs to determine macro trend positioning.
* **Risk & Drawdown:** Calculates 20-day Historical Volatility and absolute drawdown percentages from 50-day and 200-day rolling maximums to quantify current asset risk.
* **Volume Context:** Calculates short-term volume ratios to determine if there is institutional conviction behind recent price movements.

**Dividend Pipeline Features:**
* **Immediate Growth Rate:** Calculates the period-over-period percentage change to detect recent payout bumps or cuts (`Div_Growth_1`).
* **Short-Term Historical Trends:** Uses a 4-cycle (1-year) rolling average of payouts to smooth out special dividends and establish a baseline trajectory (`Rolling_Avg_4`).
* **Trailing Price Performance:** Ingests the 252-day (1-year) trailing stock price return to correlate broader corporate health and market performance with board payout decisions (`Price_Return_252`).

### 3. The Price Forecasting Engine
The price prediction pipeline is split into a highly reactive short-term model and a macro-focused long-term model.

#### The Multi-Horizon Prediction Pipeline
Instead of chaining models together, the system trains independent models for 7 specific time horizons simultaneously (e.g., 1-day, 5-day, 21-day, up to 1-year). The daily trajectory path is seamlessly stitched together using log-linear interpolation between these anchor points.

* **The Directional Classifier:** A `HistGradientBoostingClassifier` wrapped in an Isotonic calibrator (`CalibratedClassifierCV`). This translates raw algorithmic margins into a true, statistically accurate probability of the stock moving UP or DOWN over that specific timeframe.
* **The Magnitude Regressors (Quantile Regression):** Rather than predicting a single arbitrary price point and guessing the error bounds, the system trains three separate `HistGradientBoostingRegressor` models simultaneously using a **Quantile Loss Function**. 
  * The **0.5 Quantile** strictly predicts the median (most likely) target price.
  * The **0.1 & 0.9 Quantiles** mathematically construct the explicit lower and upper bounds of a 95% confidence interval for that exact timeframe.
* **Calibrated Confidence:** Raw machine learning models often output arbitrary scores. This app passes its directional predictions through an Isotonic `CalibratedClassifierCV` to ensure the UI reports statistically accurate, real-world confidence percentages.
* **The Alignment Phase:** Predicting binary direction is statistically more reliable than predicting exact dollar amounts. The Classifier acts as the authoritative voice, and its calculated confidence is paired with the Regressor's dollar amounts to ensure logical consistency across the UI.
* **The Closed Price Date Projector:** The system dynamically maps future forecasting dates based on the specific asset class. For Cryptocurrencies, dates are mapped continuously on a 365-day schedule since their markets never close. For traditional equities and ETFs, it utilizes the `pandas` `CustomBusinessDay` module combined with the US Federal Holiday Calendar to accurately skip weekends and market closures when plotting the future trajectory.

### 4. The Dividend Forecasting Engine
Corporate dividends are structured, board-approved payouts rather than market-driven trades. The application runs an isolated, parallel pipeline to predict them, sharing the exact same robust `HistGradientBoosting` Quantile Regression architecture as the price engine.

#### The Multi-Cycle Prediction Pipeline

* **The Payout Date Projector:** The projected ex-dividend dates are calculated dynamically by analyzing the historical average day-spacing between past payouts.
* **The Directional Classifier:** A calibrated `HistGradientBoostingClassifier` determines the statistical probability of the dividend increasing or decreasing across the next 4 payout cycles.
* **Calibrated Confidence:** Raw machine learning models often output arbitrary scores. This app passes its directional predictions through an Isotonic `CalibratedClassifierCV` to ensure the UI reports statistically accurate, real-world confidence percentages.
* **The Magnitude Regressors:** Three separate `HistGradientBoostingRegressor` models use a Quantile Loss Function (0.1, 0.5, 0.9) to simultaneously predict the exact median dollar amount of the upcoming yields while mathematically constructing the explicit lower and upper bounds of a 95% confidence interval margin of error.

### 5. Natural Language Processing (NLP) Sentiment Analysis
The backend utilizes the HuggingFace `transformers` library to load the highly specialized `ProsusAI/finbert` NLP model into memory. It scrapes the most recent news data for the searched asset via Yahoo Finance APIs. It then processes the raw headlines through the neural network to identify positive or negative market catalysts. It also packages the full article summary and the original publisher name so users can read the contextual news directly inside the app's glassmorphic modal overlay.

**How the Score is Calculated:**
1. **Inference Mapping:** Each headline is evaluated by FinBERT, which returns a classification (`positive`, `negative`, or `neutral`) alongside a confidence probability (0.0 to 1.0).
2. **Directional Scaling:** The confidence scores are mapped to a bounded scale where `positive` classifications are represented as positive floats, `negative` classifications as negative floats, and `neutral` as zero.
3. **Aggregation:** The directional scores of all recent articles are summed together and divided by the total number of articles, establishing a baseline mean sentiment float between -1.0 and 1.0.
4. **Media Bias Calibration:** Financial news is inherently skewed toward positive framing. The baseline score is mathematically offset downwards by a fixed threshold (e.g., `-0.10`) to counteract this systemic positive bias, ensuring that only overwhelmingly bullish news yields a net positive impact on the overarching AI Stock Grade.
5. **Driver Extraction:** The absolute strongest positive (score > 0.4) and negative (score < -0.4) headlines are explicitly extracted from the pipeline and listed in the UI so the user can exactly trace the reasoning behind the NLP score.
---

## Local Development Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lc2410/stock-market-predictor.git](https://github.com/lc2410/stock-market-predictor.git)
    cd stock-market-predictor
    ```
2.  **Environment Setup (Python 3.12):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    cd backend
    pip install -r requirements.txt
    ```
3.  **Run the Local Server:**
    ```bash
    python3 app.py
    ```
    Navigate to `http://127.0.0.1:5001` in your browser.

---

## Cloud Infrastructure (Terraform / OCI)
The production environment is hosted on an **Oracle Cloud Infrastructure (OCI)** ARM-based instance (`VM.Standard.A1.Flex` shape with 4 OCPUs and 24GB RAM). 

Rather than configuring the server manually through a web console, the entire cloud environment is strictly version-controlled and provisioned using **Terraform**. This guarantees that the network topology is reproducible, auditable, and easily deployable by anyone cloning this repository.

### Step 1: Prerequisites & Authentication
To deploy your own instance of this architecture, you must first configure Terraform to communicate securely with Oracle Cloud:
1.  **Install Terraform** on your local machine.
2.  **Generate OCI API Keys:** Create an RSA key pair in your Oracle Cloud console and copy the generated credentials.
3.  **Configure Environment Variables:** Navigate to the `infra/` directory and create a `terraform.tfvars` file (do not commit this to version control). Populate it with your specific OCI credentials (P.S. there's an existing  `terraform.tfvars.txt` file that can be renamed into the necessary `terraform.tfvars` file).
    ```hcl
    tenancy_ocid     = "ocid1.tenancy.oc1..."
    user_ocid        = "ocid1.user.oc1..."
    fingerprint      = "xx:xx:xx:xx..."
    private_key_path = "~/.oci/oci_api_key.pem"
    region           = "us-ashburn-1" # Or your local region
    compartment_ocid = "ocid1.tenancy.oc1..."
    ssh_public_key   = "ssh-rsa..."
    ```

### Step 2: Provisioning the Network and Compute Layer
The Terraform scripts in the `infra/` directory are designed to build a secure, isolated network topology from the ground up:
* **Virtual Cloud Network (VCN):** Establishes the foundational private network.
* **Internet Gateway & Route Tables:** Connects the VCN to the public internet.
* **Security Lists (Firewalls):** Strictly restricts inbound traffic. It opens **Port 22 (SSH)** for GitHub Actions automated deployments and **Port 80 (HTTP)** for public Nginx web traffic. *(Note: Port 8000 for Gunicorn is intentionally kept closed to the public internet for security, operating strictly internally).*

To provision the infrastructure, run the following commands from the `infra/` directory:
```bash
terraform init    # Initializes the OCI provider
terraform plan    # Reviews the exact infrastructure changes
terraform apply   # Provisions the VCN, Subnets, and Virtual Machine
```

---

## Testing & Code Quality
To ensure maximum reliability and prevent regressions, the application enforces strict quality gates through automated testing and static code analysis.

### 1. Backend Testing (Pytest)
A comprehensive suite of unit and integration tests validate the machine learning pipeline. Tests simulate complex edge cases including mocked Yahoo Finance outages, missing dividend histories, and sparse ticker data. Code coverage is strictly maintained at **>95%**.

**Local Execution:**
```bash
# Run tests and generate terminal coverage report and a line-by-line HTML visual report
python3 -m pytest backend/tests/ --cov=. --cov-report=term --cov-report=html
```

### 2. Frontend E2E Testing (Playwright)
Automated headless browsers simulate real human interaction. Playwright tests the full UI lifecycle, including typing into the search bar, validating loading spinners, and verifying that the API successfully returns and renders the chart data.

**Local Execution:**
```bash
# Running the python application first
python3 app.py

# Navigate to the frontend directory and install dependencies
cd frontend
npm install
npx playwright install 

# Execute the End-to-End test suite
npm run test:frontend
```

### 3. Static Analysis & Security
* **SonarQube Cloud:** Every pull request and push is automatically scanned. It acts as a strict security gate, catching vulnerabilities, code smells, log injection risks, and enforcing test coverage minimums.
* **Security & Linting:** Python code is checked for syntactical integrity using `Flake8`. Dependency trees are scanned for known CVEs and vulnerabilities using `safety` (Python) and `npm audit` (JavaScript).

---

## CI/CD Pipeline (GitHub Actions)
The continuous integration and continuous delivery/deployment lifecycle is fully automated through a rigorous, multi-stage GitHub Actions pipeline (`deploy.yml`). Pushing a commit to the `main` branch triggers the following sequence:

1. **Security & Vulnerability Scan:** Audits Python and NPM dependencies for known CVEs.
2. **Code Linting:** Runs Flake8 to ensure Python styling and syntax standards.
3. **Automated Testing:** Boots a localized version of the app and runs both the Pytest backend suite and the Playwright E2E browser tests.
4. **Static Code Analysis:** Uploads the XML test coverage reports to SonarCloud to verify the Quality Gate passes.
5. **Zero-Downtime Deployment:** Only if all prior stages pass perfectly, the pipeline establishes a secure SSH connection to the OCI production instance, pulls the latest repository updates, and gracefully restarts the Gunicorn `systemd` service to serve the new code.

---

## Live Production Environment
The application provides a clean, responsive web interface featuring the interactive Chart.js engine to display historical trends and future forecasts.

You can access the live production environment hosted on Oracle Cloud here: [http://150.136.36.70](http://150.136.36.70)
*(Note: This is currently accessible via direct IP until domain name resolution and SSL certification are configured).*

---

## Core Technologies
* **Cloud & Infrastructure:** Oracle Cloud (OCI), Terraform, Linux (Ubuntu)
* **CI/CD & DevOps:** GitHub Actions, SonarCloud (Static Analysis)
* **Web Serving:** Nginx (Reverse Proxy), Gunicorn (WSGI)
* **Back-End:** Python, Flask, Server-Sent Events (SSE)
* **Machine Learning:** Scikit-learn, Pandas, NumPy, HuggingFace Transformers (FinBERT)
* **Data Sourcing:** yfinance (Yahoo Finance API)
* **Front-End:** HTML5, CSS3, Vanilla JavaScript, Chart.js
* **Testing:** Pytest (Unit/Integration), Playwright (E2E UI Testing)