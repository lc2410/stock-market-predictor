# Real-Time Stock & Dividend Forecaster

## Overview
This project is a full-stack web application designed to bridge the gap between unstructured, raw financial data and a production-ready, interactive forecasting system. It provides a real-time, next-day forecast for a stock's closing price alongside an intelligent projection of its next dividend payment. 

The back-end is powered by a custom-built machine learning ensemble utilizing Python and Scikit-learn, served via a **Flask web framework**. The front-end delivers a clean, responsive, and highly interactive interface built with HTML, CSS, JavaScript, and Chart.js. 

Designed for high-velocity execution and scalability, the entire application operates on a monolithic, "all-in-one" client-server architecture deployed on a custom-provisioned Oracle Cloud Infrastructure (OCI) environment, fully managed by Infrastructure as Code (IaC) and a CI/CD pipeline.

---

## Application Architecture
To ensure enterprise-grade stability, security, and performance, this application relies on a multi-layered architecture running on a dedicated Linux virtual machine (Ubuntu 22.04 LTS).

![Image of Application Architecture](imgs/stock-market-predictor-arch.png)

1.  **Client (Front-End):** The user's web browser loads the HTML, CSS, and JavaScript files. The UI is decoupled from the heavy ML processing, utilizing asynchronous `fetch` calls to ensure the interface remains responsive during model training.
2.  **Web Server (Nginx Reverse Proxy):** Nginx acts as the secure front door to the application. It intercepts incoming public HTTP traffic on port 80, buffering requests to protect the internal server from slow clients or malicious spikes. It safely proxies validated dynamic requests to the internal application layer.
3.  **Application Server (Gunicorn WSGI):** Because web servers (Nginx) and Python applications (Flask) speak different protocols, Gunicorn acts as the essential Web Server Gateway Interface (WSGI) translator. It runs as a highly available background `systemd` service on internal port 8000, managing a pool of worker processes that execute the Flask code in parallel.
4.  **API & ML Execution:** The Flask routing layer catches the API request (e.g., `/predict/AAPL`), triggers the algorithmic pipeline to fetch real-time Yahoo Finance data, dynamically trains and executes the machine learning models, and returns the multi-horizon forecast as structured JSON.

---

## Core Features
* **Interactive Charting Engine:** A custom Chart.js interface featuring a draggable time-navigator. This allows users to pan and zoom seamlessly across decades of historical market data and visually explore the projected future trajectories.
* **Dual Forecasting Pipelines:** Independently predicts both the next-day stock price (direction and magnitude) and the next projected dividend payout (amount and date) using isolated ML logic.
* **Robust Temporal Logic:** Financial markets don't operate on standard calendars. The app uses the `pandas` `CustomBusinessDay` module combined with the US Federal Holiday Calendar to accurately project exact future trading dates, skipping weekends and market closures.
* **Calibrated Confidence:** Raw machine learning models often output arbitrary scores rather than true probabilities. This app passes its directional predictions through an Isotonic `CalibratedClassifierCV` to ensure the UI reports statistically accurate, real-world confidence percentages.
* **Advanced Technical Feature Engineering:** The system automatically bridges raw open/close data into complex quantitative indicators, including Relative Strength Index (RSI), MACD, Bollinger Band width/positioning, and Average True Range (ATR).

---

## The Machine Learning Process
The forecasting engine abandons simple linear models in favor of dynamic feature engineering, aggressive noise reduction, and an ensemble architecture designed to minimize variance and prevent overfitting.

### 1. Data Collection & Structuring
Upon receiving a ticker symbol, the `yfinance` library retrieves the maximum available daily price and dividend history (dating back to 2010). This unstructured raw data is immediately cleaned, indexed, and prepared for quantitative analysis.

### 2. Feature Engineering & Dynamic Pruning
Raw stock prices (Open, High, Low, Close) are just naked numbers; they don't tell the model *why* the price is moving. The pipeline engineers over 40 distinct features to give the models deep contextual awareness:

* **Time-Horizon Analysis:** Rolling volatility, momentum, and close-to-mean ratios are calculated across 9 distinct lookback windows (from 2 days up to 5 years) to capture both micro-trends and macro-cycles.
* **Quantitative Market Indicators:**
    * **Relative Strength Index (RSI - 14 Day):** Measures the speed and change of price movements. This prevents the model from blindly predicting that a stock will keep going up forever just because it had a few good days by signaling "overbought" (>70) or "oversold" (<30) conditions.
    * **MACD (Moving Average Convergence Divergence):** By tracking the relationship between the 12-day and 26-day moving averages (including the MACD Line, Signal Line, and Histogram), the model can instantly quantify and react to sudden shifts in trend direction and momentum acceleration.
    * **Bollinger Bands (20-day & 50-day):** The model calculates both the band *width* (to detect "squeezes" that signal imminent volatility breakouts) and the stock's *position* relative to the bands (to identify dynamic support and resistance levels).
    * **Normalized Average True Range (ATR):** Measures pure market volatility, accounting for overnight "gaps." By normalizing the 14-day ATR against the current close price, the magnitude regressor learns a mathematical boundary of how wildly a specific stock is capable of moving in a single day, preventing unrealistic price targets.
* **Dynamic Noise Reduction:** Before final training, the system runs a low-depth Random Forest to evaluate feature importance. It dynamically drops the bottom 25% of the lowest-performing features to aggressively reduce dimensionality and eliminate noisy data.

### 3. The Price Forecasting Engine
Stock prices are inherently noisy. To handle this, the price prediction pipeline is split into a highly reactive short-term model and a macro-focused long-term model.

#### The Next-Day Predictor (Short-Term)
This subsystem translates market momentum into an immediate forecast utilizing a "Director" (Classifier) and an "Estimator" (Regressor) that are forcibly aligned to guarantee logical consistency on the front-end:
* **The Directional Classifier (The "Director"):** A `VotingClassifier` aggregates the predictions of two algorithmically distinct models: a `RandomForest` (bagging) and a `GradientBoosting` (boosting) model. This ensemble is then wrapped in an Isotonic calibrator to translate raw algorithmic margins into a true, statistically accurate probability of the stock closing UP or DOWN.
* **The Magnitude Regressor (The "Estimator"):** A `RandomForestRegressor` is hyperparameter-tuned on the fly using `RandomizedSearchCV`. It tests various tree depths and leaf sizes in real-time—optimizing for Negative Mean Absolute Error (MAE)—to find the perfect configuration for the specific volatility profile of the requested stock.
* **The Alignment Phase:** Because predicting binary direction is statistically more reliable than predicting exact dollar amounts, the Classifier acts as the authoritative voice. If the Regressor's raw dollar amount contradicts the Classifier's direction, a custom logic block intercepts and overrides the magnitude, ensuring the final output presented to the user is logically coherent.

#### The Long-Term Trajectory Engine (252-Day Forward Path)
To map a full trading year (252 days), the system diverges from the reactive next-day models. It bridges historical momentum into a forward-looking curve by training three separate, horizon-specific regressors targeting cumulative log-returns at the 5-day, 21-day, and 252-day marks. The daily path is stitched together using log-linear interpolation and bounded by a dynamic 95% Margin of Error band derived directly from the stock's historical daily volatility.

### 4. The Dividend Forecasting Engine
Corporate dividends require a fundamentally different analytical approach than chaotic daily stock prices. They are structured, board-approved payouts rather than market-driven trades. Therefore, the application runs a completely isolated, parallel pipeline to predict them, mirroring the dual-model architecture of the price engine:

* **The Directional Classifier (Payout Trend):** Instead of relying on technical trading indicators like RSI or MACD, this dedicated `RandomForestClassifier` evaluates fundamental features engineered from unstructured payout histories—specifically dividend growth rates and rolling averages over the last 4, 6, and 8 payout cycles. It uses this targeted context to determine the binary probability of the next dividend increasing or decreasing.
* **The Magnitude Regressor (Payout Amount):** Operating in tandem with the classifier, a separate `RandomForestRegressor` utilizes the same fundamental context to forecast the exact dollar amount of the upcoming yield. 
* **The Payout Date Projector:** Because dividend announcements follow corporate fiscal calendars rather than trading algorithms, the projected date is calculated completely independently from the dollar amount. The system dynamically maps the historical spacing between past payouts to project the exact calendar date of the next distribution.

---

## Setup, Infrastructure, & Deployment

This application has been elevated from a local script to a cloud-native, production-ready system utilizing Infrastructure as Code and Continuous Deployment.

### Local Development
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lc2410/stock-market-predictor.git](https://github.com/lc2410/stock-market-predictor.git)
    cd stock-market-predictor
    ```
2.  **Environment Setup (Python 3.12):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Run the Local Server:**
    ```bash
    python app.py
    ```
    Navigate to `http://127.0.0.1:5001` in your browser.

### Cloud Infrastructure (Terraform / OCI)
The production environment is hosted on an **Oracle Cloud Infrastructure (OCI)** ARM-based instance (`VM.Standard.A1.Flex` shape with 4 OCPUs and 24GB RAM). 

Rather than configuring the server manually through a web console, the entire cloud environment is strictly version-controlled and provisioned using **Terraform**. This guarantees that the network topology is reproducible, auditable, and easily deployable by anyone cloning this repository.

#### Step 1: Prerequisites & Authentication
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

#### Step 2: Provisioning the Network and Compute Layer
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

### Continuous Deployment (GitHub Actions)
The deployment lifecycle is fully automated. Pushing a commit to the `main` branch triggers a secure GitHub Actions workflow (`.github/workflows/deploy.yml`). This CI/CD pipeline establishes an SSH connection to the OCI instance, pulls the latest repository updates, and seamlessly restarts the Gunicorn `systemd` service to serve the new code with zero manual intervention.

---

## Live Production Environment
The application provides a clean, responsive web interface featuring the interactive Chart.js engine to display historical trends and future forecasts.

You can access the live production environment hosted on Oracle Cloud here: [http://150.136.36.70](http://150.136.36.70)
*(Note: This is currently accessible via direct IP until domain name resolution and SSL certification are configured).*

---

## Core Technologies
* **Infrastructure & DevOps:** Oracle Cloud (OCI), Terraform, GitHub Actions (CI/CD), Linux (Ubuntu)
* **Web Serving:** Nginx (Reverse Proxy), Gunicorn (WSGI)
* **Back-End:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Data Sourcing:** yfinance (Yahoo Finance API)
* **Front-End:** HTML5, CSS3, JavaScript, Chart.js