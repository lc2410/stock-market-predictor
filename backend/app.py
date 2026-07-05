"""
app.py
------
This is the main bootloader for the Flask backend API.
It initializes the Flask application, configures Cross-Origin Resource Sharing (COS),
and registers the API blueprint containing our forecasting and sentiment routes.

In production, this file is executed by Gunicorn, which manages a pool of worker
processes to handle requests concurrently.
"""
import logging
from flask import Flask
from flask_cors import CORS

from apis.routes import api_bp

# Configure basic logging for debugging and tracking execution flows
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application. Note that Flask only serves as a JSON API 
# data provider here — the React SPA frontend is independently served by Nginx.
app = Flask(__name__)  # NOSONAR

# Enable CORS (Cross-Origin Resource Sharing) to allow the React frontend 
# (running on a different port/domain during dev) to communicate with this API.
CORS(app)

# Register the Blueprint that contains all of our API endpoints (e.g. /search, /predict)
app.register_blueprint(api_bp)

if __name__ == '__main__':
    # When run directly (e.g., python app.py), start the Flask development server on port 5001.
    app.run(debug=True, port=5001)