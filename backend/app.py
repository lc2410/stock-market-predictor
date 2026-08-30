"""Flask backend API bootloader."""
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask
from flask_cors import CORS

from controllers.search_controller import search_bp
from controllers.prediction_controller import prediction_bp
from controllers.screener_controller import screener_bp

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)  # NOSONAR
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

app.register_blueprint(search_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(screener_bp)

if __name__ == '__main__':  # pragma: no cover
    is_debug = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"]
    app.run(debug=is_debug, port=5001)