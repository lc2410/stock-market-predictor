import os
from flask import Flask
from flask_cors import CORS
import logging

# Import the API routes from the routes module
from backend.apis.routes import api_bp

logging.basicConfig(level=logging.INFO)

# Calculate the absolute path to the frontend folder
base_dir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.join(base_dir, 'frontend')

# Initialize Flask, pointing it to the frontend directory
app = Flask(__name__, 
            template_folder=frontend_dir, 
            static_folder=frontend_dir) # NOSONAR

CORS(app)

# Register the routes
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5001)