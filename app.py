from flask import Flask, jsonify, render_template
from flask_cors import CORS
from prediction_model import run_real_time_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    app.logger.info(f"Received prediction request for ticker: {ticker}")
    try:
        prediction_df = run_real_time_model(ticker.upper())
        if prediction_df is None:
            app.logger.warning(f"Could not generate prediction for {ticker}.")
            return jsonify({"error": f"Invalid ticker or insufficient data for {ticker}."}), 404
        
        result = prediction_df.to_dict(orient='records')[0]
        app.logger.info(f"Successfully generated prediction for {ticker}.")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error for ticker {ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
