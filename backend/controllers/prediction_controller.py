"""Prediction controller for /predict and /predict_stream endpoints."""

from flask import Blueprint, jsonify, Response, stream_with_context
import json
import logging

from services.prediction_service import run_prediction_pipeline, resolve_search_query

prediction_bp = Blueprint('prediction', __name__)
logger = logging.getLogger(__name__)

@prediction_bp.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    """Orchestrates ML math, NLP sentiment, Fundamentals, and UI formatting into a single payload."""
    safe_ticker = ticker.replace('\n', '').replace('\r', '').upper()
    logger.info(f"Received prediction request for ticker: {safe_ticker}")
    
    try:
        for update in run_prediction_pipeline(safe_ticker):
            if update["status"] == "complete":
                logger.info(f"Successfully generated prediction for {safe_ticker}.")
                return jsonify(update["result"])
            elif update["status"] == "error":
                return jsonify({"error": update["error"]}), 404
                
    except Exception as e:
        logger.error(f"Error for ticker {safe_ticker}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@prediction_bp.route('/predict_stream/<string:ticker>', methods=['GET'])
def predict_stream(ticker):
    """SSE Streaming Endpoint for UI Progress updates."""
    raw_ticker = ticker.replace('\n', '').replace('\r', '').strip()
    safe_ticker = resolve_search_query(raw_ticker)
    
    def generate():
        try:
            for update in run_prediction_pipeline(safe_ticker):
                yield f"data: {json.dumps(update)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error for ticker {safe_ticker}: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'error': 'An internal server error occurred.'})}\n\n"

    headers = {
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers=headers)
