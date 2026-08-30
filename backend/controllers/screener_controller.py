"""Screener controller for the /screener endpoint."""

from flask import Blueprint, jsonify
import logging

from services.screener_service import get_screener_dashboard_data

screener_bp = Blueprint('screener', __name__)
logger = logging.getLogger(__name__)

@screener_bp.route('/screener', methods=['GET'])
def screener():
    """Returns the aggregated data for the screener homepage."""
    try:
        data = get_screener_dashboard_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error serving screener data: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while fetching screener data."}), 500
