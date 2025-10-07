#!/usr/bin/env python3
"""
Flask Web Application for Campaign Prediction
Allows users to input campaign parameters and get predictions for impressions and engagement.
"""

from flask import Flask, request, jsonify, render_template
from predict_api import predict_campaign_metrics
import logging
import os

# Get the absolute path of the current file's directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask app with explicit template folder
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def home():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.

    Accepts JSON with campaign parameters and returns predictions.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "error_type": "InvalidRequest",
                "error_message": "Request body must be JSON"
            }), 400

        if 'total_spend' not in data:
            return jsonify({
                "status": "error",
                "error_type": "ValidationError",
                "error_message": "Missing required field: total_spend"
            }), 400

        # Make prediction
        result = predict_campaign_metrics(
            total_spend=data.get('total_spend'),
            platform=data.get('platform', 'TikTok'),
            campaign_type=data.get('campaign_type', 'Flood The Feed'),
            content_type=data.get('content_type', 'Influencer - Cfg - Boosted Only'),
            return_format='dict'
        )

        # Check if prediction was successful
        if result['status'] == 'error':
            return jsonify(result), 400

        logger.info(f"Prediction successful for spend: ${data.get('total_spend'):,.2f}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "error_type": "ServerError",
            "error_message": f"Internal server error: {str(e)}"
        }), 500


@app.route('/api/options', methods=['GET'])
def get_options():
    """Return valid options for dropdown menus."""
    return jsonify({
        "platforms": ["Meta", "TikTok", "Instagram"],
        "campaign_types": ["Bau", "Mm", "Flood The Feed"],
        "content_types": [
            "Influencer - Cfg - Boosted Only",
            "Influencer - Ogilvy - Organic Only",
            "Owned - Boosted Only",
            "Owned - Organic Only",
            "Paid - Brand",
            "Paid - Partnership"
        ]
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "campaign-prediction-api",
        "model": "Random Forest MultiOutput"
    }), 200


if __name__ == '__main__':
    print("=" * 80)
    print("Campaign Prediction API Starting...")
    print("=" * 80)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Template Folder: {app.template_folder}")
    print(f"Template exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
    print("\nOpen your browser and navigate to:")
    print("    http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5000, debug=True)
