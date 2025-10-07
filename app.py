#!/usr/bin/env python3
"""
Flask Web Application for Campaign Prediction
Allows users to input campaign parameters and get predictions for impressions and engagement.
"""

from flask import Flask, request, jsonify, render_template
from predict_api import predict_campaign_metrics
import logging

# Initialize Flask app
app = Flask(__name__)

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
 platform=data.get('platform', 'TIKTOK'),
 campaign_type=data.get('campaign_type', 'FLOOD THE FEED'),
 content_type=data.get('content_type', 'INFLUENCER - CFG - BOOSTED ONLY'),
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
 "platforms": ["META", "TIKTOK", "INSTAGRAM"],
 "campaign_types": ["BAU", "MM", "FLOOD THE FEED"],
 "content_types": [
 "INFLUENCER - CFG - BOOSTED ONLY",
 "INFLUENCER - OGILVY - ORGANIC ONLY",
 "OWNED - BOOSTED ONLY",
 "OWNED - ORGANIC ONLY",
 "PAID - BRAND",
 "PAID - PARTNERSHIP"
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
 print("\nOpen your browser and navigate to:")
 print(" http://localhost:5000")
 print("\nPress CTRL+C to stop the server\n")
 print("=" * 80)

 app.run(host='0.0.0.0', port=5000, debug=True)
