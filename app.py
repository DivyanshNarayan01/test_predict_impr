#!/usr/bin/env python3
"""
Flask Web Application for Campaign Prediction
Allows users to input campaign parameters and get predictions for impressions and engagement.
"""

from flask import Flask, request, jsonify, render_template
from predict_api import predict_campaign_metrics
import logging
import os
import json

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
            "INFLUENCER - OGILVY - BOOSTED ONLY",
            "OWNED - BOOSTED ONLY",
            "PAID - BRAND",
            "PAID - PARTNERSHIP"
        ]
    }), 200


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information and statistics from metadata file."""
    try:
        metadata_file = os.path.join(BASE_DIR, 'models', 'multi_output_model_metadata.json')

        if not os.path.exists(metadata_file):
            return jsonify({
                "status": "error",
                "error_message": "Model metadata file not found"
            }), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return jsonify({
            "status": "success",
            "model_info": metadata
        }), 200

    except Exception as e:
        logger.error(f"Error loading model metadata: {str(e)}")
        return jsonify({
            "status": "error",
            "error_message": f"Failed to load model metadata: {str(e)}"
        }), 500


@app.route('/api/training-data', methods=['GET'])
def get_training_data():
    """Get training data for visualization."""
    try:
        import pandas as pd
        import numpy as np

        # Load training data
        data_file = os.path.join(BASE_DIR, 'data', 'campaign_data.csv')

        if not os.path.exists(data_file):
            return jsonify({
                "status": "error",
                "error_message": "Training data file not found"
            }), 404

        df = pd.read_csv(data_file)

        # Calculate log-transformed values
        df['log_spend'] = np.log(df['total_spend'] + 1)
        df['log_impressions'] = np.log(df['Impressions'] + 1)

        # Prepare data for scatter plot
        training_data = []
        for _, row in df.iterrows():
            training_data.append({
                'x': float(row['log_spend']),
                'y': float(row['log_impressions']),
                'platform': row['Platform'],
                'campaign_type': row['campaign_type'],
                'content_type': row['content_type'],
                'spend': float(row['total_spend']),
                'impressions': int(row['Impressions'])
            })

        return jsonify({
            "status": "success",
            "data": training_data,
            "count": len(training_data)
        }), 200

    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return jsonify({
            "status": "error",
            "error_message": f"Failed to load training data: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Try to load metadata to get actual model name
        metadata_file = os.path.join(BASE_DIR, 'models', 'multi_output_model_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        model_name = metadata.get('model_name', 'Unknown')
    except:
        model_name = 'Unknown'

    return jsonify({
        "status": "healthy",
        "service": "campaign-prediction-api",
        "model": model_name
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
