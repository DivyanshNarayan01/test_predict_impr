#!/usr/bin/env python3
"""
Flask Web Application - Jupyter Notebook Version

This file creates a web server that provides a beautiful user interface and
REST API for campaign predictions. It's optimized for Jupyter Notebook with
extensive comments to help non-technical users understand how it works.

=============================================================================
WHAT THIS FILE DOES (IN SIMPLE TERMS):
=============================================================================
This file creates a WEBSITE (web application) that:
  1. Shows a beautiful user interface in your web browser
  2. Lets marketing teams enter campaign details through forms
  3. Makes predictions using the AI model
  4. Shows results in easy-to-read tables and charts
  5. Provides a REST API for developers to integrate with other systems

Think of it as a "wrapper" around the prediction function - it makes the
AI accessible to everyone, not just programmers!

=============================================================================
HOW TO USE IN JUPYTER NOTEBOOK:
=============================================================================
1. Convert this file to a notebook:
   - In Jupyter: File → New → Notebook
   - Copy each CELL block into a separate cell

2. OR use jupytext to convert automatically:
   pip install jupytext
   jupytext --to notebook app_notebook.py

3. Run cells in order (CELL 1, CELL 2, CELL 3, etc.)

4. After running all cells, open your browser to:
   http://localhost:5000

=============================================================================
BUSINESS VALUE:
=============================================================================
Before this web app:
  - Only data scientists could use the AI model
  - Needed to write Python code for each prediction
  - Hard to share with marketing team

After this web app:
  - Anyone can use it (no coding required!)
  - Beautiful Google Cloud-style interface
  - Predict multiple campaigns at once
  - See aggregate metrics across your entire portfolio
  - Share the URL with your team

This democratizes AI - making it accessible to everyone!

=============================================================================
TECHNICAL ARCHITECTURE:
=============================================================================
This uses FLASK, a Python web framework. Flask handles:
  - HTTP requests from browsers (GET, POST)
  - Routing (which URL goes to which function)
  - Serving HTML pages (the user interface)
  - JSON API endpoints (for developers)

The flow is:
  User's Browser → HTTP Request → Flask → predict_api.py → AI Model
                                   ↓
  User's Browser ← HTTP Response ← Flask ← Predictions ← AI Model

=============================================================================
"""

# =============================================================================
# CELL 1: Import Required Libraries
# =============================================================================
#  What this cell does:
#   - Imports Flask (the web server framework)
#   - Imports our prediction function from predict_api.py
#   - Sets up logging (to track what happens)
#
#  What each library does:
#   - Flask: Creates the web server
#   - request: Handles incoming HTTP requests (GET, POST)
#   - jsonify: Converts Python dictionaries to JSON (for API responses)
#   - render_template: Loads HTML templates
#   - logging: Tracks events and errors
#
# ⏱  Run time: <1 second
#  Expected output: No output (libraries load silently)

from flask import Flask, request, jsonify, render_template
from predict_api import predict_campaign_metrics
import logging

print(" Flask and dependencies imported successfully!")


# =============================================================================
# CELL 2: Initialize Flask Application
# =============================================================================
#  What this cell does:
#   - Creates a Flask application instance
#   - Sets up logging to track requests and errors
#
#  What Flask does:
#   - Flask is like a "receptionist" for your website
#   - It listens for incoming requests (users visiting pages)
#   - Routes requests to the right function
#   - Returns responses (HTML pages or JSON data)
#
#  What logging does:
#   - Tracks every request: "User predicted a $10,000 campaign"
#   - Records errors: "Model file not found!"
#   - Helps debug issues in production
#
# ⏱  Run time: <1 second
#  Expected output: "Flask app initialized!"

# Create the Flask application
# __name__ tells Flask where to find templates and static files
app = Flask(__name__)

# Configure logging to INFO level (shows important events)
# DEBUG = too verbose, WARNING = too quiet, INFO = just right
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" Flask app initialized!")
print("   App name:", app.name)
print("   Debug mode:", app.debug)


# =============================================================================
# CELL 3: Home Page Route - Serve the Web Interface
# =============================================================================
#  What this cell does:
#   - Defines the HOME PAGE route (the main page users see)
#   - When someone visits http://localhost:5000/, show them the UI
#
#  Web Terminology:
#   - ROUTE: A URL path like "/" or "/about"
#   - GET Request: User visits a page (browser asks "give me this page")
#   - POST Request: User submits a form (browser sends data to server)
#   - Template: HTML file that defines what the page looks like
#
#  How routing works:
#   @app.route('/') means "when user visits the homepage..."
#   def home() means "...run this function"
#   return render_template() means "...and show them this HTML page"
#
# ⏱  Run time: Instant (just defines the route, doesn't run yet)
#  Expected output: "Home route defined!"

@app.route('/')
def home():
    """
    Serve the main web interface (the homepage).

    When a user opens http://localhost:5000/ in their browser, Flask
    runs this function and returns the HTML page.

    The page is defined in templates/index.html - a beautiful Google
    Cloud Platform-style interface where users can:
      - Add multiple campaigns
      - Set budget allocations
      - Predict all campaigns with one click
      - See aggregate metrics across their portfolio

    Returns:
        HTML page rendered from templates/index.html
    """
    # render_template() looks for 'index.html' in the 'templates/' folder
    # and returns it to the user's browser
    return render_template('index.html')

print(" Home route defined: GET /")


# =============================================================================
# CELL 4: Prediction API Endpoint - The Core Business Logic
# =============================================================================
#  What this cell does:
#   - Defines the PREDICTION API endpoint
#   - Accepts campaign details via POST request (JSON format)
#   - Makes predictions using the AI model
#   - Returns results as JSON
#
#  API Terminology:
#   - API (Application Programming Interface): A way for programs to talk
#   - REST API: Uses HTTP methods (GET, POST) with JSON data
#   - Endpoint: A specific URL that does something (like /api/predict)
#   - JSON: Text format for structured data {"key": "value"}
#
#  Request/Response Flow:
#   1. User's browser sends POST to /api/predict with campaign details
#   2. Flask receives the request
#   3. We extract the data (total_spend, platform, etc.)
#   4. We call predict_campaign_metrics() from predict_api.py
#   5. AI model makes predictions
#   6. We return predictions as JSON
#   7. User's browser displays the results
#
#  Error Handling:
#   - Missing JSON body? → Return error "Request body must be JSON"
#   - Missing total_spend? → Return error "Missing required field"
#   - Invalid platform? → predict_api.py validates and returns error
#   - Server crash? → Return error "Internal server error"
#
# ⏱  Run time: Instant (just defines the route, doesn't run yet)
#  Expected output: "Prediction route defined!"

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction API endpoint.

    This is the MAIN API ENDPOINT that marketing teams and developers use
    to get campaign predictions. It accepts campaign details and returns
    predicted performance metrics.

    =================================================================
    HOW TO USE THIS API (FOR DEVELOPERS):
    =================================================================
    Send a POST request to http://localhost:5000/api/predict with JSON body:

    {
      "total_spend": 10000,
      "platform": "TikTok",
      "campaign_type": "Flood The Feed",
      "content_type": "Influencer - Cfg - Boosted Only"
    }

    You'll get back:

    {
      "status": "success",
      "predictions": {
        "impressions": 418398,
        "engagement": 50503,
        "engagement_rate": 0.1207,
        "engagement_rate_pct": "12.07%"
      },
      "metrics": {
        "cpm": 23.90,
        "cost_per_engagement": 0.20
      },
      "confidence_intervals": { ... }
    }
    =================================================================

    Accepts:
        POST request with JSON body containing:
          - total_spend (required): Campaign budget in dollars
          - platform (optional): Default 'TikTok'
          - campaign_type (optional): Default 'Flood The Feed'
          - content_type (optional): Default 'Influencer - Cfg - Boosted Only'

    Returns:
        JSON response with predictions and metrics (200 OK)
        OR error message (400 Bad Request, 500 Internal Server Error)
    """

    try:
        # ===========================
        # STEP 1: Get JSON Data from Request
        # ===========================
        # request.get_json() extracts the JSON body from the POST request
        # Returns None if body is empty or not valid JSON

        data = request.get_json()

        # ===========================
        # STEP 2: Validate Request Has Data
        # ===========================
        # If data is None, the user didn't send JSON (maybe sent plain text?)

        if not data:
            return jsonify({
                "status": "error",
                "error_type": "InvalidRequest",
                "error_message": "Request body must be JSON. Did you set Content-Type: application/json?"
            }), 400  # 400 = Bad Request (client error)

        # ===========================
        # STEP 3: Validate Required Field
        # ===========================
        # total_spend is REQUIRED - can't predict without a budget!

        if 'total_spend' not in data:
            return jsonify({
                "status": "error",
                "error_type": "ValidationError",
                "error_message": "Missing required field: total_spend"
            }), 400

        # ===========================
        # STEP 4: Make Prediction
        # ===========================
        # Call our prediction function from predict_api.py
        # Pass the campaign details, using defaults if fields are missing
        #
        # data.get('platform', 'TikTok') means:
        #   - If 'platform' exists in data, use it
        #   - Otherwise, use default 'TikTok'

        result = predict_campaign_metrics(
            total_spend=data.get('total_spend'),           # Required
            platform=data.get('platform', 'TikTok'),       # Default: TikTok
            campaign_type=data.get('campaign_type', 'Flood The Feed'),  # Default
            content_type=data.get('content_type', 'Influencer - Cfg - Boosted Only'),  # Default
            return_format='dict'                           # Return as dictionary, not JSON string
        )

        # ===========================
        # STEP 5: Check if Prediction Succeeded
        # ===========================
        # predict_campaign_metrics() returns {"status": "error", ...} if something went wrong
        # (like invalid platform, negative budget, etc.)

        if result['status'] == 'error':
            # Prediction failed (validation error, model error, etc.)
            # Return the error to the user with 400 status code
            return jsonify(result), 400

        # ===========================
        # STEP 6: Log Success (for monitoring)
        # ===========================
        # Write to the log file: "Prediction successful for spend: $10,000.00"
        # Helps us track usage and debug issues

        logger.info(f"Prediction successful for spend: ${data.get('total_spend'):,.2f}")

        # ===========================
        # STEP 7: Return Success Response
        # ===========================
        # jsonify() converts the Python dictionary to JSON
        # 200 = OK (success!)

        return jsonify(result), 200

    except Exception as e:
        # ===========================
        # ERROR HANDLING - Catch All Unexpected Errors
        # ===========================
        # If ANYTHING goes wrong that we didn't anticipate, catch it here
        # Examples: model file corrupted, out of memory, etc.

        logger.error(f"Unexpected error: {str(e)}")

        return jsonify({
            "status": "error",
            "error_type": "ServerError",
            "error_message": f"Internal server error: {str(e)}"
        }), 500  # 500 = Internal Server Error

print(" Prediction route defined: POST /api/predict")


# =============================================================================
# CELL 5: Options API Endpoint - Get Valid Dropdown Values
# =============================================================================
#  What this cell does:
#   - Returns the list of valid options for dropdown menus
#   - Used by the frontend to populate dropdown fields
#
#  Why this is useful:
#   - Frontend doesn't need to hardcode options
#   - If we add new platforms, we just update this endpoint
#   - Single source of truth for valid values
#
#  API Usage:
#   GET http://localhost:5000/api/options
#   Returns: {"platforms": [...], "campaign_types": [...], "content_types": [...]}
#
# ⏱  Run time: Instant (just defines the route)
#  Expected output: "Options route defined!"

@app.route('/api/options', methods=['GET'])
def get_options():
    """
    Return valid options for dropdown menus in the UI.

    This endpoint provides the list of valid values for:
      - Platforms (Meta, TikTok, Instagram)
      - Campaign Types (Bau, Mm, Flood The Feed)
      - Content Types (6 options)

    The frontend (index.html) calls this endpoint when the page loads
    to populate the dropdown menus dynamically.

    Why not hardcode these in the HTML?
      - Single source of truth (if we update here, UI updates automatically)
      - Easier to maintain
      - Prevents typos and inconsistencies

    Returns:
        JSON with three arrays of valid options (200 OK)
    """

    # Return the valid options as a JSON response
    return jsonify({
        "platforms": [
            "Meta",
            "TikTok",
            "Instagram"
        ],
        "campaign_types": [
            "Bau",
            "Mm",
            "Flood The Feed"
        ],
        "content_types": [
            "Influencer - Cfg - Boosted Only",
            "Influencer - Ogilvy - Organic Only",
            "Owned - Boosted Only",
            "Owned - Organic Only",
            "Paid - Brand",
            "Paid - Partnership"
        ]
    }), 200

print(" Options route defined: GET /api/options")


# =============================================================================
# CELL 6: Health Check Endpoint - Monitor Server Status
# =============================================================================
#  What this cell does:
#   - Provides a simple "health check" endpoint
#   - Used to verify the server is running
#
#  Why this is important:
#   - In production, monitoring tools ping /health every minute
#   - If /health doesn't respond, alert the team "server is down!"
#   - Helps detect issues before users notice
#
#  API Usage:
#   GET http://localhost:5000/health
#   Returns: {"status": "healthy", "service": "campaign-prediction-api", ...}
#
# ⏱  Run time: Instant (just defines the route)
#  Expected output: "Health check route defined!"

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.

    This is a simple endpoint that returns {"status": "healthy"} if the
    server is running. Monitoring tools (like Datadog, New Relic) can
    ping this endpoint to verify the service is alive.

    In production, you might add more checks:
      - Can we connect to the database?
      - Can we load the model file?
      - Is disk space running low?

    For now, we just check if Flask is responding.

    Returns:
        JSON with status and service info (200 OK)
    """

    return jsonify({
        "status": "healthy",
        "service": "campaign-prediction-api",
        "model": "Random Forest MultiOutput"
    }), 200

print(" Health check route defined: GET /health")


# =============================================================================
# CELL 7: Start the Flask Development Server
# =============================================================================
#  What this cell does:
#   - Starts the Flask web server
#   - Makes the app accessible at http://localhost:5000
#   - Prints instructions for the user
#
#  Server Configuration:
#   - host='0.0.0.0': Listen on all network interfaces
#                     (allows access from other computers on your network)
#   - port=5000: Use port 5000 (default for Flask)
#   - debug=True: Enable debug mode (auto-reload on code changes, detailed errors)
#
#  Debug Mode Benefits:
#   - Auto-reloads when you save code changes (no need to restart server)
#   - Shows detailed error pages (helps debugging)
#   - Includes debugger in browser (can inspect variables)
#
#   IMPORTANT - Debug Mode Warning:
#   NEVER use debug=True in production! It exposes security vulnerabilities.
#   For production, use: gunicorn -w 4 app:app
#
# ⏱  Run time: Runs forever (until you stop it with Ctrl+C)
#  Expected output: Server startup message and logs

def start_server():
    """
    Start the Flask development server.

    This function starts the web server and makes your application
    accessible in the browser. Once running, you can:
      1. Open http://localhost:5000 in your browser (see the UI)
      2. Send API requests to http://localhost:5000/api/predict
      3. Check health at http://localhost:5000/health

    The server runs in a loop, handling requests until you stop it.

    How to stop the server:
      - In terminal: Press Ctrl+C
      - In Jupyter: Interrupt the kernel (square stop button)

    Note: If you see "Address already in use", it means port 5000 is taken.
          Either stop the other server or change the port number.
    """

    # Print welcome message
    print("=" * 80)
    print(" CAMPAIGN PREDICTION WEB SERVER STARTING...")
    print("=" * 80)
    print()
    print(" Server will start on: http://localhost:5000")
    print()
    print(" Available Routes:")
    print("    Web Interface:    http://localhost:5000/")
    print("    Prediction API:   http://localhost:5000/api/predict  (POST)")
    print("    Options API:      http://localhost:5000/api/options  (GET)")
    print("    Health Check:     http://localhost:5000/health       (GET)")
    print()
    print(" Tips:")
    print("   - Open the Web Interface URL in your browser to see the UI")
    print("   - Press Ctrl+C to stop the server")
    print("   - Debug mode is ON (auto-reloads on code changes)")
    print()
    print("=" * 80)
    print()

    # Start the Flask server
    # This runs forever until you stop it (Ctrl+C)
    app.run(
        host='0.0.0.0',     # Listen on all network interfaces (allows remote access)
        port=5000,          # Use port 5000
        debug=True          # Enable debug mode (auto-reload, detailed errors)
    )

print(" Server start function defined!")


# =============================================================================
# CELL 8: Run the Server (Execute This!)
# =============================================================================
#   What this cell does:
#   - Checks if this file is being run directly (not imported)
#   - If yes, starts the Flask server
#
#  What if __name__ == '__main__' means:
#   - When you run: python3 app.py → __name__ is '__main__' → server starts
#   - When you import: from app import app → __name__ is 'app' → server doesn't start
#   - This prevents the server from starting when you just want to import functions
#
# ⏱  Run time: Runs forever (web server loop)
#  Expected output: Server startup logs, then waits for requests

# Only run the server if this file is executed directly
# (not when imported as a module)
if __name__ == '__main__':
    start_server()
else:
    print(" Flask app loaded (not running server yet)")
    print("   To start server, run: start_server()")

print("\n" + "=" * 80)
print(" FLASK APPLICATION READY!")
print("=" * 80)
print()
print(" NEXT STEPS:")
print("   1. Run all cells in order")
print("   2. Execute the final cell to start the server")
print("   3. Open http://localhost:5000 in your browser")
print("   4. Start making predictions!")
print()
print("=" * 80)
