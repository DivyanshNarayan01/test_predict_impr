#!/usr/bin/env python3
"""
Campaign Prediction API - Jupyter Notebook Version

This file makes predictions for social media campaigns using a trained AI model.
It's optimized for running in Jupyter Notebook with clear cell divisions and
extensive comments to help non-technical users understand how it works.

=============================================================================
WHAT THIS FILE DOES (IN SIMPLE TERMS):
=============================================================================
Think of this as a "crystal ball" for marketing campaigns. You tell it:
  - How much money you want to spend ($10,000)
  - Which platform (TikTok, Instagram, or Meta/Facebook)
  - What type of campaign (Flood The Feed, Bau, or Mm)
  - What content type (Influencer, Paid Ads, or Owned content)

And it predicts:
  - How many IMPRESSIONS (people who will see your ad)
  - How much ENGAGEMENT (likes, shares, comments, saves)
  - ENGAGEMENT RATE (what % of viewers will interact)
  - Cost metrics (CPM, cost per engagement)

=============================================================================
HOW TO USE IN JUPYTER NOTEBOOK:
=============================================================================
1. Convert this file to a notebook:
   - In Jupyter: File → New → Notebook
   - Copy each CELL block into a separate cell

2. OR use jupytext to convert automatically:
   pip install jupytext
   jupytext --to notebook predict_api_notebook.py

3. Run cells in order (CELL 1, CELL 2, CELL 3, etc.)

=============================================================================
BUSINESS CONTEXT:
=============================================================================
This prediction system helps marketing teams answer questions like:
  - "If I spend $50,000 on TikTok influencers, how many people will see it?"
  - "Should I invest in Instagram paid ads or Meta influencer content?"
  - "What's my expected engagement rate for this campaign?"

The AI model has been trained on 1,000 past campaigns and can predict
performance BEFORE you launch, saving you from wasting money on ineffective
campaigns.

=============================================================================
"""

# =============================================================================
# CELL 1: Import Required Libraries
# =============================================================================
#  What this cell does:
#   - Imports Python libraries needed for predictions
#   - These are like "toolboxes" that provide ready-made functions
#
#  What each library does:
#   - json: Handles data in JSON format (like structured text)
#   - joblib: Loads the saved AI model from disk
#   - numpy: Does math operations on numbers and arrays
#   - pandas: Works with data tables (like Excel spreadsheets)
#   - pathlib: Handles file paths on your computer
#
# ⏱  Run time: <1 second
#  Expected output: No output (libraries load silently)

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from pathlib import Path
import warnings

# Hide warning messages to keep output clean
warnings.filterwarnings('ignore')

print(" All libraries imported successfully!")


# =============================================================================
# CELL 2: Define Custom Error Types
# =============================================================================
#  What this cell does:
#   - Creates special types of errors for different problems
#   - Helps us identify exactly what went wrong if something fails
#
#  Why this matters:
#   - If the model file is missing, we get a "ModelPredictionError"
#   - If you enter invalid data (like negative budget), we get a "ValidationError"
#   - This makes debugging much easier than generic "Error" messages
#
# ⏱  Run time: <1 second
#  Expected output: No output (just defines error types)

class ModelPredictionError(Exception):
    """
    Error raised when something goes wrong with the AI model.

    Examples:
      - Model file not found on disk
      - Model file is corrupted
      - Model fails to make a prediction
    """
    pass


class ValidationError(Exception):
    """
    Error raised when user input is invalid.

    Examples:
      - Budget is negative or zero
      - Platform is "YouTube" (not supported)
      - Content type has a typo
    """
    pass

print(" Custom error types defined!")


# =============================================================================
# CELL 3: Input Validation Function
# =============================================================================
#  What this cell does:
#   - Checks if your campaign inputs are valid before making predictions
#   - Like a "bouncer" at a club - only lets valid data through
#
#  What it validates:
#   1. BUDGET (total_spend):
#      - Must be a number (not text)
#      - Must be positive (can't spend -$100!)
#      - Must be under $1,000,000 (maximum allowed)
#
#   2. PLATFORM:
#      - Must be exactly one of: Meta, TikTok, Instagram
#      - Case-sensitive! "tiktok" won't work, must be "TikTok"
#
#   3. CAMPAIGN TYPE:
#      - Must be: Bau, Mm, or Flood The Feed
#
#   4. CONTENT TYPE:
#      - Must be one of 6 valid types (see list below)
#
# ⏱  Run time: <1 second
#  Expected output: No output if valid, raises error if invalid

def validate_inputs(
    total_spend: float,
    platform: str,
    campaign_type: str,
    content_type: str
) -> None:
    """
    Validate all input parameters before making predictions.

    This is like a quality check before sending your data to the AI.
    If anything is wrong, it stops immediately and tells you what to fix.

    Args:
        total_spend: Your campaign budget in dollars (e.g., 10000.0 = $10,000)
        platform: Which social media platform (Meta, TikTok, or Instagram)
        campaign_type: Strategy type (Bau, Mm, or Flood The Feed)
        content_type: Content classification (6 options available)

    Raises:
        ValidationError: If any input is invalid (with helpful error message)
    """

    # ===========================
    # STEP 1: Validate Budget
    # ===========================

    # Check if total_spend is a number (not text like "ten thousand")
    if not isinstance(total_spend, (int, float)):
        raise ValidationError(
            f" Budget must be a number, but you gave: {type(total_spend).__name__}"
        )

    # Check if budget is positive (can't spend $0 or negative money)
    if total_spend <= 0:
        raise ValidationError(
            f" Budget must be positive, but you gave: ${total_spend:,.2f}"
        )

    # Check if budget is under $1 million (our model's maximum)
    if total_spend > 1_000_000:
        raise ValidationError(
            f" Budget exceeds maximum of $1,000,000. You gave: ${total_spend:,.2f}"
        )

    # ===========================
    # STEP 2: Define Valid Options
    # ===========================

    # These are the ONLY valid platforms (trained in our model)
    VALID_PLATFORMS = ['Meta', 'TikTok', 'Instagram']

    # These are the ONLY valid campaign types
    VALID_CAMPAIGN_TYPES = ['Bau', 'Mm', 'Flood The Feed']

    # These are the ONLY valid content types (6 options)
    VALID_CONTENT_TYPES = [
        'Influencer - Cfg - Boosted Only',          # Influencer content with paid boost
        'Influencer - Ogilvy - Organic Only',       # Organic influencer (no paid boost)
        'Owned - Boosted Only',                     # Your own content with paid boost
        'Owned - Organic Only',                     # Your own content (no paid boost)
        'Paid - Brand',                             # Traditional paid brand advertising
        'Paid - Partnership'                        # Paid partnership/collaboration ads
    ]

    # ===========================
    # STEP 3: Validate Platform
    # ===========================

    if platform not in VALID_PLATFORMS:
        raise ValidationError(
            f" Invalid platform '{platform}'. Must be one of: {', '.join(VALID_PLATFORMS)}\n"
            f"   (Note: Platform names are case-sensitive!)"
        )

    # ===========================
    # STEP 4: Validate Campaign Type
    # ===========================

    if campaign_type not in VALID_CAMPAIGN_TYPES:
        raise ValidationError(
            f" Invalid campaign type '{campaign_type}'. Must be one of: {', '.join(VALID_CAMPAIGN_TYPES)}"
        )

    # ===========================
    # STEP 5: Validate Content Type
    # ===========================

    if content_type not in VALID_CONTENT_TYPES:
        raise ValidationError(
            f" Invalid content type '{content_type}'.\n"
            f"   Must be one of:\n" +
            '\n'.join([f"     - {ct}" for ct in VALID_CONTENT_TYPES])
        )

    # If we get here, all inputs are valid! 

print(" Input validation function defined!")


# =============================================================================
# CELL 4: Model Loading Function
# =============================================================================
# 🤖 What this cell does:
#   - Loads the trained AI model from disk (like opening a saved game)
#   - Also loads the "preprocessor" (prepares data for the model)
#
#  What files it looks for:
#   - models/best_multi_output_model_random_forest_multioutput.pkl (the AI brain)
#   - models/multi_output_preprocessor.pkl (the data preparation tool)
#
#  Why we need both:
#   - The MODEL makes predictions
#   - The PREPROCESSOR converts your inputs into the format the model expects
#   - Like a translator: you speak English, the model speaks "numbers"
#
# ⏱  Run time: 1-2 seconds (loading from disk)
#  Expected output: No output if successful, error if files missing

def load_model_artifacts(models_dir: str = 'models') -> tuple:
    """
    Load the trained AI model and data preprocessor from disk.

    Think of this as "waking up" the AI. The model was trained on 1,000 past
    campaigns and saved to disk. Now we're loading it back into memory so it
    can make predictions for your new campaign.

    Args:
        models_dir: Folder where model files are stored (default: 'models')

    Returns:
        A tuple containing:
          - model: The trained Random Forest AI model
          - preprocessor: Tool that prepares your data for the model

    Raises:
        ModelPredictionError: If model files are missing or can't be loaded
    """

    # Convert string path to a Path object (easier to work with)
    models_path = Path(models_dir)

    # ===========================
    # STEP 1: Check if models folder exists
    # ===========================

    if not models_path.exists():
        raise ModelPredictionError(
            f" Models folder not found: {models_dir}\n"
            f"   Did you run the training script first? (multi_output_training.py)"
        )

    # ===========================
    # STEP 2: Define file paths
    # ===========================

    # The main AI model file (Random Forest with 100 decision trees)
    model_file = models_path / 'best_multi_output_model_random_forest_multioutput.pkl'

    # The preprocessor file (converts text to numbers, scales values)
    preprocessor_file = models_path / 'multi_output_preprocessor.pkl'

    # ===========================
    # STEP 3: Check if files exist
    # ===========================

    if not model_file.exists():
        raise ModelPredictionError(
            f" Model file not found: {model_file}\n"
            f"   Train the model first by running: python3 multi_output_training.py"
        )

    if not preprocessor_file.exists():
        raise ModelPredictionError(
            f" Preprocessor file not found: {preprocessor_file}\n"
            f"   Train the model first by running: python3 multi_output_training.py"
        )

    # ===========================
    # STEP 4: Load files from disk
    # ===========================

    try:
        # Load the AI model (this is the "brain")
        model = joblib.load(model_file)

        # Load the preprocessor (this is the "translator")
        preprocessor = joblib.load(preprocessor_file)

        return model, preprocessor

    except Exception as e:
        raise ModelPredictionError(
            f" Failed to load model files. They may be corrupted.\n"
            f"   Error details: {str(e)}"
        )

print(" Model loading function defined!")


# =============================================================================
# CELL 5: Main Prediction Function (THE CORE!)
# =============================================================================
#  What this cell does:
#   - This is THE MAIN FUNCTION that makes campaign predictions
#   - Takes your campaign details and returns predicted performance
#
#  What it predicts:
#   1. IMPRESSIONS: How many people will see your content
#   2. ENGAGEMENT: How many people will interact (like, share, comment, save)
#   3. ENGAGEMENT RATE: What % of viewers will engage
#   4. CPM: Cost per 1,000 impressions (standard advertising metric)
#   5. COST PER ENGAGEMENT: How much you pay for each interaction
#   6. CONFIDENCE INTERVALS: Range of likely outcomes (95% confidence)
#
#  How it works (simplified):
#   1. Validates your inputs (makes sure they're correct)
#   2. Loads the trained AI model
#   3. Converts your inputs to numbers the model understands
#   4. Feeds the numbers into the AI
#   5. AI returns predictions (in log scale)
#   6. Converts predictions back to normal numbers
#   7. Calculates additional metrics (CPM, engagement rate, etc.)
#   8. Returns everything in a nice dictionary format
#
# ⏱  Run time: 1-3 seconds
#  Expected output: Dictionary with predictions and metrics

def predict_campaign_metrics(
    total_spend: float,
    platform: str = "TikTok",
    campaign_type: str = "Flood The Feed",
    content_type: str = "Influencer - Cfg - Boosted Only",
    models_dir: str = "models",
    return_format: str = "dict"
) -> Union[Dict, str]:
    """
    Predict social media campaign performance using AI.

    THIS IS THE MAIN FUNCTION! Give it your campaign details, and it will
    predict how well your campaign will perform BEFORE you launch it.

    =================================================================
    EXAMPLE BUSINESS USE CASE:
    =================================================================
    You're a marketing manager deciding between two campaigns:
      Option A: $50,000 on TikTok influencers
      Option B: $50,000 on Instagram paid ads

    Run this function twice (once for each option) and compare:
      - Which gets more impressions?
      - Which gets better engagement rate?
      - Which has lower cost per engagement?

    Now you can make a data-driven decision instead of guessing!
    =================================================================

    Args:
        total_spend: Your campaign budget in dollars (e.g., 10000.0 = $10,000)
                    Must be positive and under $1,000,000

        platform: Social media platform where you'll run the campaign
                 Options: 'Meta', 'TikTok', 'Instagram'
                 Default: 'TikTok' (usually highest engagement)

        campaign_type: Your campaign strategy
                      Options: 'Bau', 'Mm', 'Flood The Feed'
                      Default: 'Flood The Feed' (aggressive reach strategy)

        content_type: Type of content you're creating
                     Options:
                       - 'Influencer - Cfg - Boosted Only' (paid influencer boost)
                       - 'Influencer - Ogilvy - Organic Only' (organic influencer)
                       - 'Owned - Boosted Only' (your content + paid boost)
                       - 'Owned - Organic Only' (your content, no boost)
                       - 'Paid - Brand' (traditional brand ads)
                       - 'Paid - Partnership' (partnership ads)
                     Default: 'Influencer - Cfg - Boosted Only'

        models_dir: Where the AI model files are stored
                   Default: 'models' (usually don't need to change this)

        return_format: How to return the results
                      Options: 'dict' (Python dictionary) or 'json' (text)
                      Default: 'dict'

    Returns:
        A dictionary (or JSON string) containing:
        {
          "status": "success",
          "input": {
            "total_spend": 10000.0,
            "platform": "TikTok",
            "campaign_type": "Flood The Feed",
            "content_type": "Influencer - Cfg - Boosted Only"
          },
          "predictions": {
            "impressions": 418398,              ← How many people will see it
            "engagement": 50503,                ← How many will interact
            "engagement_rate": 0.1207,          ← 12.07% engagement rate
            "engagement_rate_pct": "12.07%"     ← Same as above (formatted)
          },
          "metrics": {
            "cpm": 23.90,                       ← Cost per 1,000 impressions
            "cost_per_engagement": 0.20         ← $0.20 per like/share/comment
          },
          "confidence_intervals": {
            "impressions": {
              "lower": 64833,                   ← Minimum likely impressions
              "upper": 771962,                  ← Maximum likely impressions
              "range": "64,833 - 771,962"
            },
            "engagement": {
              "lower": 0,
              "upper": 104886,
              "range": "0 - 104,886"
            },
            "confidence_level": "95%",
            "description": "95% confidence interval based on tree variance"
          }
        }

    Examples:
        >>> # Predict a $10,000 TikTok influencer campaign
        >>> result = predict_campaign_metrics(total_spend=10000.0)
        >>> print(f"Expected impressions: {result['predictions']['impressions']:,}")
        Expected impressions: 418,398

        >>> # Compare Instagram vs TikTok
        >>> instagram = predict_campaign_metrics(total_spend=5000, platform="Instagram")
        >>> tiktok = predict_campaign_metrics(total_spend=5000, platform="TikTok")
        >>> print(f"Instagram engagement: {instagram['predictions']['engagement']:,}")
        >>> print(f"TikTok engagement: {tiktok['predictions']['engagement']:,}")
    """

    try:
        # ===========================
        # STEP 1: Validate Inputs
        # ===========================
        # Before doing anything, make sure the user gave us valid data
        # This prevents errors later and gives helpful feedback if something's wrong

        validate_inputs(total_spend, platform, campaign_type, content_type)

        # ===========================
        # STEP 2: Load AI Model
        # ===========================
        # Load the trained Random Forest model from disk
        # This model learned patterns from 1,000 past campaigns

        model, preprocessor = load_model_artifacts(models_dir)

        # ===========================
        # STEP 3: Prepare Input Data
        # ===========================
        # Create a dictionary with your campaign details

        campaign_data = {
            'Platform': platform,
            'campaign_type': campaign_type,
            'content_type': content_type,
            'total_spend': total_spend
        }

        # Convert to a pandas DataFrame (table format)
        # The model expects data in table format, even for just one campaign
        campaign_df = pd.DataFrame([campaign_data])

        # ===========================
        # STEP 4: Feature Engineering
        # ===========================
        # Apply LOG TRANSFORMATION to the budget
        #
        # Why? Because spending has "diminishing returns":
        #   - Doubling your budget doesn't double your impressions
        #   - $20,000 doesn't get 2x the results of $10,000
        #   - Maybe it gets 1.8x the results (diminishing returns)
        #
        # Log transformation captures this pattern mathematically

        campaign_df['Log_Spend_Total'] = np.log(campaign_df['total_spend'] + 1)

        # ===========================
        # STEP 5: Select Features in Correct Order
        # ===========================
        # The model expects features in a SPECIFIC ORDER (same as training)
        # Like a lock combination - order matters!

        categorical_features = ['Platform', 'campaign_type', 'content_type']
        numerical_features = ['Log_Spend_Total']
        X = campaign_df[categorical_features + numerical_features]

        # ===========================
        # STEP 6: Preprocess Data
        # ===========================
        # The preprocessor does two things:
        #   1. Converts text to numbers (Platform "TikTok" → [0, 1, 0])
        #   2. Scales numbers to a standard range
        #
        # This is the "translation" step - converting human-readable data
        # into the format the AI model understands

        X_processed = preprocessor.transform(X)

        # ===========================
        # STEP 7: Make Predictions
        # ===========================
        # Feed the processed data into the AI model
        # The model returns TWO predictions (multi-output):
        #   - predictions_log[0] = log(impressions)
        #   - predictions_log[1] = log(engagement)
        #
        # Note: Predictions are in LOG SCALE (we'll convert them back next)

        predictions_log = model.predict(X_processed)[0]

        # ===========================
        # STEP 8: Convert from Log Scale to Normal Scale
        # ===========================
        # The model predicts in log scale, so we need to reverse the transformation
        #
        # Mathematical note:
        #   - If model predicts log(impressions) = 12.5
        #   - Real impressions = exp(12.5) - 1 = 267,114
        #
        # We use expm1() which is exp(x) - 1 (more numerically stable)

        impressions_pred = np.expm1(predictions_log[0])
        engagement_pred = np.expm1(predictions_log[1])

        # ===========================
        # STEP 9: Calculate Confidence Intervals (Advanced!)
        # ===========================
        # This section calculates the RANGE of likely outcomes
        #
        # Instead of saying "you'll get exactly 400,000 impressions",
        # we say "you'll likely get between 65,000 and 770,000 impressions (95% confidence)"
        #
        # How it works:
        #   - Our Random Forest has 100 decision trees
        #   - Each tree makes a slightly different prediction
        #   - We calculate the standard deviation (spread) of all 100 predictions
        #   - 95% confidence interval = prediction ± 1.96 × standard_deviation
        #
        # This is more honest than a single point estimate!

        impressions_confidence_lower = None
        impressions_confidence_upper = None
        engagement_confidence_lower = None
        engagement_confidence_upper = None

        try:
            # Check if model has multiple estimators (Random Forest does)
            if hasattr(model, 'estimators_'):

                # --- Confidence Interval for IMPRESSIONS ---

                # Get the impressions estimator (first one)
                impressions_estimator = model.estimators_[0]

                if hasattr(impressions_estimator, 'estimators_'):
                    # Get predictions from all 100 individual trees
                    impressions_tree_preds = []
                    for tree in impressions_estimator.estimators_:
                        tree_pred_log = tree.predict(X_processed)[0]
                        impressions_tree_preds.append(np.expm1(tree_pred_log))

                    # Calculate spread (standard deviation)
                    impressions_std = np.std(impressions_tree_preds)

                    # 95% confidence interval
                    impressions_confidence_lower = max(0, impressions_pred - 1.96 * impressions_std)
                    impressions_confidence_upper = impressions_pred + 1.96 * impressions_std

                # --- Confidence Interval for ENGAGEMENT ---

                # Get the engagement estimator (second one)
                engagement_estimator = model.estimators_[1]

                if hasattr(engagement_estimator, 'estimators_'):
                    # Get predictions from all 100 individual trees
                    engagement_tree_preds = []
                    for tree in engagement_estimator.estimators_:
                        tree_pred_log = tree.predict(X_processed)[0]
                        engagement_tree_preds.append(np.expm1(tree_pred_log))

                    # Calculate spread
                    engagement_std = np.std(engagement_tree_preds)

                    # 95% confidence interval
                    engagement_confidence_lower = max(0, engagement_pred - 1.96 * engagement_std)
                    engagement_confidence_upper = engagement_pred + 1.96 * engagement_std

        except Exception:
            # If confidence interval calculation fails, just skip it
            # We'll still return the main predictions
            pass

        # ===========================
        # STEP 10: Calculate Business Metrics
        # ===========================
        # Now we calculate additional useful metrics for marketers

        # ENGAGEMENT RATE: What % of people who see it will interact?
        # Example: 50,000 engagement / 400,000 impressions = 12.5% engagement rate
        engagement_rate = engagement_pred / impressions_pred if impressions_pred > 0 else 0

        # CPM (Cost Per Mille): Cost per 1,000 impressions
        # This is a standard advertising metric
        # Example: $10,000 / 400,000 impressions × 1000 = $25 CPM
        cpm = (total_spend / impressions_pred * 1000) if impressions_pred > 0 else 0

        # COST PER ENGAGEMENT: How much you pay for each like/share/comment
        # Example: $10,000 / 50,000 engagement = $0.20 per engagement
        cost_per_engagement = (total_spend / engagement_pred) if engagement_pred > 0 else 0

        # ===========================
        # STEP 11: Prepare Response Dictionary
        # ===========================
        # Package everything into a nice, organized dictionary

        response = {
            "status": "success",
            "input": {
                "total_spend": float(total_spend),
                "platform": platform,
                "campaign_type": campaign_type,
                "content_type": content_type
            },
            "predictions": {
                "impressions": int(round(impressions_pred)),
                "engagement": int(round(engagement_pred)),
                "engagement_rate": round(engagement_rate, 4),
                "engagement_rate_pct": f"{engagement_rate * 100:.2f}%"
            },
            "metrics": {
                "cpm": round(cpm, 2),
                "cost_per_engagement": round(cost_per_engagement, 2)
            }
        }

        # Add confidence intervals if we calculated them
        if impressions_confidence_lower is not None:
            response["confidence_intervals"] = {
                "impressions": {
                    "lower": int(round(impressions_confidence_lower)),
                    "upper": int(round(impressions_confidence_upper)),
                    "range": f"{int(round(impressions_confidence_lower)):,} - {int(round(impressions_confidence_upper)):,}"
                },
                "engagement": {
                    "lower": int(round(engagement_confidence_lower)),
                    "upper": int(round(engagement_confidence_upper)),
                    "range": f"{int(round(engagement_confidence_lower)):,} - {int(round(engagement_confidence_upper)):,}"
                },
                "confidence_level": "95%",
                "description": "95% confidence interval based on tree variance in Random Forest"
            }

        # ===========================
        # STEP 12: Return Results
        # ===========================
        # Return either as dictionary or JSON string (based on user preference)

        if return_format.lower() == "json":
            return json.dumps(response, indent=2)
        else:
            return response

    # ===========================
    # ERROR HANDLING
    # ===========================
    # If anything goes wrong, return a helpful error message

    except (ValidationError, ModelPredictionError) as e:
        # Known errors (validation or model issues)
        error_response = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

        if return_format.lower() == "json":
            return json.dumps(error_response, indent=2)
        else:
            return error_response

    except Exception as e:
        # Unexpected errors (shouldn't happen, but just in case)
        error_response = {
            "status": "error",
            "error_type": "UnexpectedError",
            "error_message": f"Unexpected error occurred: {str(e)}"
        }

        if return_format.lower() == "json":
            return json.dumps(error_response, indent=2)
        else:
            return error_response

print(" Main prediction function defined!")


# =============================================================================
# CELL 6: Demo Examples (Try These!)
# =============================================================================
#  What this cell does:
#   - Shows 5 example predictions to help you understand how it works
#   - You can run these examples to see the function in action
#
#  Examples included:
#   1. Basic prediction with default parameters (TikTok influencer)
#   2. Custom Instagram campaign
#   3. JSON output format
#   4. Error handling - invalid platform
#   5. Error handling - negative budget
#
# ⏱  Run time: 5-10 seconds (runs 5 predictions)
#  Expected output: 5 prediction results and 2 error examples

def run_demo_examples():
    """
    Run example predictions to demonstrate how the function works.

    This is like a "playground" where you can see the prediction function
    in action without risking anything. All examples are safe to run!
    """

    print("=" * 80)
    print(" CAMPAIGN PREDICTION API - DEMO EXAMPLES")
    print("=" * 80)

    # ========================================
    # EXAMPLE 1: Basic Prediction
    # ========================================
    print("\n[Example 1] Basic prediction with default parameters:")
    print("-" * 80)
    print(" Campaign: $10,000 on TikTok influencer (Flood The Feed)")
    print()

    result1 = predict_campaign_metrics(total_spend=10000.0)
    print(json.dumps(result1, indent=2))

    # ========================================
    # EXAMPLE 2: Custom Instagram Campaign
    # ========================================
    print("\n[Example 2] Custom Instagram campaign:")
    print("-" * 80)
    print(" Campaign: $5,000 on Instagram paid brand ads (Bau strategy)")
    print()

    result2 = predict_campaign_metrics(
        total_spend=5000.0,
        platform="Instagram",
        campaign_type="Bau",
        content_type="Paid - Brand"
    )
    print(json.dumps(result2, indent=2))

    # ========================================
    # EXAMPLE 3: JSON Output Format
    # ========================================
    print("\n[Example 3] Same prediction, but returned as JSON string:")
    print("-" * 80)
    print(" Campaign: $15,000 on TikTok influencer")
    print()

    result3 = predict_campaign_metrics(
        total_spend=15000.0,
        platform="TikTok",
        content_type="Influencer - Cfg - Boosted Only",
        return_format="json"
    )
    print(result3)

    # ========================================
    # EXAMPLE 4: Error Handling - Invalid Platform
    # ========================================
    print("\n[Example 4] Error handling - invalid platform:")
    print("-" * 80)
    print(" Trying to use 'YouTube' (not supported)")
    print()

    result4 = predict_campaign_metrics(
        total_spend=5000.0,
        platform="YouTube"  #  Invalid! Only Meta, TikTok, Instagram supported
    )
    print(json.dumps(result4, indent=2))

    # ========================================
    # EXAMPLE 5: Error Handling - Negative Budget
    # ========================================
    print("\n[Example 5] Error handling - negative budget:")
    print("-" * 80)
    print(" Trying to use negative budget: -$1,000")
    print()

    result5 = predict_campaign_metrics(
        total_spend=-1000.0  #  Invalid! Budget must be positive
    )
    print(json.dumps(result5, indent=2))

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE!")
    print("=" * 80)
    print("\n TIP: Modify the examples above to test your own campaigns!")

print(" Demo function defined! Run run_demo_examples() to see it in action.")


# =============================================================================
# CELL 7: Run the Demo (Optional)
# =============================================================================
#   What this cell does:
#   - Runs all the demo examples from Cell 6
#   - Shows you how the prediction function works in practice
#
# ⏱  Run time: 5-10 seconds
#  Expected output: 5 example predictions with results

# Uncomment the line below to run the demo:
# run_demo_examples()

print(" Notebook ready! Run run_demo_examples() to see examples, or create your own predictions!")
