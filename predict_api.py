#!/usr/bin/env python3
"""
Campaign Prediction API Function (Pure Pandas Preprocessing - No Disk Saves)

A robust function that loads the pickled multi-output model and makes predictions
with comprehensive error handling and validation.

PURE PANDAS APPROACH:
- No preprocessor pickle file needed (preprocessing metadata stored in JSON)
- Preprocessing recreated inline using pure pandas operations
- Reduced disk I/O and increased transparency

Usage:
    from predict_api import predict_campaign_metrics

    result = predict_campaign_metrics(
        total_spend=10000.0,
        platform="TikTok",
        campaign_type="Flood The Feed",
        content_type="Influencer - Cfg - Boosted Only"
    )

    print(result)
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelPredictionError(Exception):
    """Custom exception for model prediction errors."""
    pass


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_inputs(
    total_spend: float,
    platform: str,
    campaign_type: str,
    content_type: str
) -> None:
    """
    Validate all input parameters.

    Args:
        total_spend: Campaign budget (must be positive)
        platform: Social media platform
        campaign_type: Type of campaign strategy
        content_type: Content classification

    Raises:
        ValidationError: If any input is invalid
    """
    # Validate total_spend
    if not isinstance(total_spend, (int, float)):
        raise ValidationError(f"total_spend must be a number, got {type(total_spend).__name__}")

    if total_spend <= 0:
        raise ValidationError(f"total_spend must be positive, got {total_spend}")

    if total_spend > 1_000_000:
        raise ValidationError(f"total_spend exceeds maximum allowed (1,000,000), got {total_spend}")

    # Valid categorical values (based on training data - UPPERCASE)
    VALID_PLATFORMS = ['META', 'TIKTOK', 'INSTAGRAM']
    VALID_CAMPAIGN_TYPES = ['BAU', 'MM', 'FLOOD THE FEED']
    VALID_CONTENT_TYPES = [
        'INFLUENCER - CFG - BOOSTED ONLY',
        'INFLUENCER - OGILVY - ORGANIC ONLY',
        'INFLUENCER - OGILVY - BOOSTED ONLY',
        'OWNED - BOOSTED ONLY',
        'PAID - BRAND',
        'PAID - PARTNERSHIP'
    ]

    # Validate categorical inputs
    if platform not in VALID_PLATFORMS:
        raise ValidationError(
            f"Invalid platform '{platform}'. Must be one of: {', '.join(VALID_PLATFORMS)}"
        )

    if campaign_type not in VALID_CAMPAIGN_TYPES:
        raise ValidationError(
            f"Invalid campaign_type '{campaign_type}'. Must be one of: {', '.join(VALID_CAMPAIGN_TYPES)}"
        )

    if content_type not in VALID_CONTENT_TYPES:
        raise ValidationError(
            f"Invalid content_type '{content_type}'. Must be one of: {', '.join(VALID_CONTENT_TYPES)}"
        )


def preprocess_features(campaign_df: pd.DataFrame, preprocessing_metadata: dict) -> np.ndarray:
    """
    Recreate preprocessing using manual one-hot encoding (fixes single-row encoding bug).

    Args:
        campaign_df: DataFrame with raw features
        preprocessing_metadata: Dictionary with preprocessing information

    Returns:
        Preprocessed features as numpy array

    Raises:
        ModelPredictionError: If preprocessing fails
    """
    try:
        categorical_features = preprocessing_metadata['categorical_features']
        scaler_mean = preprocessing_metadata['scaler_mean']
        scaler_std = preprocessing_metadata['scaler_std']
        expected_columns = preprocessing_metadata['feature_columns']

        # Create a dictionary to hold the encoded features
        # Start with the numerical feature
        encoded_data = {
            'Log_Spend_Total': campaign_df['Log_Spend_Total'].values[0]
        }

        # Manually create one-hot encoded columns
        # Initialize all categorical columns to 0
        for col in expected_columns:
            if col != 'Log_Spend_Total':
                encoded_data[col] = 0

        # Set the appropriate one-hot encoded columns to 1 based on input values
        for cat_feature in categorical_features:
            value = campaign_df[cat_feature].values[0]
            # Create column name in the format: feature_value
            # Note: pd.get_dummies with drop_first=True drops the first category
            # If the column doesn't exist, it means it's the dropped baseline category
            # In that case, all one-hot columns for that feature should remain 0
            column_name = f"{cat_feature}_{value}"
            if column_name in expected_columns:
                encoded_data[column_name] = 1
            # If not in expected_columns, it's the baseline - do nothing (leave as 0)

        # Create DataFrame with proper column order
        df_encoded = pd.DataFrame([encoded_data])[expected_columns]

        # Standardize numerical feature
        df_encoded['Log_Spend_Total'] = (df_encoded['Log_Spend_Total'] - scaler_mean) / scaler_std

        return df_encoded.values

    except Exception as e:
        raise ModelPredictionError(f"Preprocessing failed: {str(e)}")


def load_model_artifacts(models_dir: str = 'models') -> tuple:
    """
    Load the trained model and preprocessing metadata from disk.

    Args:
        models_dir: Directory containing model files

    Returns:
        Tuple of (model, preprocessing_metadata)

    Raises:
        ModelPredictionError: If model files cannot be loaded
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        raise ModelPredictionError(f"Models directory not found: {models_dir}")

    # Auto-detect the best model file (supports any model type: random_forest, xgboost, lightgbm)
    model_files = list(models_path.glob('best_multi_output_model_*.pkl'))

    if not model_files:
        raise ModelPredictionError(f"No model file found matching pattern 'best_multi_output_model_*.pkl' in {models_dir}")

    if len(model_files) > 1:
        # If multiple models exist, prioritize based on metadata or use first one
        print(f"Warning: Multiple model files found: {[f.name for f in model_files]}. Using first one.")

    model_file = model_files[0]
    metadata_file = models_path / 'multi_output_model_metadata.json'

    print(f"Loading model: {model_file.name}")

    if not metadata_file.exists():
        raise ModelPredictionError(f"Metadata file not found: {metadata_file}")

    try:
        model = joblib.load(model_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Extract preprocessing metadata
        if 'preprocessing' not in metadata:
            raise ModelPredictionError("Metadata file missing preprocessing information")

        preprocessing_metadata = metadata['preprocessing']
        return model, preprocessing_metadata
    except Exception as e:
        raise ModelPredictionError(f"Failed to load model artifacts: {str(e)}")


def predict_campaign_metrics(
    total_spend: float,
    platform: str = "TIKTOK",
    campaign_type: str = "FLOOD THE FEED",
    content_type: str = "INFLUENCER - CFG - BOOSTED ONLY",
    models_dir: str = "models",
    return_format: str = "dict"
) -> Union[Dict, str]:
    """
    Predict campaign impressions and engagement metrics.

    Args:
        total_spend: Campaign budget in dollars (must be > 0)
        platform: Social media platform - one of ['META', 'TIKTOK', 'INSTAGRAM']
        campaign_type: Campaign strategy - one of ['BAU', 'MM', 'FLOOD THE FEED']
        content_type: Content classification - one of:
            - 'INFLUENCER - CFG - BOOSTED ONLY'
            - 'INFLUENCER - OGILVY - ORGANIC ONLY'
            - 'INFLUENCER - OGILVY - BOOSTED ONLY'
            - 'OWNED - BOOSTED ONLY'
            - 'PAID - BRAND'
            - 'PAID - PARTNERSHIP'
        models_dir: Directory containing model files (default: 'models')
        return_format: Output format - 'dict' or 'json' (default: 'dict')

    Returns:
        Dictionary or JSON string containing:
        {
            "status": "success",
            "input": {
                "total_spend": float,
                "platform": str,
                "campaign_type": str,
                "content_type": str
            },
            "predictions": {
                "impressions": int,
                "engagement": int,
                "engagement_rate": float,
                "engagement_rate_pct": str
            },
            "metrics": {
                "cpm": float,  # Cost per 1000 impressions
                "cost_per_engagement": float
            }
        }

    Raises:
        ValidationError: If input validation fails
        ModelPredictionError: If model loading or prediction fails

    Examples:
        >>> # Basic usage
        >>> result = predict_campaign_metrics(total_spend=10000.0)
        >>> print(result['predictions']['impressions'])

        >>> # Custom campaign
        >>> result = predict_campaign_metrics(
        ...     total_spend=5000.0,
        ...     platform="INSTAGRAM",
        ...     campaign_type="BAU",
        ...     content_type="PAID - BRAND"
        ... )

        >>> # Get JSON output
        >>> json_result = predict_campaign_metrics(
        ...     total_spend=10000.0,
        ...     return_format="json"
        ... )
    """
    try:
        # Validate inputs
        validate_inputs(total_spend, platform, campaign_type, content_type)

        # Load model and preprocessing metadata
        model, preprocessing_metadata = load_model_artifacts(models_dir)

        # Prepare input data
        campaign_data = {
            'Platform': platform,
            'campaign_type': campaign_type,
            'content_type': content_type,
            'total_spend': total_spend
        }

        # Create DataFrame
        campaign_df = pd.DataFrame([campaign_data])

        # Add engineered feature (log transformation)
        campaign_df['Log_Spend_Total'] = np.log(campaign_df['total_spend'] + 1)

        # Select features in the correct order (must match training)
        categorical_features = ['Platform', 'campaign_type', 'content_type']
        numerical_features = ['Log_Spend_Total']
        X = campaign_df[categorical_features + numerical_features]

        # Preprocess features using pure pandas (no saved preprocessor)
        X_processed = preprocess_features(X, preprocessing_metadata)

        # Make predictions (returns log-transformed values)
        predictions_log = model.predict(X_processed)[0]

        # Transform predictions back to original scale
        impressions_pred = np.expm1(predictions_log[0])
        engagement_pred = np.expm1(predictions_log[1])

        # Calculate prediction intervals using individual tree predictions (for Random Forest)
        # This provides confidence intervals based on tree variance
        impressions_confidence_lower = None
        impressions_confidence_upper = None
        engagement_confidence_lower = None
        engagement_confidence_upper = None

        try:
            # MultiOutputRegressor wraps individual estimators (one per target)
            if hasattr(model, 'estimators_'):
                # For MultiOutputRegressor, each estimator is a RandomForestRegressor
                # estimators_[0] is for impressions, estimators_[1] is for engagement

                # Get individual tree predictions for impressions
                impressions_estimator = model.estimators_[0]
                if hasattr(impressions_estimator, 'estimators_'):
                    impressions_tree_preds = []
                    for tree in impressions_estimator.estimators_:
                        tree_pred_log = tree.predict(X_processed)[0]
                        impressions_tree_preds.append(np.expm1(tree_pred_log))

                    impressions_std = np.std(impressions_tree_preds)
                    impressions_confidence_lower = max(0, impressions_pred - 1.96 * impressions_std)
                    impressions_confidence_upper = impressions_pred + 1.96 * impressions_std

                # Get individual tree predictions for engagement
                engagement_estimator = model.estimators_[1]
                if hasattr(engagement_estimator, 'estimators_'):
                    engagement_tree_preds = []
                    for tree in engagement_estimator.estimators_:
                        tree_pred_log = tree.predict(X_processed)[0]
                        engagement_tree_preds.append(np.expm1(tree_pred_log))

                    engagement_std = np.std(engagement_tree_preds)
                    engagement_confidence_lower = max(0, engagement_pred - 1.96 * engagement_std)
                    engagement_confidence_upper = engagement_pred + 1.96 * engagement_std
        except Exception:
            # If confidence calculation fails, continue without it
            pass

        # Calculate derived metrics
        engagement_rate = engagement_pred / impressions_pred if impressions_pred > 0 else 0
        cpm = (total_spend / impressions_pred * 1000) if impressions_pred > 0 else 0
        cost_per_engagement = (total_spend / engagement_pred) if engagement_pred > 0 else 0

        # Prepare response
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
                "engagement_rate": float(round(engagement_rate, 4)),
                "engagement_rate_pct": f"{engagement_rate * 100:.2f}%"
            },
            "metrics": {
                "cpm": float(round(cpm, 2)),  # Cost per 1000 impressions
                "cost_per_engagement": float(round(cost_per_engagement, 2))
            }
        }

        # Add confidence intervals if calculated
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

        # Return in requested format
        if return_format.lower() == "json":
            return json.dumps(response, indent=2)
        else:
            return response

    except (ValidationError, ModelPredictionError) as e:
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
        error_response = {
            "status": "error",
            "error_type": "UnexpectedError",
            "error_message": f"Unexpected error occurred: {str(e)}"
        }

        if return_format.lower() == "json":
            return json.dumps(error_response, indent=2)
        else:
            return error_response


def main():
    """Demo function showing various usage examples."""

    print("=" * 80)
    print("CAMPAIGN PREDICTION API - DEMO")
    print("=" * 80)

    # Example 1: Basic usage with defaults
    print("\n[1] Basic prediction with default parameters:")
    print("-" * 80)
    result1 = predict_campaign_metrics(total_spend=10000.0)
    print(json.dumps(result1, indent=2))

    # Example 2: Custom campaign configuration
    print("\n[2] Custom Instagram campaign:")
    print("-" * 80)
    result2 = predict_campaign_metrics(
        total_spend=5000.0,
        platform="INSTAGRAM",
        campaign_type="BAU",
        content_type="PAID - BRAND"
    )
    print(json.dumps(result2, indent=2))

    # Example 3: JSON output format
    print("\n[3] JSON output format:")
    print("-" * 80)
    result3 = predict_campaign_metrics(
        total_spend=15000.0,
        platform="TIKTOK",
        content_type="INFLUENCER - CFG - BOOSTED ONLY",
        return_format="json"
    )
    print(result3)

    # Example 4: Error handling - invalid input
    print("\n[4] Error handling - invalid platform:")
    print("-" * 80)
    result4 = predict_campaign_metrics(
        total_spend=5000.0,
        platform="YOUTUBE"  # Invalid
    )
    print(json.dumps(result4, indent=2))

    # Example 5: Error handling - negative budget
    print("\n[5] Error handling - negative budget:")
    print("-" * 80)
    result5 = predict_campaign_metrics(
        total_spend=-1000.0  # Invalid
    )
    print(json.dumps(result5, indent=2))

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
