//#!/usr/bin/env python3
"""
Business-Friendly Budget Optimization Engine

Designed for business users who input:
- Total budget (e.g., $10M)
- Percentage distribution across Platform, campaign_type, content_type

Quarter feature removed.

This module provides simple interfaces for:
1. Predicting performance based on user's budget distribution
2. Optimizing budget allocation to maximize impressions + engagement
3. Comparing user allocation vs AI-optimized allocation
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class BusinessBudgetPredictor:
    """
    Simplified predictor for business users - no Quarter needed.
    """

    def __init__(self, model_path, preprocessor_path):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained multi-output model
            preprocessor_path: Path to preprocessor
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        # Feature order
        self.categorical_features = ['Platform', 'campaign_type', 'content_type']
        self.numerical_features = ['Log_Spend_Total']

        # Valid values
        self.valid_platforms = ['Meta', 'TikTok', 'Instagram']
        self.valid_campaign_types = ['Bau', 'Mm', 'Flood The Feed']
        self.valid_content_types = [
            'Paid - Brand',
            'Influencer - Cfg - Boosted Only',
            'Paid - Partnership',
            'Owned - Boosted Only',
            'Influencer - Ogilvy - Organic Only',
            'Influencer - Ogilvy - Boosted Only'
        ]

    def predict_from_allocation(self, total_budget: float, allocation: List[Dict]) -> Dict:
        """
        Predict performance from budget allocation.

        Args:
            total_budget: Total budget (e.g., 10000000)
            allocation: List of dicts with:
                - Platform: str
                - campaign_type: str
                - content_type: str
                - budget_percentage: float (e.g., 25.0 for 25%)

        Returns:
            dict with predictions and metrics
        """
        # Validate allocation sums to 100%
        total_pct = sum(a['budget_percentage'] for a in allocation)
        if abs(total_pct - 100.0) > 0.01:
            raise ValueError(f"Budget percentages must sum to 100%, got {total_pct}%")

        # Convert to DataFrame
        df = pd.DataFrame(allocation)
        df['total_spend'] = df['budget_percentage'] / 100 * total_budget

        # Add engineered feature
        df['Log_Spend_Total'] = np.log(df['total_spend'] + 1)

        # Select features
        X = df[['Platform', 'campaign_type', 'content_type', 'Log_Spend_Total']]

        # Preprocess
        X_processed = self.preprocessor.transform(X)

        # Predict
        predictions_log = self.model.predict(X_processed)

        # Transform to original scale
        df['predicted_impressions'] = np.expm1(predictions_log[:, 0])
        df['predicted_engagement'] = np.expm1(predictions_log[:, 1])
        df['predicted_engagement_rate'] = df['predicted_engagement'] / df['predicted_impressions']
        df['cpm'] = df['total_spend'] / df['predicted_impressions'] * 1000
        df['cost_per_engagement'] = df['total_spend'] / df['predicted_engagement']

        # Calculate totals
        total_impressions = df['predicted_impressions'].sum()
        total_engagement = df['predicted_engagement'].sum()

        return {
            'allocation_detail': df[[
                'Platform', 'campaign_type', 'content_type',
                'budget_percentage', 'total_spend',
                'predicted_impressions', 'predicted_engagement',
                'predicted_engagement_rate', 'cpm', 'cost_per_engagement'
            ]].to_dict(orient='records'),
            'summary': {
                'total_budget': float(total_budget),
                'total_impressions': float(total_impressions),
                'total_engagement': float(total_engagement),
                'overall_engagement_rate': float(total_engagement / total_impressions),
                'average_cpm': float(df['cpm'].mean()),
                'average_cost_per_engagement': float(df['cost_per_engagement'].mean())
            }
        }

    def predict_with_confidence(self, total_budget: float, allocation: List[Dict],
                               n_bootstrap: int = 50) -> Dict:
        """
        Predict with confidence intervals (faster bootstrap for business users).

        Args:
            total_budget: Total budget
            allocation: Budget allocation list
            n_bootstrap: Number of bootstrap samples (default 50 for speed)

        Returns:
            dict with predictions and confidence intervals
        """
        # Get point prediction
        point_pred = self.predict_from_allocation(total_budget, allocation)

        # Bootstrap for confidence intervals
        impressions_samples = []
        engagement_samples = []

        base_impressions = point_pred['summary']['total_impressions']
        base_engagement = point_pred['summary']['total_engagement']

        # Simulate uncertainty (25% std for impressions, 30% for engagement)
        for _ in range(n_bootstrap):
            imp_sample = np.random.normal(base_impressions, base_impressions * 0.25)
            eng_sample = np.random.normal(base_engagement, base_engagement * 0.30)
            impressions_samples.append(max(0, imp_sample))
            engagement_samples.append(max(0, eng_sample))

        # Calculate 95% CI
        impressions_ci = np.percentile(impressions_samples, [2.5, 97.5])
        engagement_ci = np.percentile(engagement_samples, [2.5, 97.5])

        return {
            **point_pred,
            'confidence_intervals': {
                'impressions': {
                    'lower': float(impressions_ci[0]),
                    'upper': float(impressions_ci[1]),
                    'confidence_level': 0.95
                },
                'engagement': {
                    'lower': float(engagement_ci[0]),
                    'upper': float(engagement_ci[1]),
                    'confidence_level': 0.95
                }
            }
        }


class BusinessBudgetOptimizer:
    """
    Optimizer for business users - input percentages across segments.
    """

    def __init__(self, predictor: BusinessBudgetPredictor):
        """
        Initialize optimizer.

        Args:
            predictor: BusinessBudgetPredictor instance
        """
        self.predictor = predictor

    def optimize_allocation(self, total_budget: float,
                           objective: str = 'both',
                           weights: Dict[str, float] = None,
                           top_n: int = 15) -> Dict:
        """
        Optimize budget allocation across Platform, campaign_type, content_type.

        Args:
            total_budget: Total budget to allocate
            objective: 'impressions', 'engagement', or 'both'
            weights: {'impressions': 0.5, 'engagement': 0.5} if objective='both'
            top_n: Number of top campaign combinations to use

        Returns:
            dict with optimized allocation
        """
        if weights is None:
            weights = {'impressions': 0.5, 'engagement': 0.5}

        # Generate all possible combinations
        combinations = []
        for platform in self.predictor.valid_platforms:
            for campaign_type in self.predictor.valid_campaign_types:
                for content_type in self.predictor.valid_content_types:
                    combinations.append({
                        'Platform': platform,
                        'campaign_type': campaign_type,
                        'content_type': content_type
                    })

        # Test each with small budget to rank
        test_budget = 1000
        results = []

        for combo in combinations:
            test_allocation = [{
                **combo,
                'budget_percentage': 100.0
            }]

            try:
                pred = self.predictor.predict_from_allocation(test_budget, test_allocation)
                impr_per_dollar = pred['summary']['total_impressions'] / test_budget
                eng_per_dollar = pred['summary']['total_engagement'] / test_budget

                results.append({
                    **combo,
                    'impressions_per_dollar': impr_per_dollar,
                    'engagement_per_dollar': eng_per_dollar
                })
            except:
                continue

        results_df = pd.DataFrame(results)

        # Score each combination
        if objective == 'impressions':
            results_df['score'] = results_df['impressions_per_dollar']
        elif objective == 'engagement':
            results_df['score'] = results_df['engagement_per_dollar']
        else:  # both
            impr_norm = results_df['impressions_per_dollar'] / results_df['impressions_per_dollar'].max()
            eng_norm = results_df['engagement_per_dollar'] / results_df['engagement_per_dollar'].max()
            results_df['score'] = weights['impressions'] * impr_norm + weights['engagement'] * eng_norm

        # Select top N
        top_combos = results_df.nlargest(top_n, 'score')

        # Allocate budget proportionally to scores
        top_combos['budget_weight'] = top_combos['score'] / top_combos['score'].sum()
        top_combos['budget_percentage'] = top_combos['budget_weight'] * 100

        # Create allocation list
        optimized_allocation = top_combos[[
            'Platform', 'campaign_type', 'content_type', 'budget_percentage'
        ]].to_dict(orient='records')

        # Get final predictions
        final_pred = self.predictor.predict_from_allocation(total_budget, optimized_allocation)

        return {
            'optimized_allocation': optimized_allocation,
            'predicted_performance': final_pred['summary'],
            'allocation_detail': final_pred['allocation_detail'],
            'optimization_params': {
                'objective': objective,
                'weights': weights,
                'n_campaigns': len(optimized_allocation)
            }
        }

    def compare_allocations(self, total_budget: float,
                           user_allocation: List[Dict]) -> Dict:
        """
        Compare user's allocation vs AI-optimized allocation.

        Args:
            total_budget: Total budget
            user_allocation: User's allocation (list of dicts with Platform,
                           campaign_type, content_type, budget_percentage)

        Returns:
            dict with comparison
        """
        # Get user's predicted performance
        user_pred = self.predictor.predict_from_allocation(total_budget, user_allocation)

        # Get optimized allocation
        optimized = self.optimize_allocation(total_budget, objective='both')

        # Calculate improvement
        user_impr = user_pred['summary']['total_impressions']
        user_eng = user_pred['summary']['total_engagement']
        opt_impr = optimized['predicted_performance']['total_impressions']
        opt_eng = optimized['predicted_performance']['total_engagement']

        improvement = {
            'impressions_gain': float(opt_impr - user_impr),
            'impressions_gain_percentage': float((opt_impr / user_impr - 1) * 100),
            'engagement_gain': float(opt_eng - user_eng),
            'engagement_gain_percentage': float((opt_eng / user_eng - 1) * 100),
            'engagement_rate_improvement': float(
                optimized['predicted_performance']['overall_engagement_rate'] -
                user_pred['summary']['overall_engagement_rate']
            )
        }

        return {
            'user_allocation': {
                'allocation': user_allocation,
                'predicted_performance': user_pred['summary'],
                'detail': user_pred['allocation_detail']
            },
            'optimized_allocation': optimized,
            'improvement': improvement,
            'recommendation': 'optimized' if improvement['impressions_gain'] > 0 else 'user',
            'improvement_summary': (
                f"Optimized allocation provides {improvement['impressions_gain_percentage']:.1f}% more impressions "
                f"and {improvement['engagement_gain_percentage']:.1f}% more engagement"
            )
        }


def create_deployment_package(model_path, preprocessor_path, output_path='business_deployment_package.pkl'):
    """
    Create deployment package for business users.

    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
        output_path: Output pickle file

    Returns:
        dict with predictor and optimizer
    """
    predictor = BusinessBudgetPredictor(model_path, preprocessor_path)
    optimizer = BusinessBudgetOptimizer(predictor)

    package = {
        'predictor': predictor,
        'optimizer': optimizer,
        'version': '2.0.0',
        'features': ['Platform', 'campaign_type', 'content_type'],  # Quarter removed
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'note': 'Quarter feature removed'
    }

    joblib.dump(package, output_path)
    print(f"Business deployment package saved to: {output_path}")

    return package


if __name__ == "__main__":
    print("Business Budget Optimizer - Ready for Deployment")
    print("=" * 60)
    print("\nFeatures:")
    print("✅ Quarter removed")
    print("✅ Business-friendly percentage-based input")
    print("✅ Automatic optimization across Platform/campaign/content")
    print("✅ Simple comparison: User vs AI-optimized")
    print("\nUse create_deployment_package() to package for Flask.")
