# Social Media Campaign Performance Prediction

Machine learning system to predict campaign performance metrics (impressions, engagement) for social media campaigns across multiple platforms.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Web Interface](#web-interface)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Statistical Background](#statistical-background)

---

## Overview

This system solves a critical marketing challenge: **predicting social media campaign performance before launch**.

### Key Capabilities

- **Dual-Target Prediction**: Simultaneously predicts impressions AND engagement from a single model
- **Interactive Web Interface**: Simple prediction tool for marketing teams
- **REST API**: Programmatic access for integration with existing systems
- **Multi-Platform Support**: TikTok, Instagram, and Meta campaigns

### Business Value

- Predict campaign performance before launch
- Compare efficiency across platforms and content types
- Data-driven decision making with statistical confidence
- Estimate CPM and cost per engagement metrics

---

## Model Architecture

### Multi-Output Regression Approach

We use a **Random Forest Multi-Output Regressor** that predicts two targets simultaneously:
1. **Log(Impressions)** - Campaign reach
2. **Log(Engagement)** - Total user interactions (likes + shares + comments + saves)

### Why Multi-Output?

Traditional approaches would train separate models for impressions and engagement. Our multi-output approach provides:

1. **Coherent Predictions**: Engagement predictions are consistent with impression predictions
2. **Shared Learning**: The model learns patterns common to both metrics
3. **Simplified Deployment**: Single model, single preprocessing pipeline
4. **Computational Efficiency**: Train once, predict twice

### Input Features (4 variables)

```python
features = [
    'Platform',        # Meta, TikTok, Instagram
    'campaign_type',   # Bau, Mm, Flood The Feed
    'content_type',    # 6 types (Influencer, Paid, Owned variants)
    'Log_Spend_Total'  # log(total_spend + 1)
]
```

### Feature Engineering

**Log Transformations**: Applied to all monetary and count variables
- `Log_Spend_Total = log(total_spend + 1)`
- `Log_Impressions = log(impressions + 1)`
- `Log_Engagement = log(engagement + 1)`

**Why logarithmic transformation?**
- Handles wide range of values (spend: $100 to $50,000+)
- Stabilizes variance across different budget levels
- Captures diminishing returns (doubling spend doesn't double impressions)
- Normal distribution assumption for regression

**Categorical Encoding**: One-hot encoding for 3 categorical features (Platform, campaign_type, content_type)
- Results in 10 total features after encoding (after dropping first level of each categorical)

---

## Model Performance

### Random Forest MultiOutput (Best Model)

**Model Configuration**:
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
```

**Dataset Information**:
- **Total Samples**: 1,000 campaigns
- **Training Set**: 800 campaigns (80%)
- **Testing Set**: 200 campaigns (20%)
- **Input Features**: 4 (Platform, Campaign Type, Content Type, Spend)
- **Target Variables**: 2 (Impressions, Engagement)

**Performance Metrics** (Test Set, N=200):

| Metric | Impressions | Engagement |
|--------|-------------|------------|
| **R² Score** | 0.165 | **0.283** |
| **MAE** | 239,241 | 17,339 |
| **RMSE** | 381,646 | 32,251 |
| **MAPE** | 54.81% | 60.33% |

**Average R² Score**: 0.224 (across both targets)

### Model Comparison

We evaluated three multi-output approaches:

| Model | Avg R² | Impressions R² | Engagement R² | Impressions MAPE | Engagement MAPE |
|-------|--------|----------------|---------------|------------------|-----------------|
| **Random Forest** | **0.224** | **0.165** | **0.283** | **54.81%** | **60.33%** |
| LightGBM | 0.198 | 0.156 | 0.239 | 56.24% | 62.15% |
| XGBoost | 0.194 | 0.143 | 0.245 | 57.12% | 61.89% |

**Winner**: Random Forest provides the best overall performance with lowest overfitting and best engagement rate prediction.

### Interpretation of R² Scores

- **Impressions R² = 0.165**: Model explains 16.5% of variance in impressions
- **Engagement R² = 0.283**: Model explains 28.3% of variance in engagement

While these may seem moderate, they are **reasonable for social media prediction**:
- High inherent randomness in social media performance
- External factors not captured (creative quality, timing, competition, virality)
- Model captures systematic patterns in spend allocation and platform effects
- **Engagement is more predictable** than impressions (higher R²)

---

## Quick Start

### 1. Start the Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask server
python3 app.py

# Open browser
http://localhost:5000
```

### 2. Make a Prediction (Python)

```python
from predict_api import predict_campaign_metrics

result = predict_campaign_metrics(
    total_spend=10000,
    platform='TikTok',
    campaign_type='Flood The Feed',
    content_type='Influencer - Cfg - Boosted Only'
)

print(f"Impressions: {result['predictions']['impressions']:,}")
print(f"Engagement: {result['predictions']['engagement']:,}")
print(f"Engagement Rate: {result['predictions']['engagement_rate_pct']}")
```

### 3. Use the REST API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_spend": 10000,
    "platform": "TikTok",
    "campaign_type": "Flood The Feed",
    "content_type": "Influencer - Cfg - Boosted Only"
  }'
```

---

## API Endpoints

### `POST /api/predict`

Get predictions for a single campaign.

**Request**:
```json
{
  "total_spend": 10000,
  "platform": "TikTok",
  "campaign_type": "Flood The Feed",
  "content_type": "Influencer - Cfg - Boosted Only"
}
```

**Response**:
```json
{
  "status": "success",
  "input": {
    "total_spend": 10000.0,
    "platform": "TikTok",
    "campaign_type": "Flood The Feed",
    "content_type": "Influencer - Cfg - Boosted Only"
  },
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
  "confidence_intervals": {
    "impressions": {
      "lower": 64833,
      "upper": 771962,
      "range": "64,833 - 771,962"
    },
    "engagement": {
      "lower": 0,
      "upper": 104886,
      "range": "0 - 104,886"
    },
    "confidence_level": "95%",
    "description": "95% confidence interval based on tree variance in Random Forest"
  }
}
```

### `GET /api/options`

Get valid values for dropdown fields (platforms, campaign types, content types).

### `GET /health`

Health check endpoint.

---

## Web Interface

### Features

1. **Multi-Campaign Planning**: Add and manage multiple campaigns simultaneously
2. **Budget Allocation**: Set overall budget and allocate percentages to each campaign
3. **Batch Predictions**: Predict all campaigns with a single click
4. **Real-Time Calculations**: Automatic budget allocation updates
5. **Aggregate Metrics**: View total impressions, engagement, CPM, and CPE across all campaigns
6. **Google Cloud Design**: Professional UI with Google's design language
7. **Model Statistics**: View detailed dataset and performance metrics

### User Workflow

1. Set overall budget amount
2. Add campaign(s) using the "+ Add Campaign" button
3. For each campaign, select:
   - Platform (TikTok, Instagram, or Meta)
   - Campaign type (Flood The Feed, Bau, or Mm)
   - Content type (6 options)
   - Budget allocation percentage
4. Click "🔮 Predict All Campaigns" to get ML predictions
5. View individual and aggregate performance metrics in the table
6. Check total row for portfolio-wide metrics

---

## Project Structure

```
MM/
├── Core Application Files
│   ├── app.py                          # Flask web server with REST API
│   ├── predict_api.py                  # Prediction function (core logic)
│
├── Model Training & Data
│   ├── generate_dummy_data.py          # Generate synthetic training data
│   ├── multi_output_training.py        # Train models & select best performer
│   └── multi_output_training_notebook.py  # Jupyter-friendly training script
│
├── Analysis & Exploration
│   └── model_analysis.ipynb            # Data exploration & visualization notebook
│
├── Web Interface
│   └── templates/
│       └── index.html                  # Interactive web UI
│
├── Trained Models (artifacts)
│   ├── models/
│   │   ├── best_multi_output_model_random_forest_multioutput.pkl  # CURRENT MODEL (R²=0.224)
│   │   ├── multi_output_preprocessor.pkl                          # Feature pipeline
│   │   └── multi_output_model_metadata.json                       # Performance metrics
│
├── Data (generated)
│   ├── data/
│   │   ├── campaign_data.csv                        # Training data (1000 campaigns)
│   │   ├── campaign_data_multi_output_engineered.csv  # With log features
│   │   ├── X_train_multi.npy, X_test_multi.npy      # Preprocessed features
│   │   └── y_train_multi.npy, y_test_multi.npy      # Target variables
│
├── Results (outputs)
│   ├── results/
│   │   ├── multi_output_model_comparison.csv        # Model performance comparison
│   │   ├── multi_output_model_comparison.png        # Model comparison chart
│   │   ├── multi_output_target_distributions.png    # Data distribution visualizations
│   │   └── demo_predictions.csv                     # Sample predictions
│
├── Documentation
│   ├── README.md                       # This file (comprehensive guide)
│   └── claude.md                       # Technical context, deployment guide & development history
│
└── Configuration
    ├── requirements.txt                # Python dependencies
    └── .claude/settings.local.json     # Claude Code settings
```

### File Status & Purpose

#### ✅ **ACTIVE & CURRENT FILES** (Use these)

| File | Purpose | Status |
|------|---------|--------|
| `app.py` | Flask web server with REST API | ✅ Up-to-date, prediction-only |
| `predict_api.py` | Core prediction function | ✅ Up-to-date, Random Forest |
| `generate_dummy_data.py` | Data generation | ✅ Up-to-date |
| `multi_output_training.py` | Model training pipeline | ✅ Up-to-date, trains 3 models |
| `multi_output_training_notebook.py` | Jupyter training script | ✅ Up-to-date, 15 cells |
| `model_analysis.ipynb` | Exploratory data analysis | ✅ Up-to-date, Random Forest |
| `models/best_multi_output_model_random_forest_multioutput.pkl` | Current best model | ✅ Active model (R²=0.224) |
| `models/multi_output_preprocessor.pkl` | Feature preprocessor | ✅ Active preprocessor |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib
- flask
- xgboost, lightgbm

### First-Time Setup

```bash
# 1. Generate training data
python3 generate_dummy_data.py

# 2. Train models (trains Random Forest, XGBoost, LightGBM and selects best)
python3 multi_output_training.py

# 3. Start Flask server
python3 app.py

# 4. Open browser to http://localhost:5000
```

---

## Statistical Background

### Model Training Process

1. **Data Generation**: 1000 synthetic campaigns with realistic patterns
2. **Train/Test Split**: 80/20 split (800 train, 200 test)
3. **Feature Engineering**: Log transforms, one-hot encoding
4. **Model Training**: Random Forest with 100 estimators, max depth 10
5. **Validation**: Multiple metrics across both targets
6. **Model Selection**: Automatically selects best performer based on average R²

### Assumptions & Limitations

**Assumptions**:
1. Historical patterns continue (no major platform algorithm changes)
2. Creative quality is average (not exceptional or poor)
3. No external events (viral moments, crises)
4. Spend-performance relationship is log-linear

**Limitations**:
1. Synthetic data (replace with real campaign data for production)
2. No time-series effects (day-of-week, long-term seasonality)
3. No competitive effects (other advertisers' spending)
4. No creative quality metrics (images, videos, copy)

---

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## Recent Enhancements (October 2025)

### UI/UX Improvements
1. ✅ **Google Cloud Platform Design**: Complete UI redesign matching GCP design language
2. ✅ **Multi-Campaign Interface**: Manage and predict multiple campaigns simultaneously
3. ✅ **Batch Predictions**: Single-click prediction for all campaigns
4. ✅ **Real-Time Budget Allocation**: Auto-calculated dollar amounts from percentages
5. ✅ **Aggregate Totals**: Portfolio-wide metrics with weighted averages
6. ✅ **Detailed Model Stats**: Train/test split, sample counts, MAPE metrics

### API Enhancements
1. ✅ **Confidence Intervals**: 95% prediction intervals using Random Forest tree variance
2. ✅ **Enhanced Metadata**: Dataset information including train/test split details
3. ✅ **Comprehensive Response**: CPM, CPE, engagement rate, and confidence intervals

### Statistical Improvements
1. ✅ **MAPE Metrics**: Added Impressions MAPE (54.81%) and Engagement MAPE (60.33%)
2. ✅ **Prediction Uncertainty**: Tree-based confidence intervals for both targets
3. ✅ **Dataset Transparency**: Full visibility into training/testing sample counts

## Future Enhancements

1. **A/B Testing**: Framework for validating predictions against actual campaigns
2. **Real Data Integration**: Replace synthetic data with actual campaign results
3. **Temporal Features**: Day-of-week, seasonality, campaign duration effects
4. **Creative Quality**: Image/video analysis, copy sentiment scoring
5. **Budget Optimizer**: AI-powered budget allocation recommendations

---

## License

MIT License - Free for commercial and educational use.

---

## Support

For questions or issues:
1. Check this documentation (README.md)
2. Review `claude.md` for technical context, deployment guide, and development history

---

**Last Updated**: 2025-10-06
**Version**: 5.0 (Multi-Campaign Planner with Confidence Intervals)
**Current Model**: Random Forest MultiOutput (R² = 0.224)
**UI Design**: Google Cloud Platform Design Language
**Key Features**: Batch predictions, confidence intervals, aggregate metrics, train/test transparency
