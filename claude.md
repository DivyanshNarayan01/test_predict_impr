# Claude Context - Social Media Campaign Prediction (Multi-Output)

This file provides context and guidance for Claude when working on this project.

---

## Project Overview

This is a **Social Media Campaign Performance Prediction** machine learning system that predicts **both Impressions AND Engagement** for marketing campaigns across social media platforms using **Random Forest multi-output regression**.

### Key Objectives
- **Multi-Output Prediction**: Simultaneously predict Impressions and Engagement from a single model
- Compare multiple machine learning approaches (Random Forest, XGBoost, LightGBM)
- Provide business insights for campaign optimization on both reach and engagement
- Enable data-driven marketing decisions with coherent dual-metric forecasts
- **Budget Optimization**: AI-powered allocation algorithm for maximum ROI

---

## Current Project Status (October 6, 2025)

**Implementation**: ✅ **PRODUCTION READY**
**Best Model**: Random Forest MultiOutput (Avg R² 0.224)
**Features**: 4 input variables (Quarter **REMOVED**)
**UI Version**: 5.0 - Google Cloud Platform Design with Multi-Campaign Planner
**Capabilities**:
- Impressions R² 0.165 (MAPE 54.81%), Engagement R² 0.283 (MAPE 60.33%)
- **Flask Web Application**: Multi-campaign portfolio planner with GCP design
- **REST API**: Predictions with 95% confidence intervals
- **Batch Predictions**: Predict all campaigns with single click
- **Real-Time Calculations**: Auto-updated budget allocations and totals
- **Model Transparency**: Full train/test split details and statistical metrics

---

## Project Structure

```
MM/
├── Core Application Files (✅ PRODUCTION READY)
│   ├── app.py                          # Flask web server with REST API
│   ├── predict_api.py                  # Core prediction function (Random Forest)
│   ├── budget_optimizer.py             # Advanced budget optimization engine
│   └── business_budget_optimizer.py    # Simplified optimizer for business users
│
├── Model Training & Data Generation
│   ├── generate_dummy_data.py          # Generate synthetic training data (no Quarter)
│   ├── multi_output_training.py        # Train 3 models & select best performer
│   └── multi_output_training_notebook.py  # Jupyter-friendly training script (15 cells)
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
│   │   ├── best_multi_output_model_random_forest_multioutput.pkl  # ✅ CURRENT MODEL (R²=0.224)
│   │   ├── multi_output_preprocessor.pkl                          # Feature pipeline
│   │   └── multi_output_model_metadata.json                       # Performance metrics
│
├── Data (generated)
│   ├── data/
│   │   ├── campaign_data.csv                        # Training data (1000 campaigns, 6 columns)
│   │   ├── campaign_data_multi_output_engineered.csv  # With log features
│   │   ├── X_train_multi.npy, X_test_multi.npy      # Preprocessed features
│   │   └── y_train_multi.npy, y_test_multi.npy      # Target variables (2D)
│
├── Results (outputs)
│   ├── results/
│   │   ├── multi_output_model_comparison.csv        # Model performance comparison
│   │   ├── multi_output_model_comparison.png        # 6-panel comparison chart
│   │   ├── multi_output_target_distributions.png    # Data distribution visualizations
│   │   ├── demo_predictions.csv                     # Sample predictions
│   │   ├── business_optimization_results.json       # Optimization results
│   │   └── optimized_budget_allocation.csv          # Optimized allocations
│
├── Documentation
│   ├── README.md                       # Professional documentation (comprehensive)
│   ├── claude.md                       # This file (AI assistant context)
│   ├── MULTI_OUTPUT_SUMMARY.md         # Multi-output modeling explanation
│   └── DEPLOYMENT.md                   # Deployment instructions
│
└── Configuration
    ├── requirements.txt                # Python dependencies
    └── .claude/settings.local.json     # Claude Code settings
```

### ⚠️ **OUTDATED FILES** (Should be deleted)

| File | Issue | Replaced By |
|------|-------|-------------|
| `predict_multi_output.py` | References Quarter, old LightGBM model | `predict_api.py` |
| `demo_budget_optimization.py` | References Quarter, old model | Flask API in `app.py` |
| `demo_business_optimizer.py` | References old LightGBM model | Flask API in `app.py` |
| `models/best_multi_output_model_lightgbm_multioutput.pkl` | Old model (R²=0.198 vs 0.224) | Random Forest model |

---

## Data Schema (Current Version - October 6, 2025)

### Input Features (4 variables)

```python
features = [
    'Platform',        # Meta, TikTok, Instagram
    'campaign_type',   # Bau, Mm, Flood The Feed
    'content_type',    # 6 types (Influencer, Paid, Owned variants)
    'Log_Spend_Total'  # log(total_spend + 1)
]
```

**❌ REMOVED**: Quarter feature (minimal predictive value, removed Oct 6, 2025)

### Content Types (6 valid values)
1. Influencer - Cfg - Boosted Only
2. Influencer - Ogilvy - Organic Only
3. Owned - Boosted Only
4. Owned - Organic Only
5. Paid - Brand
6. Paid - Partnership

### Target Variables (2) - Multi-Output
- **Impressions**: Campaign reach metric (30K - 5.4M range)
- **Engagement**: Aggregate engagement (Likes + Shares + Comments + Saves)

### Engineered Features
- **Log_Spend_Total**: Log-transformed total spend
- **Log_Impressions**: Log-transformed impressions (target)
- **Log_Engagement**: Log-transformed engagement (target)
- **One-Hot Encoding**: 3 categorical features → 10 total features after encoding

---

## Model Architecture

### Current Best Model: Random Forest MultiOutput (R² = 0.224)

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
```

**Performance Metrics** (Test Set, N=200):

| Metric | Impressions | Engagement |
|--------|-------------|------------|
| **R² Score** | 0.165 | **0.283** |
| **MAE** | 239,241 | 17,339 |
| **RMSE** | 381,646 | 32,251 |
| **MAPE** | 54.81% | 60.33% |
| **Overfitting** | 21.8% | 15.0% |

**Engagement Rate MAPE**: 27.16%

### Model Comparison (After Quarter Removal)

| Model | Avg R² | Impressions R² | Engagement R² | Eng. Rate MAPE |
|-------|--------|----------------|---------------|----------------|
| **Random Forest** | **0.224** | **0.165** | **0.283** | **27.16%** |
| LightGBM | 0.198 | 0.156 | 0.239 | 28.53% |
| XGBoost | 0.194 | 0.143 | 0.245 | 27.63% |

**Winner**: Random Forest provides best overall performance with lowest overfitting.

---

## Common Tasks

### Setup and Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Generate data (no Quarter feature)
python3 generate_dummy_data.py
```

### Multi-Output Workflow
```bash
# 1. Generate data with Engagement column
python3 generate_dummy_data.py

# 2. Train all 3 multi-output models and compare (selects Random Forest)
python3 multi_output_training.py

# 3. Start Flask web app
python3 app.py

# 4. Open browser to http://localhost:5000
```

### Jupyter Notebook Training
```bash
# Use Jupyter-friendly training script
# Run cells in order (15 cells total)
jupyter notebook multi_output_training_notebook.py
```

---

## Making Predictions

### Python API (Current Implementation)

```python
from predict_api import predict_campaign_metrics

result = predict_campaign_metrics(
    total_spend=10000.0,
    platform='TikTok',
    campaign_type='Flood The Feed',
    content_type='Influencer - Cfg - Boosted Only'
)

print(f"Impressions: {result['predictions']['impressions']:,}")
print(f"Engagement: {result['predictions']['engagement']:,}")
print(f"Engagement Rate: {result['predictions']['engagement_rate_pct']}")
```

**Output:**
```
Impressions: 803,944
Engagement: 80,080
Engagement Rate: 9.96%
```

### Direct Model Usage

```python
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('models/best_multi_output_model_random_forest_multioutput.pkl')
preprocessor = joblib.load('models/multi_output_preprocessor.pkl')

# Create campaign
campaign = pd.DataFrame([{
    'Platform': 'TikTok',
    'campaign_type': 'Flood The Feed',
    'content_type': 'Influencer - Cfg - Boosted Only',
    'total_spend': 10000.0
}])

# Engineer features
campaign['Log_Spend_Total'] = np.log(campaign['total_spend'] + 1)

# Select features (order matters!)
X = campaign[['Platform', 'campaign_type', 'content_type', 'Log_Spend_Total']]

# Predict both targets
X_processed = preprocessor.transform(X)
predictions_log = model.predict(X_processed)[0]

# Transform back to original scale
impressions = np.expm1(predictions_log[0])
engagement = np.expm1(predictions_log[1])
engagement_rate = engagement / impressions

print(f"Predicted Impressions: {impressions:,.0f}")
print(f"Predicted Engagement: {engagement:,.0f}")
print(f"Engagement Rate: {engagement_rate:.2%}")
```

---

## Flask Web Application

### Quick Start
```bash
# Start the Flask server
python3 app.py

# Open browser
http://localhost:5000
```

### REST API Endpoints

#### `POST /api/predict`
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

**Response** (includes 95% confidence intervals):
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

#### `POST /api/optimize`
Optimize budget allocation across multiple campaigns.

**Request**:
```json
{
  "total_budget": 100000,
  "campaigns": [
    {
      "platform": "TikTok",
      "campaign_type": "Flood The Feed",
      "content_type": "Influencer - Cfg - Boosted Only",
      "budget_percentage": 50
    },
    {
      "platform": "Instagram",
      "campaign_type": "Bau",
      "content_type": "Paid - Brand",
      "budget_percentage": 50
    }
  ]
}
```

#### `GET /api/options`
Get valid values for dropdown fields (platforms, campaign types, content types).

#### `GET /health`
Health check endpoint.

---

## Budget Optimization Algorithm

### Algorithm Overview

The optimizer uses an **efficiency-based greedy allocation** approach with statistical justification.

### Step-by-Step Process

**STEP 1: Calculate Efficiency at User Budget**
```python
# For each campaign at user-allocated budget
impressions_i = model.predict(campaign_i)[0]
engagement_i = model.predict(campaign_i)[1]

impressions_per_dollar = impressions_i / user_budget_i
engagement_per_dollar = engagement_i / user_budget_i

# Combined efficiency (equal weights)
efficiency_i = 0.5 * engagement_per_dollar + 0.5 * (impressions_per_dollar / 100)
```

**STEP 2: Sort by Efficiency**
- Rank all campaigns by efficiency score (highest ROI first)

**STEP 3: Allocate Budget**
- Redistribute budget to high-efficiency campaigns
- Respect minimum budget constraints
- Use efficiency² weighting to amplify differences

### Statistical Justification

**Diminishing Returns**: The log-linear model captures diminishing returns:
```
If impressions ∝ spend^0.85, then:
- Doubling spend → 1.80× impressions (not 2×)
- Tripling spend → 2.54× impressions (not 3×)
```

**Efficiency Calculation**:
- Equal weights (0.5 each) balance reach and engagement goals
- Division by 100 normalizes impressions to match engagement scale
- Measured at actual user budget to capture real marginal returns

**Example Results**:
- User allocation: 1.7M impressions, 126K engagement
- Optimized allocation: 1.9M impressions, 159K engagement
- **Lift**: +11.5% impressions, +26.6% engagement

---

## Key Business Insights

### Platform Performance (Engagement Rate)
1. **TikTok**: Highest engagement rate
2. **Instagram**: Medium engagement rate
3. **Meta**: Lower engagement rate

### Content Performance
1. **Influencer - Cfg - Boosted Only**: Highest engagement
2. **Influencer - Ogilvy - Organic Only**: High engagement
3. **Paid - Brand**: Lowest engagement

**Key Insight**: Influencer content generates significantly better engagement than paid brand ads.

---

## Development History

### October 6, 2025 (Latest) - UI Enhancement & Statistical Transparency

**UI Redesign - Google Cloud Platform Design:**
1. ✅ Complete visual redesign to match Google Cloud Platform aesthetic
2. ✅ Google-branded top bar with multi-colored logo
3. ✅ Multi-campaign interface (add/delete campaigns dynamically)
4. ✅ Batch prediction (single "Predict All Campaigns" button)
5. ✅ Real-time budget allocation calculator (percentage → dollar amount)
6. ✅ Aggregate totals row with weighted CPM/CPE averages
7. ✅ Center-aligned table content for better readability
8. ✅ Light yellow (#fef7e0) input styling across all fields

**Statistical Enhancements:**
1. ✅ Added train/test split details to model metadata (800/200 split)
2. ✅ UI displays: Total samples, training samples, testing samples
3. ✅ Replaced MAE metrics with MAPE for better interpretability
4. ✅ Confidence intervals (95% CI) in API responses using tree variance
5. ✅ Model performance section shows both R² and MAPE metrics

**Files Modified:**
- `templates/index.html` - Complete UI overhaul with GCP design
- `models/multi_output_model_metadata.json` - Added dataset information
- `predict_api.py` - Added confidence interval calculations
- `README.md` - Updated with new features and API response format
- `claude.md` - This file, updated with latest changes

### October 6, 2025 - Quarter Variable Removal

**Major Changes:**
1. ✅ Removed Quarter variable from entire project (7 files updated)
2. ✅ Regenerated training data (1000 campaigns, now 6 columns instead of 7)
3. ✅ Retrained all 3 models
4. ✅ Random Forest became new best model (R² = 0.224 vs LightGBM R² = 0.198)
5. ✅ Updated all documentation (README.md, claude.md)
6. ✅ Updated model_analysis.ipynb to use Random Forest and remove Quarter

**Files Updated:**
- generate_dummy_data.py
- multi_output_training.py
- multi_output_training_notebook.py
- predict_api.py
- app.py
- budget_optimizer.py
- business_budget_optimizer.py
- model_analysis.ipynb

**Files Identified as Outdated:**
- predict_multi_output.py (still references Quarter)
- demo_budget_optimization.py (still references Quarter)
- demo_business_optimizer.py (still references old model)
- models/best_multi_output_model_lightgbm_multioutput.pkl (old model)

### October 2, 2025 - Multi-Output Implementation

**Major Additions:**
1. ✅ Added **Engagement** target variable to data generation
2. ✅ Implemented 3 multi-output models (RF, XGBoost, LightGBM)
3. ✅ Created `multi_output_training.py` - complete pipeline
4. ✅ Created Flask web application with optimization UI
5. ✅ Generated comprehensive comparison across 8+ metrics

### September 2024 - Initial Implementation

- Schema simplification (reduced to 5 input features)
- Changed target from Views to Impressions
- Reduced platforms from 6 to 3 (Meta, TikTok, Instagram)
- Updated to 6 content_type categories

---

## Troubleshooting

### Common Issues

1. **Missing XGBoost/LightGBM**
   ```bash
   pip install xgboost lightgbm
   ```

2. **Model file not found**
   - Train models first: `python3 multi_output_training.py`
   - Check `models/` directory for `best_multi_output_model_random_forest_multioutput.pkl`

3. **Shape mismatch errors**
   - Ensure feature order: `['Platform', 'campaign_type', 'content_type', 'Log_Spend_Total']`
   - Do NOT include Quarter
   - Multi-output targets must be 2D array: shape (n_samples, 2)

4. **Quarter-related errors**
   - Old files still reference Quarter
   - Use updated files: `predict_api.py`, `app.py`, etc.
   - Delete outdated demo files

### File Dependencies

**Current Pipeline:**
```
generate_dummy_data.py
    ↓ creates campaign_data.csv (6 columns, no Quarter)
multi_output_training.py
    ↓ creates models/best_multi_output_model_random_forest_multioutput.pkl
    ↓ creates models/multi_output_preprocessor.pkl
    ↓ creates models/multi_output_model_metadata.json
app.py
    ↓ uses saved models for predictions and optimization
```

---

## Testing Commands

```bash
# Generate fresh data (no Quarter)
python3 generate_dummy_data.py

# Train multi-output models (takes ~5 minutes, selects Random Forest)
python3 multi_output_training.py

# Start Flask app
python3 app.py

# Test API prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_spend": 10000,
    "platform": "TikTok",
    "campaign_type": "Flood The Feed",
    "content_type": "Influencer - Cfg - Boosted Only"
  }'

# Check model metadata
cat models/multi_output_model_metadata.json

# View comparison results
cat results/multi_output_model_comparison.csv
```

---

## Quick Reference

### Data Schema Summary
**Inputs**: Platform, campaign_type, content_type, Log_Spend_Total (4 features)
**Outputs**: Impressions, Engagement (2 targets)
**Engagement Rate**: Mean ~6.5%, Range 0.5-15%

### Best Model Performance
**Random Forest MultiOutput**: Avg R² 0.224
- Impressions: R² 0.165, MAE 239K, MAPE 54.8%
- Engagement: R² 0.283, MAE 17.3K, MAPE 60.3%
- Engagement Rate: MAPE 27.2%

### File Paths (Key Artifacts)
- Model: `models/best_multi_output_model_random_forest_multioutput.pkl`
- Preprocessor: `models/multi_output_preprocessor.pkl`
- Metadata: `models/multi_output_model_metadata.json`
- Data: `data/campaign_data.csv` (6 columns, no Quarter)

### Command Shortcuts
```bash
# Full pipeline
python3 generate_dummy_data.py && python3 multi_output_training.py

# Start web app
python3 app.py

# Check performance
cat models/multi_output_model_metadata.json
```

---

---

## Deployment Guide

### Production Deployment Options

#### Option 1: Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Option 2: Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY app.py predict_api.py requirements.txt ./
COPY templates/ ./templates/
COPY models/ ./models/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t campaign-optimizer .
docker run -p 5000:5000 campaign-optimizer
```

#### Option 3: Cloud Platforms
- **Heroku**: `echo "web: gunicorn app:app" > Procfile`
- **Google Cloud Run**: `gcloud run deploy --image gcr.io/PROJECT_ID/campaign-optimizer`
- **AWS Elastic Beanstalk**: Zip and upload to console

### Performance Benchmarks
- Single prediction: ~10ms
- Batch prediction (10 campaigns): ~50ms
- Model size: ~15MB
- Memory per prediction: ~1KB

---

## Multi-Output Implementation Technical Details

### Why Multi-Output?
Traditional approaches train separate models for impressions and engagement. Multi-output provides:
1. **Coherent Predictions**: Engagement consistent with impressions
2. **Shared Learning**: Model learns patterns common to both metrics
3. **Simplified Deployment**: Single model, single pipeline
4. **Computational Efficiency**: Train once, predict twice

### Implementation Approach
Uses `sklearn.multioutput.MultiOutputRegressor`:
```python
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(RandomForestRegressor(...))
# y_train shape: (800, 2) for [Log_Impressions, Log_Engagement]
model.fit(X_train, y_train)
```

### Engagement Modeling Details
- **Base Rate**: 4% of impressions
- **Range**: 0.5% - 15% (clipped for realism)
- **Platform Effects**: TikTok 1.5x, Instagram 1.2x, Meta 0.9x
- **Content Effects**: Influencer 1.4-1.6x, Paid ads 0.8x
- **Mean Engagement Rate**: 6.58%

### Confidence Intervals Calculation
Uses Random Forest tree variance for 95% prediction intervals:
```python
# Get predictions from all 100 individual trees
tree_predictions = [tree.predict(X) for tree in model.estimators_[0].estimators_]
std = np.std(tree_predictions)
lower = prediction - 1.96 * std
upper = prediction + 1.96 * std
```

---

## Reference Documentation

### Primary Docs (Production Ready - Oct 6, 2025)
- **`README.md`** - Professional documentation with model explanations
- **`claude.md`** - This file - complete technical context
- **`model_analysis.ipynb`** - Interactive notebook with data visualizations
- **`results/multi_output_model_comparison.csv`** - Detailed metrics table

---

**Current Status**: ✅ **PRODUCTION READY (v5.0 - Enhanced UI & Statistics)**

**Best Model**: Random Forest MultiOutput (Avg R² 0.224)

**Features**: 4 input variables (Platform, campaign_type, content_type, total_spend)

**UI Design**: Google Cloud Platform Design Language v5.0

**Capabilities**:
- Multi-campaign portfolio planner with batch predictions
- Real-time budget allocation calculator (% → $)
- 95% confidence intervals for all predictions
- Aggregate totals with weighted average metrics
- Detailed train/test statistics (800/200 split)
- MAPE metrics: Impressions 54.81%, Engagement 60.33%

**Business Value**:
- Professional GCP-style interface for enterprise users
- Portfolio-wide campaign planning and analysis
- Statistical transparency with confidence intervals
- Zero-code solution with comprehensive metrics
- Single-click batch prediction for efficiency

**Last Updated**: October 6, 2025 (v5.0)
