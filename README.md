# Student Stress & Academic Performance Predictor

**Group G-10 | BE CSE AI&ML 4th Sem | Batch 2024**

A machine learning system that predicts academic performance scores from student stress and lifestyle factors using regression models.

## Project Overview

This project builds a comprehensive ML pipeline:
- **Data pipeline:** Clean, impute, and engineer features from messy survey data
- **Regression models:** Compare 6 models (AdaBoost, Gradient Boosting, Ridge, Linear Regression, Lasso, ElasticNet)
- **REST API:** FastAPI backend serving predictions
- **Web dashboard:** Live predictor UI with model comparison charts

### Team
- **Hritabrata Das** (Lead)
- **Harshit**
- **Abhinav Dadwal**

## Project Structure

```
student-stress-predictor/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── StressLevelDataset_dirty_og.csv
│   │   └── StressLevelDataset_dirty.csv
│   └── processed/                    # Cleaned & imputed datasets
│       ├── StressLevelDataset_cleaned.csv
│       └── after_eda.csv
├── notebooks/
│   ├── main.ipynb           # Full pipeline: cleaning, EDA, modelling, export
├── models/                           # Saved trained models (.joblib files)
│   └── ... (model files for each)
├── api/                              # FastAPI backend
│   ├── main.py                       # API endpoints & CORS setup
│   ├── predictor.py                  # ML prediction logic & composite scores
│   ├── schemas.py                    # Pydantic input/output schemas
│   └── __pycache__/
├── ui/                         # Dashboard UI (vanilla JS)
│   ├── index.html                    # Main dashboard
│   ├── app.js                        # Tabs, forms, charts, API calls
│   └── style.css                     # Basic styling
├── requirements.txt                  # Python dependencies
├── README.md                         # (this file)
└── .gitignore
```

## Quick Start

### 1. Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API Server

```bash
cd api
python -m uvicorn main:app --reload --port 8001
```

The API will be available at `http://localhost:8001`.

**API Endpoints:**
- `GET /health` — Health check + list available models
- `POST /predict?model_name=AdaBoost` — Single model prediction
- `POST /predict-all` — All models comparison
- `GET /models` — List all models with metrics
- `GET /metrics` — Get raw metrics

### 4. Access the Dashboard

Open `frontend/index.html` in your browser, or serve via HTTP:

```bash
# Using Python's built-in server (run from project root)
python -m http.server 8080 --directory frontend
```

Navigate to `http://localhost:8080`.

**Dashboard features:**
- 📊 Model Comparison tab with R² and RMSE charts
- 🎯 Live Predictor tab for instant predictions
- 📈 Real-time model comparison

### 5. Explore the Notebook

To see the full data processing and training pipeline:

```bash
jupyter notebook
```

Open `notebooks/EDA6_improved.ipynb`. The notebook covers:
- Data loading & cleaning
- Outlier detection & treatment
- Missing value imputation (KNN)
- Feature engineering (composite scores)
- Model comparison (6 regression models)
- Model export for API

## Key Milestones

| Phase | Status | Work |
|-------|--------|------|
| **Data** | ✅ Complete | Loaded, cleaned, imputed |
| **EDA** | ✅ Complete | Distributions, correlations, outliers |
| **Modelling** | ✅ Complete | 6 regression models trained & tuned |
| **API** | ✅ Complete | FastAPI backend with all endpoints |
| **Dashboard** | ✅ Complete | Live predictor + model comparison |
| **Export** | ✅ Complete | All models saved as .joblib files |

## Dataset & Features

**Source:** StressLevelDataset (survey-based, 11,000+ rows)

**Feature categories:**
- **Psychological factors:** anxiety_level, depression, self_esteem, sleep_quality
- **Environmental factors:** safety, basic_needs, living_conditions, noise_level
- **Social factors:** teacher_student_relationship, social_support, peer_pressure, bullying
- **Academic factors:** study_load, headache, future_career_concerns, blood_pressure
- **Lifestyle:** breathing_problem, extracurricular_activities, stress_level, mental_health_history

**Target:** `academic_performance` (continuous, 0–5 scale)

**Processing pipeline:**
1. **Cleaning:** Standardize categorical columns (yes/no, ordinal encoding)
2. **Outlier treatment:** Winsorize at 1st–99th percentile (anxiety, depression, self-esteem)
3. **Imputation:** KNN (k=5) on scaled features
4. **Feature engineering:** Create 4 composite scores
5. **Redundancy removal:** Drop correlated features (noise_level, bullying)

## Models

All models predict `academic_performance` (continuous regression).

| Model | CV R² | Test R² | RMSE | Notes |
|-------|-------|---------|------|-------|
| **AdaBoost** ⭐ | 0.6113 | 0.6921 | 0.8019 | **RECOMMENDED** — Ensemble of weak learners |
| GradientBoosting | 0.5983 | 0.6437 | 0.8609 | Boosted trees, slower to train |
| Ridge | 0.5896 | 0.6348 | 0.8719 | Linear with L2 regularization |
| LinearRegression | 0.5773 | 0.6294 | 0.8785 | Baseline — simple & interpretable |
| Lasso | 0.5840 | 0.6260 | 0.8827 | Linear with L1 (feature selection) |
| ElasticNet | 0.5918 | 0.6357 | 0.8708 | Blends L1 + L2 regularization |

**Hyperparameter tuning:** RandomizedSearchCV (40 iterations, 5-fold CV)

**Feature scaling:** StandardScaler (applied for linear models only)

## Composite Features

Feature engineering groups related variables into interpretable composite scores:

| Score | Components | Scale | Meaning |
|-------|------------|-------|---------|
| `env_score` | safety + basic_needs + living_conditions | 0–15 | Environmental quality & safety |
| `mental_score` | anxiety + depression − self_esteem | −5 to 10 | Psychological burden |
| `pressure_score` | peer_pressure + study_load + future_career | 0–15 | Academic & social pressure |
| `support_score` | teacher_relationship + social_support | 0–10 | External support network |

These reduce multicollinearity and improve interpretability without sacrificing performance.

## API Usage Examples

### Single prediction (AdaBoost)

```bash
curl -X POST "http://localhost:8001/predict?model_name=AdaBoost" \
  -H "Content-Type: application/json" \
  -d '{
    "anxiety_level": 7,
    "depression": 5,
    "self_esteem": 8,
    "sleep_quality": 1,
    "stress_level": 1,
    "safety": 4,
    "basic_needs": 4,
    "living_conditions": 3,
    "teacher_student_relationship": 7,
    "social_support": 6,
    "peer_pressure": 5,
    "study_load": 2,
    "future_career_concerns": 6,
    "headache": 4,
    "blood_pressure": 3,
    "breathing_problem": 2,
    "mental_health_history": 0,
    "extracurricular_activities": 2,
    "noise_level": 3,
    "bullying": 1
  }'
```

**Response:**
```json
{
  "predicted_score": 3.45,
  "performance_band": "Average",
  "model_name": "AdaBoost",
  "model_r2": 0.6958,
  "model_rmse": 0.8019
}
```

### Compare all models

```bash
curl -X POST "http://localhost:8001/predict-all" \
  -H "Content-Type: application/json" \
  -d { ... same input ... }
```

Returns predictions from all 6 models for comparison.

## Model Export Format

Each model is saved as three files:

```
models/
├── AdaBoost.joblib              # Trained model (with StandardScaler pipeline)
├── AdaBoost__scaler.joblib      # Scaler for linear models only
└── AdaBoost__features.joblib    # Feature column order (critical!)
```

**Loading in code:**
```python
import joblib
import pandas as pd

model = joblib.load('models/AdaBoost.joblib')
features = joblib.load('models/AdaBoost__features.joblib')

# Prepare input
X = pd.DataFrame([input_dict])[features]

# Predict
score = model.predict(X)[0]
```

## Useful Commands

```bash
# Check installed packages
pip list

# Freeze current environment
pip freeze > requirements.txt

# Run Jupyter notebook
jupyter notebook

# Stop API server (if running)
# Ctrl+C in terminal
```

## Architecture

```
┌─────────────────┐
│   Jupyter       │
│   Notebook      │  EDA6_improved.ipynb
│   (Pipeline)    │  - Data cleaning & imputation
│                 │  - Feature engineering
│                 │  - Model training & export
└────────┬────────┘
         │ Exports to
         ▼
    ┌─────────────────┐
    │   models/       │
    │ *.joblib files  │  AdaBoost, Ridge, Lasso, etc.
    │   + scalers     │  + feature lists
    └────────┬────────┘
             │ Loaded by
             ▼
    ┌─────────────────────┐
    │   FastAPI Backend   │  main.py + predictor.py
    │   Port 8001         │  Composite score computation
    │   6 endpoints       │  Model inference
    └────────┬────────────┘
             │ HTTP API calls
             ▼
    ┌─────────────────────┐
    │  Frontend Dashboard │  index.html + app.js
    │  (Browser)          │  Model comparison charts
    │  Port 8080          │  Live predictor form
    └─────────────────────┘
```

## Performance Summary

**Best Model:** AdaBoost
- **CV R²:** 0.6113 (5-fold average)
- **Test R²:** 0.6921 (20% held-out test set)
- **RMSE:** 0.8019 (predictions within ±0.80 points on 0–5 scale)

**Performance band mapping:**
- Score < 2.75 → "At Risk" (< 55% performance)
- 2.75–3.75 → "Average" (55–75%)
- Score > 3.75 → "Performing" (> 75%)

## Development Notes

- **Data leakage prevention:** All scaling & imputation fit on training set only
- **Cross-validation:** 5-fold stratified on target variable
- **Hyperparameter tuning:** RandomizedSearchCV (40 iterations per model)
- **Model persistence:** Pipeline-based (includes scalers where needed)
- **Feature tracking:** Separate .joblib file for each model's feature order

## References

- **Scikit-learn:** https://scikit-learn.org/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Pandas:** https://pandas.pydata.org/docs/
- **Joblib:** https://joblib.readthedocs.io/

---

**Last updated:** 28 April 2026  
**Status:** ✅ Full pipeline complete | ✅ API & Dashboard live | ✅ Models exported
