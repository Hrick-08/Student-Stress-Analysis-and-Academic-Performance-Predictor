# Student Stress & Academic Performance Predictor — PRD v2.0

*Version 2.0 · Updated April 2026*

| **Course:** | 24CAI0203 — SUL |
|---|---|
| **Team:** | Hritabrata Das (Lead) · Harshit · Abhinav Dadwal |
| **Group:** | G-10 · BE CSE AI&ML · 4th Sem · Batch 2024 |
| **Eval 1:** | 23–27 March 2026 |
| **Final Eval:** | 27 April – 1 May 2026 |

---

## 1. Project Overview

### 1.1 Problem Statement

Engineering students often experience declining academic performance due to poor lifestyle habits — inadequate sleep, high screen time, low physical activity, and chronic stress. This project builds a supervised ML system that predicts a student's academic performance score from lifestyle and stress features, served through a FastAPI backend and a web dashboard.

### 1.2 What Changed in v2.0

The original PRD assumed a classification task (At Risk vs Performing) with 4 classifiers trained on primary survey data. The actual implementation differs:

- **Dataset:** `StressLevelDataset_dirty.csv` (secondary dataset)
- **Task:** Regression — predicting `academic_performance` as a continuous score
- **Models trained:** AdaBoost, GradientBoosting, LinearRegression, Ridge, Lasso (all regressors)
- **Feature set used:** Composite-only features (`df_fs2`) derived in `EDA6_final.ipynb`
- **Models saved:** 5 `.joblib` files in `../models/`

This PRD replaces Sections 4, 7, and 8 of v1.0 to reflect exactly how to wire these models into the backend and frontend.

### 1.3 Goals & Success Criteria

| # | Goal | Success Metric | Priority |
|---|---|---|---|
| G1 | Train 5 supervised regressors and compare | Best model Test R² ≥ 0.65 | **P0 — Critical** |
| G2 | Live predictor web dashboard | Working FastAPI + frontend, demo-able at eval | **P1 — High** |
| G3 | Full project report submitted | All 6 report sections completed | **P1 — High** |
| G4 | Viva-ready team | All 3 members can explain every result | **P1 — High** |

---

## 2. Trained Models — Reference

All 5 models were trained on the **composite-only feature set** (`df_fs2`) in `EDA6_final.ipynb` and exported in Section 13 of that notebook.

### 2.1 Model Performance Summary

| Model | Test R² | RMSE | Notes |
|---|---|---|---|
| AdaBoost | 0.6958 | 0.7957 | **Best model — use this for the API** |
| GradientBoosting | 0.6724 | 0.8258 | Second best |
| Ridge | 0.6519 | 0.8512 | Linear — needs scaler |
| LinearRegression | 0.6519 | 0.8512 | Linear — needs scaler |
| Lasso | 0.6517 | 0.8514 | Linear — needs scaler |

### 2.2 Saved Files in `models/`

```
models/
├── AdaBoost.joblib
├── AdaBoost__features.joblib
├── GradientBoosting.joblib
├── GradientBoosting__features.joblib
├── Lasso.joblib
├── Lasso__features.joblib
├── Lasso__scaler.joblib
├── LinearRegression.joblib
├── LinearRegression__features.joblib
├── LinearRegression__scaler.joblib
├── Ridge.joblib
├── Ridge__features.joblib
└── Ridge__scaler.joblib
```

**Rule:** Tree-based models (AdaBoost, GradientBoosting) do NOT need scaling. Linear models (Lasso, Ridge, LinearRegression) MUST have input scaled before prediction using their `__scaler.joblib`.

### 2.3 Input Features (Composite-Only)

The models expect **4 composite scores** plus the remaining raw features. The backend must construct composites from the raw user inputs before calling `model.predict()`.

**Composite scores (constructed in backend):**

| Feature | How it is built | 
|---|---|
| `env_score` | Weighted sum of `safety`, `basic_needs`, `living_conditions` |
| `mental_score` | `0.4 × anxiety_level + 0.4 × depression − 0.6 × self_esteem` |
| `pressure_score` | Weighted sum of `peer_pressure`, `study_load`, `future_career_concerns` |
| `support_score` | Weighted sum of `teacher_student_relationship`, `social_support` |

**Remaining raw features passed directly:**

| Feature | Range |
|---|---|
| `anxiety_level` | 0 – 21 |
| `self_esteem` | 0 – 30 |
| `mental_health_history` | 0 or 1 |
| `depression` | 0 – 27 |
| `headache` | 0 – 5 |
| `blood_pressure` | 1 – 3 |
| `sleep_quality` | 0 (poor), 1 (average), 2 (good) |
| `breathing_problem` | 0 – 5 |
| `stress_level` | 0 (Low), 1 (Medium), 2 (High) |

> Always load `AdaBoost__features.joblib` at startup to get the exact ordered column list. Never hardcode column order.

---

## 3. Project File Structure

```
student_stress_predictor/
├── data/
│   └── raw/
│       └── StressLevelDataset_dirty.csv
├── notebooks/
│   └── EDA6_final.ipynb          ← full pipeline: EDA, preprocessing, training, export
├── models/                       ← exported by Section 13 of EDA6_final.ipynb
│   ├── AdaBoost.joblib
│   ├── AdaBoost__features.joblib
│   ├── GradientBoosting.joblib
│   ├── GradientBoosting__features.joblib
│   ├── Lasso.joblib
│   ├── Lasso__features.joblib
│   ├── Lasso__scaler.joblib
│   ├── LinearRegression.joblib
│   ├── LinearRegression__features.joblib
│   ├── LinearRegression__scaler.joblib
│   ├── Ridge.joblib
│   ├── Ridge__features.joblib
│   └── Ridge__scaler.joblib
├── api/
│   ├── main.py                   ← FastAPI app
│   ├── schemas.py                ← Pydantic request/response models
│   └── predictor.py              ← model loading + inference logic
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
└── README.md
```

---

## 4. Backend — FastAPI

### 4.1 Endpoints

| Method | Endpoint | Purpose | Response |
|---|---|---|---|
| GET | `/health` | Health check | `{"status": "ok", "model": "AdaBoost"}` |
| POST | `/predict` | Predict academic performance score | `PredictionResult` schema |
| GET | `/metrics` | Return stored model evaluation metrics | `{model_name: {test_r2, rmse}}` |
| GET | `/models` | List all available models | `["AdaBoost", "GradientBoosting", ...]` |

### 4.2 Pydantic Schemas (`api/schemas.py`)

The frontend sends raw per-feature values. The backend derives composite scores internally.

```python
# api/schemas.py
from pydantic import BaseModel, Field

class StudentInput(BaseModel):
    # Raw stress/lifestyle features
    anxiety_level:                int   = Field(..., ge=0,  le=21)
    self_esteem:                  int   = Field(..., ge=0,  le=30)
    mental_health_history:        int   = Field(..., ge=0,  le=1)
    depression:                   int   = Field(..., ge=0,  le=27)
    headache:                     int   = Field(..., ge=0,  le=5)
    blood_pressure:               int   = Field(..., ge=1,  le=3)
    sleep_quality:                int   = Field(..., ge=0,  le=2)
    breathing_problem:            int   = Field(..., ge=0,  le=5)
    stress_level:                 int   = Field(..., ge=0,  le=2)
    # Raw components for composite scores
    safety:                       int   = Field(..., ge=0,  le=5)
    basic_needs:                  int   = Field(..., ge=0,  le=5)
    living_conditions:            int   = Field(..., ge=0,  le=5)
    peer_pressure:                int   = Field(..., ge=0,  le=5)
    study_load:                   int   = Field(..., ge=0,  le=5)
    future_career_concerns:       int   = Field(..., ge=0,  le=5)
    teacher_student_relationship: int   = Field(..., ge=0,  le=5)
    social_support:               int   = Field(..., ge=0,  le=5)

class PredictionResult(BaseModel):
    predicted_score:  float   # continuous academic_performance value
    performance_band: str     # "At Risk" | "Average" | "Performing"
    model_used:       str     # "AdaBoost"
    test_r2:          float   # 0.6958
```

### 4.3 Predictor Logic (`api/predictor.py`)

```python
# api/predictor.py
import joblib
import pandas as pd
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_NAME = "AdaBoost"   # change here to swap models

# Load model and feature list
model    = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.joblib"))
features = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}__features.joblib"))

# Linear models (Ridge, Lasso, LinearRegression) also need:
# scaler = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}__scaler.joblib"))

LINEAR_MODELS = {"Ridge", "Lasso", "LinearRegression"}
scaler = None
if MODEL_NAME in LINEAR_MODELS:
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}__scaler.joblib"))

# Composite score weights — must match EDA6_final.ipynb Section 7b
# Equal weights used here as approximation; export exact weights from notebook if needed
def build_features(inp: dict) -> dict:
    env_score      = (inp["safety"] + inp["basic_needs"] + inp["living_conditions"]) / 3
    mental_score   = 0.4 * inp["anxiety_level"] + 0.4 * inp["depression"] - 0.6 * inp["self_esteem"]
    pressure_score = (inp["peer_pressure"] + inp["study_load"] + inp["future_career_concerns"]) / 3
    support_score  = (inp["teacher_student_relationship"] + inp["social_support"]) / 2
    return {
        **inp,
        "env_score":      env_score,
        "mental_score":   mental_score,
        "pressure_score": pressure_score,
        "support_score":  support_score,
    }

def score_to_band(score: float) -> str:
    if score < 55:   return "At Risk"
    if score < 75:   return "Average"
    return "Performing"

def predict(raw_input: dict) -> dict:
    enriched = build_features(raw_input)
    X = pd.DataFrame([enriched])[features]   # enforce exact column order
    if scaler:
        X = scaler.transform(X)
    predicted = float(model.predict(X)[0])
    return {
        "predicted_score":  round(predicted, 2),
        "performance_band": score_to_band(predicted),
        "model_used":       MODEL_NAME,
        "test_r2":          0.6958,
    }
```

### 4.4 FastAPI App (`api/main.py`)

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import StudentInput, PredictionResult
from predictor import predict

app = FastAPI(title="Student Stress Predictor API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "AdaBoost", "task": "regression"}

@app.post("/predict", response_model=PredictionResult)
def predict_endpoint(inp: StudentInput):
    return predict(inp.model_dump())

@app.get("/metrics")
def metrics():
    return {
        "AdaBoost":         {"test_r2": 0.6958, "rmse": 0.7957},
        "GradientBoosting": {"test_r2": 0.6724, "rmse": 0.8258},
        "Ridge":            {"test_r2": 0.6519, "rmse": 0.8512},
        "LinearRegression": {"test_r2": 0.6519, "rmse": 0.8512},
        "Lasso":            {"test_r2": 0.6517, "rmse": 0.8514},
    }

@app.get("/models")
def available_models():
    return ["AdaBoost", "GradientBoosting", "Ridge", "Lasso", "LinearRegression"]
```

### 4.5 Running the API

```bash
cd api/
uvicorn main:app --reload --port 8000

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"anxiety_level":5,"self_esteem":20,"mental_health_history":0,"depression":4,
       "headache":1,"blood_pressure":1,"sleep_quality":2,"breathing_problem":0,
       "stress_level":0,"safety":4,"basic_needs":4,"living_conditions":4,
       "peer_pressure":2,"study_load":3,"future_career_concerns":3,
       "teacher_student_relationship":4,"social_support":4}'
```

---

## 5. Frontend Dashboard

### 5.1 Tab Specifications

| Tab | Content | Data Source | Chart Type |
|---|---|---|---|
| Dataset Overview | Feature distributions, correlation heatmap | Static JSON from EDA notebook | Bar charts, heatmap table |
| Model Comparison | Test R² and RMSE for all 5 models | `GET /metrics` | Grouped bar chart (Chart.js) |
| Live Predictor | Input form → predicted score + band | `POST /predict` | Score display, colour-coded result card |

### 5.2 Live Predictor — API Call (`frontend/app.js`)

```javascript
// frontend/app.js

const API_BASE = "http://localhost:8000";

async function runPrediction() {
  // Collect form values
  const payload = {
    anxiety_level:                parseInt(document.getElementById("anxiety_level").value),
    self_esteem:                  parseInt(document.getElementById("self_esteem").value),
    mental_health_history:        parseInt(document.getElementById("mental_health_history").value),
    depression:                   parseInt(document.getElementById("depression").value),
    headache:                     parseInt(document.getElementById("headache").value),
    blood_pressure:               parseInt(document.getElementById("blood_pressure").value),
    sleep_quality:                parseInt(document.getElementById("sleep_quality").value),
    breathing_problem:            parseInt(document.getElementById("breathing_problem").value),
    stress_level:                 parseInt(document.getElementById("stress_level").value),
    safety:                       parseInt(document.getElementById("safety").value),
    basic_needs:                  parseInt(document.getElementById("basic_needs").value),
    living_conditions:            parseInt(document.getElementById("living_conditions").value),
    peer_pressure:                parseInt(document.getElementById("peer_pressure").value),
    study_load:                   parseInt(document.getElementById("study_load").value),
    future_career_concerns:       parseInt(document.getElementById("future_career_concerns").value),
    teacher_student_relationship: parseInt(document.getElementById("teacher_student_relationship").value),
    social_support:               parseInt(document.getElementById("social_support").value),
  };

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    document.getElementById("result").textContent = "Error — check all fields.";
    return;
  }

  const result = await response.json();
  // result = { predicted_score, performance_band, model_used, test_r2 }

  document.getElementById("score").textContent = result.predicted_score.toFixed(1);
  document.getElementById("band").textContent  = result.performance_band;
  document.getElementById("band").className    = bandClass(result.performance_band);
  document.getElementById("model-note").textContent =
    `Model: ${result.model_used} · Test R² = ${result.test_r2}`;
}

function bandClass(band) {
  if (band === "At Risk")    return "band at-risk";     // red
  if (band === "Average")    return "band average";      // amber
  return "band performing";                              // green
}
```

### 5.3 Model Comparison Tab

```javascript
async function loadMetrics() {
  const res  = await fetch(`${API_BASE}/metrics`);
  const data = await res.json();

  const labels = Object.keys(data);
  const r2     = labels.map(m => data[m].test_r2);
  const rmse   = labels.map(m => data[m].rmse);

  new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Test R²",  data: r2,   backgroundColor: "#2E75B6" },
        { label: "RMSE",     data: rmse, backgroundColor: "#E05C5C" },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1.1 } },
    },
  });
}
```

---

## 6. Switching the Active Model

To use a different model, update two lines in `predictor.py`:

```python
# Example: switch to GradientBoosting (no scaler needed)
MODEL_NAME = "GradientBoosting"

# Example: switch to Ridge (scaler required)
MODEL_NAME = "Ridge"
# scaler loads automatically because "Ridge" is in LINEAR_MODELS set
```

No other code needs to change — `features` is loaded dynamically from `{MODEL_NAME}__features.joblib`.

---

## 7. Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML | scikit-learn 1.4+, joblib | Model training and serialisation |
| Backend | FastAPI 0.111+, Uvicorn, Pydantic v2 | REST API |
| Frontend | HTML + JS + Chart.js 4.x | Dashboard UI |
| Notebook | Jupyter Notebook | EDA, training, model export |

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn
pip install fastapi "uvicorn[standard]" pydantic joblib
pip install jupyter notebook

pip freeze > requirements.txt
```

---

## 8. Sprint Plan (Revised)

| Sprint | Dates | Tasks | Owner |
|---|---|---|---|
| Sprint 3 — API | Apr 14–17 | Build `predictor.py`, `schemas.py`, `main.py`. Test all endpoints. | Hritabrata |
| Sprint 4 — Frontend | Apr 18–21 | Build 3-tab dashboard. Wire Live Predictor to `/predict`. Render metrics from `/metrics`. | Hritabrata |
| Sprint 5 — Report + Viva | Apr 22–26 | Write all 6 report sections. Architecture diagram. Full demo dry-run. | All members |
| **FINAL EVAL** | **Apr 27–May 1** | Live dashboard demo. Report submitted. Viva. | All members |

---

## 9. Demo Profiles for Viva

Run these 3 profiles live during the demo to show the model working:

| Profile | stress_level | sleep_quality | anxiety_level | Expected Band |
|---|---|---|---|---|
| A — Healthy student | 0 (Low) | 2 (Good) | 3 | **Performing** |
| B — Burned out | 2 (High) | 0 (Poor) | 18 | **At Risk** |
| C — Edge case | 1 (Medium) | 1 (Average) | 10 | **Average** |

---

## 10. Team Responsibilities

| Member | Role | Key Deliverables |
|---|---|---|
| **Hritabrata Das** | ML Lead + Backend + Integration | `predictor.py`, `main.py`, dashboard wiring, report compilation |
| **Harshit** | Data + EDA | EDA notebook, preprocessing docs, cluster interpretation |
| **Abhinav Dadwal** | Analysis + Report | Metrics write-up, architecture diagram, comparison notebook, viva slides |

> **All 3 members must be able to explain every algorithm and result independently. Viva is 5/10 marks.**

---

## 11. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Composite weights not matching training | Load exact weights from notebook or export them as a separate `.joblib` file |
| R2 | Feature column order mismatch → wrong predictions | Always load `{ModelName}__features.joblib` and use it to reorder the DataFrame |
| R3 | Linear model called without scaler | The `LINEAR_MODELS` check in `predictor.py` handles this automatically |
| R4 | FastAPI not ready for demo | Fallback: run prediction live in Jupyter using loaded `.joblib` files directly |
| R5 | Team member unprepared for viva | Weekly 20-min verbal mock sessions from Apr 14 |

---

## 12. Project Report Checklist

| # | Section | What to Include |
|---|---|---|
| a | Problem Definition | Motivation, problem statement, why regression, dataset choice |
| b | System Architecture | Diagram: dataset → EDA → composite features → 5 models → joblib export → FastAPI → dashboard |
| c | Methodology | Preprocessing pipeline, composite score construction with correlation weights, model selection |
| d | Dataset Description | All features, cleaning steps (MHH encoding, sleep/stress ordinal encoding, KNN imputation, outlier clipping) |
| e | Comparative Results | R² and RMSE table for all 5 models, residual plots, feature importance for AdaBoost and GradientBoosting |
| f | Limitations | Secondary dataset (not primary survey), R² ceiling ~0.70, composite weights are approximations in API |

---

*PRD v2.0 · Hritabrata Das · Harshit · Abhinav Dadwal · Group G-10*
