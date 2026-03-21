# Student Stress & Academic Performance Predictor

**Group G-10 | BE CSE AI&ML 4th Sem | Batch 2024**

A machine learning system that predicts academic performance risk and discovers natural lifestyle archetypes among engineering students.

## Project Overview

This project builds a two-track ML system:
- **Supervised track:** Predict student performance category (At Risk vs Performing)
- **Unsupervised track:** Discover lifestyle archetypes using K-Means clustering
- **Live predictor:** Web dashboard with instant risk predictions

### Team
- **Hritabrata Das** (Lead) — ML + Backend
- **Hashit** — Data + Unsupervised Learning
- **Abhinav Dadwal** — Analysis + Report

## Project Structure

```
student_stress_predictor/
├── data/
│   ├── raw/                  # Original CSV from Google Form
│   └── processed/            # Cleaned & split datasets
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data cleaning pipeline
│   ├── 03_supervised.ipynb   # Classifier training (4 models)
│   ├── 04_unsupervised.ipynb # K-Means + PCA
│   └── 05_comparison.ipynb   # Results comparison
├── models/                   # Saved ML models (.pkl files)
├── api/                      # FastAPI backend
├── frontend/                 # Dashboard UI
├── report/                   # Final project report
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and start with `01_EDA.ipynb`.

### 4. Run the API Server

```bash
cd api
python -m uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 5. Access the Dashboard

Open `frontend/index.html` in your browser (or serve via HTTP server):

```bash
# Using Python's built-in server
python -m http.server 8080 --directory frontend
```

Navigate to `http://localhost:8080`.

## Key Milestones

| Sprint | Dates | Milestone |
|--------|-------|-----------|
| Sprint 0 | Mar 13–14 | Project setup (you are here) |
| Sprint 1 | Mar 15–20 | Data collection (≥50 responses) |
| **EVAL 1** | **Mar 23–27** | Present problem & dataset |
| Sprint 3–4 | Mar 28–Apr 13 | Model training & evaluation |
| Sprint 5 | Apr 14–20 | API + Dashboard build |
| Sprint 6 | Apr 21–26 | Report + Viva preparation |
| **FINAL EVAL** | **Apr 27–May 1** | Live demo + Viva |

## Dataset Schema

**15 features** collected via Google Form:
1. Attendance %
2. Daily study hours
3. Sleep hours/night
4. Screen time (hrs)
5. Physical activity days/week
6. Extracurricular activities
7. Part-time work
8. Accommodation type (Hostel/Day Scholar)
9. Phone after midnight
10. Academic stress level
11. Overwhelmed by workload
12. Social support rating
13. Skips meals due to workload
14. Life satisfaction rating
15. Current CGPA (used only to create target label)

**Target:** Binary classification (At Risk: CGPA < 7.0 | Performing: CGPA ≥ 7.0)

## Models

### Supervised Learning (4 classifiers)
- **Logistic Regression** — Baseline, interpretable
- **Decision Tree** — Rule-based, prone to overfit
- **Random Forest** — Primary model for API
- **SVM** — Strong on small datasets

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Unsupervised Learning
- **K-Means Clustering** (k=3)
- **PCA** for dimensionality reduction & visualization

**Validation:** Silhouette Score (target ≥ 0.40)

## Next Steps (Sprint 0)

1. ✅ **Project structure initialized** (you are here)
2. 📋 **Google Form created** — Send to batch today
3. 📊 **Data collection** — Target 50–100 responses within 5 days
4. 🔄 **Sprint 1 begins** — EDA + preprocessing

## Useful Commands

```bash
# Check installed packages
pip list

# Freeze current environment
pip freeze > requirements.txt

# Run a notebook cell in terminal
jupyter nbconvert --to script --execute notebook.ipynb

# Clear Jupyter cache
jupyter clean
```

## References

- **Scikit-learn:** https://scikit-learn.org/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Pandas documentation:** https://pandas.pydata.org/docs/

## Notes

- **IMPORTANT:** SMOTE must be fit on **train split only** (no data leakage)
- **Data privacy:** Form collects zero PII
- **Report deadline:** 26 April 2026 (by 11:59 PM)
- **Viva:** All 3 members must explain algorithms independently

---

**Last updated:** 13 March 2026  
**Status:** ✅ Project initialized | 📋 Awaiting data collection
