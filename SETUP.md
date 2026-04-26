# Setup & Running the Student Stress Predictor Application

This document provides step-by-step instructions for running the complete frontend and backend of the Student Stress & Academic Performance Predictor.

## System Requirements

- **Python 3.8+** (preferably 3.10 or 3.11)
- **Node.js** (optional, for frontend dev server)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)

## Quick Start

### 1. Backend Setup

#### Step 1a: Navigate to API directory
```bash
cd api
```

#### Step 1b: Activate virtual environment (if not already active)
```bash
# Windows
..\venv\Scripts\activate

# macOS/Linux
source ../venv/bin/activate
```

#### Step 1c: Install dependencies
```bash
pip install fastapi uvicorn pydantic joblib pandas numpy scikit-learn
```

#### Step 1d: Start the API Server
```bash
python -m uvicorn main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['...']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [PID] using StatReload
```

> **Note:** If port 8000 is in use, try `--port 8888` or another available port. If you change the port, update `API_BASE` in `frontend/app.js`.

### 2. Frontend Setup

#### Step 2a: Navigate to frontend directory (in a new terminal)
```bash
cd frontend
```

#### Step 2b: Serve the frontend
**Option A: Using Python's built-in HTTP server**
```bash
python -m http.server 8001
```

**Option B: Using Node.js http-server (if installed)**
```bash
npx http-server -p 8001
```

**Expected Output:**
```
Serving HTTP on 127.0.0.1 port 8001 (http://127.0.0.1:8001) ...
```

### 3. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8001
```

## API Endpoints

The backend API provides the following endpoints:

### Health Check
```
GET /health
```
**Response:**
```json
{
    "status": "ok",
    "message": "Student Stress Predictor API is running",
    "models_available": ["AdaBoost", "GradientBoosting", "Ridge", "LinearRegression", "Lasso"],
    "task": "regression"
}
```

### Get Available Models
```
GET /models
```
**Response:**
```json
{
    "models": [
        {
            "name": "AdaBoost",
            "test_r2": 0.6958,
            "rmse": 0.7957
        },
        ...
    ]
}
```

### Get Model Metrics
```
GET /metrics
```
**Response:**
```json
{
    "AdaBoost": {"r2": 0.6958, "rmse": 0.7957},
    "GradientBoosting": {"r2": 0.6724, "rmse": 0.8258},
    ...
}
```

### Single Model Prediction
```
POST /predict?model_name=AdaBoost
Content-Type: application/json

{
    "safety": 4,
    "basic_needs": 4,
    "living_conditions": 4,
    "anxiety_level": 5,
    "depression": 4,
    "self_esteem": 20,
    "peer_pressure": 2,
    "study_load": 3,
    "future_career_concerns": 3,
    "teacher_student_relationship": 4,
    "social_support": 4,
    "mental_health_history": 0,
    "headache": 1,
    "blood_pressure": 1,
    "sleep_quality": 2,
    "breathing_problem": 0,
    "extracurricular_activities": 2,
    "stress_level": 0
}
```

**Response:**
```json
{
    "predicted_score": 65.32,
    "performance_band": "Average",
    "model_name": "AdaBoost",
    "model_r2": 0.6958,
    "model_rmse": 0.7957
}
```

### All Models Prediction
```
POST /predict-all
Content-Type: application/json

{
    ...same request body as above...
}
```

**Response:**
```json
{
    "input_summary": {
        "stress_level": 0,
        "anxiety_level": 5,
        "sleep_quality": 2,
        "self_esteem": 20,
        "social_support": 4
    },
    "predictions": {
        "AdaBoost": {
            "predicted_score": 65.32,
            "performance_band": "Average",
            "model_name": "AdaBoost",
            "model_r2": 0.6958,
            "model_rmse": 0.7957
        },
        ...
    },
    "selected_model": "AdaBoost"
}
```

## Troubleshooting

### Issue: Port Already in Use
**Problem:** `[WinError 10013] An attempt was made to access a socket in a way forbidden`

**Solutions:**
1. Use a different port: `python -m uvicorn main:app --port 8888`
2. Kill the process using the port:
   - Windows: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
   - macOS/Linux: `lsof -i :8000` then `kill -9 <PID>`
3. Restart your computer

### Issue: ModuleNotFoundError
**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:** Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Models Not Loading
**Problem:** `FileNotFoundError: models/AdaBoost.joblib`

**Solution:** Ensure you're running from the correct directory:
```bash
# Correct
cd "C:\path\to\project\api"
python -m uvicorn main:app --reload --port 8000

# Wrong (will fail)
cd "C:\path\to\project"
python -m uvicorn api.main:app --reload --port 8000
```

### Issue: Frontend Can't Connect to API
**Problem:** CORS errors or "API Offline" message in browser console

**Solutions:**
1. Verify the API is running: Visit `http://127.0.0.1:8000/health` in your browser
2. Check the port matches in `frontend/app.js`: `const API_BASE = "http://localhost:8000";`
3. If using different port, update the JavaScript: `const API_BASE = "http://localhost:8888";`
4. Clear browser cache and reload (Ctrl+Shift+R or Cmd+Shift+R)

## Project Structure

```
.
├── api/
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic models for request/response
│   └── predictor.py         # Model loading and prediction logic
├── frontend/
│   ├── index.html           # Main HTML page
│   ├── app.js               # JavaScript for interactivity
│   └── style.css            # Styling
├── models/
│   ├── AdaBoost.joblib
│   ├── AdaBoost__features.joblib
│   ├── AdaBoost__scaler.joblib
│   ├── GradientBoosting.joblib
│   ├── GradientBoosting__features.joblib
│   ├── Ridge.joblib
│   ├── Ridge__features.joblib
│   ├── Ridge__scaler.joblib
│   ├── LinearRegression.joblib
│   ├── LinearRegression__features.joblib
│   ├── LinearRegression__scaler.joblib
│   ├── Lasso.joblib
│   ├── Lasso__features.joblib
│   └── Lasso__scaler.joblib
├── notebooks/
│   └── EDA6_final.ipynb     # Training and model export
├── data/
│   └── processed/           # Cleaned datasets
└── requirements.txt         # Python dependencies
```

## Model Information

All 5 models were trained on the **composite-only feature set** (df_fs2) from the notebook:

| Model | Test R² | RMSE | Status |
|-------|---------|------|--------|
| AdaBoost | 0.6958 | 0.7957 | ⭐ **Recommended** |
| GradientBoosting | 0.6724 | 0.8258 | ✓ Available |
| Ridge | 0.6519 | 0.8512 | ✓ Available |
| LinearRegression | 0.6519 | 0.8512 | ✓ Available |
| Lasso | 0.6517 | 0.8514 | ✓ Available |

## Features Used by All Models

**Composite Scores** (calculated from raw components):
- `env_score`: (safety + basic_needs + living_conditions) / 3
- `mental_score`: 0.4 × anxiety_level + 0.4 × depression - 0.6 × self_esteem
- `pressure_score`: (peer_pressure + study_load + future_career_concerns) / 3
- `support_score`: (teacher_student_relationship + social_support) / 2

**Direct Features** (passed without modification):
- mental_health_history (0-1)
- headache (0-5)
- blood_pressure (1-3)
- sleep_quality (0-2)
- breathing_problem (0-5)
- extracurricular_activities (0-5)
- stress_level (0-2)

## Development Tips

### API Development
- The API automatically reloads when you modify files in the `api/` directory
- Use `--reload` flag in development, remove for production
- Check API documentation at: `http://127.0.0.1:8000/docs` (Swagger UI)

### Frontend Development
- Open browser developer tools (F12) to see console messages
- Check Network tab to see API requests
- Frontend code is plain JavaScript (no build step needed)
- You can edit CSS and HTML directly and refresh the browser

### Testing the API with curl

```bash
# Health check
curl http://127.0.0.1:8000/health

# Get models
curl http://127.0.0.1:8000/models

# Predict with AdaBoost
curl -X POST http://127.0.0.1:8000/predict?model_name=AdaBoost \
  -H "Content-Type: application/json" \
  -d '{
    "safety": 4, "basic_needs": 4, "living_conditions": 4,
    "anxiety_level": 5, "depression": 4, "self_esteem": 20,
    "peer_pressure": 2, "study_load": 3, "future_career_concerns": 3,
    "teacher_student_relationship": 4, "social_support": 4,
    "mental_health_history": 0, "headache": 1, "blood_pressure": 1,
    "sleep_quality": 2, "breathing_problem": 0,
    "extracurricular_activities": 2, "stress_level": 0
  }'
```

## Demo Profiles for Viva

### Profile A: Healthy Student
```
Stress Level: Low (0)
Sleep Quality: Good (2)
Anxiety Level: Low (3)
Self Esteem: High (25)
Expected: Performing ✅
```

### Profile B: Burned Out Student
```
Stress Level: High (2)
Sleep Quality: Poor (0)
Anxiety Level: Very High (18)
Self Esteem: Low (8)
Expected: At Risk ⚠️
```

### Profile C: Edge Case
```
Stress Level: Medium (1)
Sleep Quality: Average (1)
Anxiety Level: Medium (10)
Self Esteem: Medium (15)
Expected: Average 📊
```

## Support & Contact

For questions or issues:
1. Check the Troubleshooting section above
2. Review the API logs in the terminal
3. Check browser console (F12 → Console tab)
4. Verify all files are in the correct directories

---

**Last Updated:** April 2026
**Version:** 2.0
**Team:** Group G-10, BE CSE AI&ML, Batch 2024
