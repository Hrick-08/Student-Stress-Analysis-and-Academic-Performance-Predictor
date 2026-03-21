"""
FastAPI Backend for Student Stress & Academic Performance Predictor

Serves REST API endpoints for:
- Health check
- Performance predictions
- Model metrics
- Cluster information
- Feature importance
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
from pathlib import Path

# Import schemas and predictor logic
from schemas import StudentInput, PredictionResult, MetricsResponse, ClustersResponse, FeatureImportanceResponse
from predictor import predict_performance

# Load models on startup
models_dir = Path(__file__).parent.parent / "models"
scaler = None
model = None
kmeans = None
pca = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global scaler, model, kmeans, pca
    
    print("Loading models...")
    scaler = joblib.load(models_dir / "scaler.pkl")
    model = joblib.load(models_dir / "random_forest.pkl")
    kmeans = joblib.load(models_dir / "kmeans.pkl")
    pca = joblib.load(models_dir / "pca.pkl")
    print("✅ Models loaded successfully")
    
    yield
    
    print("Cleaning up...")

# Initialize FastAPI app with CORS
app = FastAPI(
    title="Student Stress Predictor API",
    description="Predicts academic performance risk based on lifestyle habits",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "random_forest",
        "models_available": ["random_forest", "logistic_regression", "decision_tree", "svm"]
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(student_input: StudentInput):
    """
    Predict performance category for a student.
    
    Returns:
    - label: 0 = At Risk, 1 = Performing
    - confidence: Probability of predicted class
    - cluster_id: K-Means cluster assignment
    - cluster_name: Lifestyle archetype name
    """
    result = predict_performance(student_input, scaler, model, kmeans, pca)
    return result

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return model evaluation metrics (to be populated from training notebooks)."""
    # Placeholder - will be populated after training
    return {
        "random_forest": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0
        }
    }

@app.get("/clusters", response_model=ClustersResponse)
async def get_clusters():
    """Return K-Means cluster information."""
    # Placeholder - will be populated from training
    return {
        "clusters": [
            {
                "id": 0,
                "name": "Balanced Achiever",
                "description": "Good sleep, low stress, regular activity",
                "centroid": []
            }
        ]
    }

@app.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Return Random Forest feature importance."""
    # Placeholder - will be populated from training
    return {"features": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
