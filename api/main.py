"""
FastAPI Backend for Student Stress & Academic Performance Predictor

Serves REST API endpoints for:
- Health check
- Model comparison
- Single model prediction
- All models prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import StudentInput, PredictionResult, AllModelsResult, ModelsListResponse
from predictor import predict_single_model, predict_all_models, MODELS_INFO

# Initialize FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend on different port)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Student Stress Predictor API is running",
        "models_available": list(MODELS_INFO.keys()),
        "task": "regression"
    }

@app.post("/predict")
def predict_endpoint(
    student_input: StudentInput,
    model_name: str = "AdaBoost"
) -> PredictionResult:
    """
    Predict academic performance using specified model.
    
    Query Parameters:
    - model_name (str): One of "AdaBoost", "GradientBoosting", "Ridge", "LinearRegression", "Lasso"
                        Defaults to "AdaBoost" (best model)
    """
    if model_name not in MODELS_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from: {list(MODELS_INFO.keys())}"
        )
    
    try:
        result = predict_single_model(student_input, model_name)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict-all")
def predict_all_endpoint(student_input: StudentInput) -> AllModelsResult:
    """
    Predict using all available models and return comparison.
    """
    try:
        predictions = predict_all_models(student_input)
        
        return AllModelsResult(
            input_summary={
                "stress_level": student_input.stress_level,
                "anxiety_level": student_input.anxiety_level,
                "sleep_quality": student_input.sleep_quality,
                "self_esteem": student_input.self_esteem,
                "social_support": student_input.social_support,
            },
            predictions=predictions,
            selected_model="AdaBoost"  # Best model
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/models")
def list_models() -> ModelsListResponse:
    """
    List all available models with their performance metrics.
    """
    models = [
        {
            "name": name,
            "test_r2": info["r2"],
            "rmse": info["rmse"]
        }
        for name, info in MODELS_INFO.items()
    ]
    return ModelsListResponse(models=models)

@app.get("/metrics")
def get_metrics():
    """
    Return model performance metrics.
    """
    return MODELS_INFO

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
