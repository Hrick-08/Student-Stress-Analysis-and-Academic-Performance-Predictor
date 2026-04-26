"""
Prediction logic for Student Stress & Academic Performance Predictor

Loads all 5 trained models and provides inference with composite score construction.
"""

import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from schemas import StudentInput, PredictionResult

# Models directory
MODELS_DIR = Path(__file__).parent.parent / "models"

# Available models
MODELS_INFO = {
    "AdaBoost": {"r2": 0.6958, "rmse": 0.7957},
    "GradientBoosting": {"r2": 0.6724, "rmse": 0.8258},
    "Ridge": {"r2": 0.6519, "rmse": 0.8512},
    "LinearRegression": {"r2": 0.6519, "rmse": 0.8512},
    "Lasso": {"r2": 0.6517, "rmse": 0.8514},
}

# Models that require scaling
LINEAR_MODELS = {"Ridge", "Lasso", "LinearRegression"}

# Cache for loaded models
_models_cache = {}
_scalers_cache = {}

def load_model(model_name: str):
    """Load a model and its associated scaler (if needed)."""
    if model_name not in _models_cache:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        _models_cache[model_name] = joblib.load(model_path)
        
        if model_name in LINEAR_MODELS:
            scaler_path = MODELS_DIR / f"{model_name}__scaler.joblib"
            _scalers_cache[model_name] = joblib.load(scaler_path)
        else:
            _scalers_cache[model_name] = None
    
    return _models_cache[model_name], _scalers_cache.get(model_name)

def get_features_list(model_name: str) -> list:
    """Load the exact features list for a model."""
    features_path = MODELS_DIR / f"{model_name}__features.joblib"
    return joblib.load(features_path)

def compute_composite_scores(student_data: dict) -> dict:
    """
    Compute composite scores from raw component features.
    
    Composites:
    - env_score: (safety + basic_needs + living_conditions) / 3
    - mental_score: 0.4 * anxiety + 0.4 * depression - 0.6 * self_esteem
    - pressure_score: (peer_pressure + study_load + future_career) / 3
    - support_score: (teacher_relationship + social_support) / 2
    
    Args:
        student_data: Dict with all input features
        
    Returns:
        Dict with all features including computed composites
    """
    result = student_data.copy()
    
    # Environmental wellbeing
    env_score = (
        student_data["safety"] +
        student_data["basic_needs"] +
        student_data["living_conditions"]
    ) / 3
    result["env_score"] = env_score
    
    # Mental health burden
    # Higher anxiety/depression = higher burden (positive)
    # Higher self_esteem = lower burden (negative coefficient)
    mental_score = (
        0.4 * student_data["anxiety_level"] +
        0.4 * student_data["depression"] -
        0.6 * student_data["self_esteem"]
    )
    result["mental_score"] = mental_score
    
    # Academic pressure
    pressure_score = (
        student_data["peer_pressure"] +
        student_data["study_load"] +
        student_data["future_career_concerns"]
    ) / 3
    result["pressure_score"] = pressure_score
    
    # Social support
    support_score = (
        student_data["teacher_student_relationship"] +
        student_data["social_support"]
    ) / 2
    result["support_score"] = support_score
    
    return result

def score_to_band(score: float) -> str:
    """
    Convert prediction score to performance band.
    
    Scale: 0-5
    - At Risk: < 2.75 (< 55%)
    - Average: 2.75-3.75 (55-75%)
    - Performing: > 3.75 (> 75%)
    """
    if score < 2.75:
        return "At Risk"
    elif score < 3.75:
        return "Average"
    else:
        return "Performing"

def predict_single_model(
    student_input: StudentInput,
    model_name: str
) -> PredictionResult:
    """
    Predict academic performance using a single model.
    
    Args:
        student_input: StudentInput with all raw features
        model_name: Name of model to use
        
    Returns:
        PredictionResult with score, band, and model metrics
    """
    # Convert input to dict for composite computation
    input_dict = student_input.model_dump()
    
    # Compute composite scores
    enriched_dict = compute_composite_scores(input_dict)
    
    # Get features list and reorder
    features = get_features_list(model_name)
    X = pd.DataFrame([enriched_dict])[features]
    
    # Load model and scaler
    model, scaler = load_model(model_name)
    
    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    predicted_score = float(model.predict(X)[0])
    
    # Get metrics
    metrics = MODELS_INFO[model_name]
    
    return PredictionResult(
        predicted_score=round(predicted_score, 2),
        performance_band=score_to_band(predicted_score),
        model_name=model_name,
        model_r2=metrics["r2"],
        model_rmse=metrics["rmse"]
    )

def predict_all_models(student_input: StudentInput) -> dict:
    """
    Predict using all available models.
    
    Args:
        student_input: StudentInput with all raw features
        
    Returns:
        Dict mapping model names to PredictionResult objects
    """
    predictions = {}
    for model_name in MODELS_INFO.keys():
        try:
            predictions[model_name] = predict_single_model(student_input, model_name)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            predictions[model_name] = None
    
    return predictions
