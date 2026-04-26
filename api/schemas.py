"""
Pydantic schemas for Student Stress Predictor API
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

class StudentInput(BaseModel):
    """Input schema for student stress/lifestyle data.
    
    Raw component features used to compute composites:
    - env components: safety, basic_needs, living_conditions
    - mental components: anxiety_level, depression, self_esteem
    - pressure components: peer_pressure, study_load, future_career_concerns
    - support components: teacher_student_relationship, social_support
    
    Direct features (no composite):
    - mental_health_history, headache, blood_pressure, sleep_quality,
      breathing_problem, extracurricular_activities, stress_level
    """
    
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "safety": 4, "basic_needs": 4, "living_conditions": 4,
                "anxiety_level": 5, "depression": 4, "self_esteem": 20,
                "peer_pressure": 2, "study_load": 3, "future_career_concerns": 3,
                "teacher_student_relationship": 4, "social_support": 4,
                "mental_health_history": 0, "headache": 1, "blood_pressure": 1,
                "sleep_quality": 2, "breathing_problem": 0,
                "extracurricular_activities": 2, "stress_level": 0
            }
        }
    )
    
    # Raw components for composites
    safety: int = Field(..., ge=0, le=5, description="Safety rating (0-5)")
    basic_needs: int = Field(..., ge=0, le=5, description="Basic needs satisfaction (0-5)")
    living_conditions: int = Field(..., ge=0, le=5, description="Living conditions rating (0-5)")
    anxiety_level: int = Field(..., ge=0, le=21, description="Anxiety level (0-21)")
    depression: int = Field(..., ge=0, le=27, description="Depression score (0-27)")
    self_esteem: int = Field(..., ge=0, le=30, description="Self esteem (0-30)")
    peer_pressure: int = Field(..., ge=0, le=5, description="Peer pressure (0-5)")
    study_load: int = Field(..., ge=0, le=5, description="Study load (0-5)")
    future_career_concerns: int = Field(..., ge=0, le=5, description="Future career concerns (0-5)")
    teacher_student_relationship: int = Field(..., ge=0, le=5, description="Teacher-student relationship (0-5)")
    social_support: int = Field(..., ge=0, le=5, description="Social support (0-5)")
    
    # Direct features
    mental_health_history: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    headache: int = Field(..., ge=0, le=5, description="Headache frequency (0-5)")
    blood_pressure: int = Field(..., ge=1, le=3, description="Blood pressure (1: Low, 2: Normal, 3: High)")
    sleep_quality: int = Field(..., ge=0, le=2, description="Sleep quality (0: Poor, 1: Average, 2: Good)")
    breathing_problem: int = Field(..., ge=0, le=5, description="Breathing problems (0-5)")
    extracurricular_activities: int = Field(..., ge=0, le=5, description="Extracurricular activities (0-5)")
    stress_level: int = Field(..., ge=0, le=2, description="Stress level (0: Low, 1: Medium, 2: High)")

class PredictionResult(BaseModel):
    """Output schema for a single model prediction."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_score: float = Field(..., description="Predicted academic performance (continuous)")
    performance_band: str = Field(..., description="Band: 'At Risk' | 'Average' | 'Performing'")
    model_name: str = Field(..., description="Name of the model used")
    model_r2: float = Field(..., description="Test R² for this model")
    model_rmse: float = Field(..., description="Test RMSE for this model")

class AllModelsResult(BaseModel):
    """Output with predictions from all available models."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    input_summary: Dict = Field(..., description="Summary of input features")
    predictions: Dict[str, Optional[PredictionResult]] = Field(..., description="Model name -> prediction")
    selected_model: str = Field(..., description="Currently selected model for deployment")

class MetricsResponse(BaseModel):
    """Model performance metrics."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    test_r2: float
    rmse: float

class ModelsListResponse(BaseModel):
    """List of available models with metrics."""
    
    models: List[Dict] = Field(..., description="List of {name, test_r2, rmse}")
