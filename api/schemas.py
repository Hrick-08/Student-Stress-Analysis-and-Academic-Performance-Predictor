"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict

class StudentInput(BaseModel):
    """Input schema for student data - 14 features."""
    
    attendance_pct: float = Field(..., ge=0, le=100, description="Attendance percentage")
    study_hours: int = Field(..., ge=1, le=4, description="1:<2hrs, 2:2-4hrs, 3:4-6hrs, 4:>6hrs")
    sleep_hrs: float = Field(..., ge=4, le=12, description="Hours of sleep per night")
    screen_time: float = Field(..., ge=0, le=18, description="Daily screen time in hours")
    activity_days: int = Field(..., ge=0, le=7, description="Physical activity days per week")
    extracurricular: int = Field(..., ge=0, le=3, description="0:None, 1:1 activity, 2:2 activities, 3:3+ activities")
    part_time_work: int = Field(..., ge=0, le=1, description="0:No, 1:Yes")
    accommodation: int = Field(..., ge=0, le=1, description="0:Day Scholar, 1:Hostel")
    phone_midnight: int = Field(..., ge=0, le=2, description="0:Rarely, 1:Sometimes, 2:Often")
    stress_level: int = Field(..., ge=1, le=10, description="Academic stress level (1-10)")
    overwhelmed: int = Field(..., ge=0, le=3, description="0:Never, 1:Sometimes, 2:Often, 3:Always")
    social_support: int = Field(..., ge=1, le=5, description="Social support rating (1-5)")
    skips_meals: int = Field(..., ge=0, le=2, description="0:Never, 1:Rarely, 2:Often")
    life_satisfaction: int = Field(..., ge=1, le=10, description="Life satisfaction rating (1-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "attendance_pct": 85,
                "study_hours": 3,
                "sleep_hrs": 7,
                "screen_time": 6,
                "activity_days": 3,
                "extracurricular": 1,
                "part_time_work": 0,
                "accommodation": 1,
                "phone_midnight": 1,
                "stress_level": 5,
                "overwhelmed": 1,
                "social_support": 4,
                "skips_meals": 0,
                "life_satisfaction": 7
            }
        }

class PredictionResult(BaseModel):
    """Output schema for prediction results."""
    
    label: int = Field(..., description="0: At Risk, 1: Performing")
    label_text: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Confidence score (0-1)")
    cluster_id: int = Field(..., description="K-Means cluster assignment")
    cluster_name: str = Field(..., description="Lifestyle archetype name")

class MetricsResponse(BaseModel):
    """Model evaluation metrics."""
    
    random_forest: Dict[str, float] = Field(...)
    logistic_regression: Dict[str, float] = Field(...)
    decision_tree: Dict[str, float] = Field(...)
    svm: Dict[str, float] = Field(...)

class ClusterInfo(BaseModel):
    """Information about a single cluster."""
    
    id: int
    name: str
    description: str
    centroid: List[float]
    percentage: float = Field(default=0.0, description="% of dataset in this cluster")

class ClustersResponse(BaseModel):
    """K-Means cluster information."""
    
    clusters: List[ClusterInfo]

class FeatureImportanceItem(BaseModel):
    """Single feature importance score."""
    
    feature: str
    importance: float

class FeatureImportanceResponse(BaseModel):
    """Random Forest feature importance."""
    
    features: List[FeatureImportanceItem]
