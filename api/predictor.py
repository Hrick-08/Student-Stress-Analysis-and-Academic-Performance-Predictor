"""
Prediction logic for the Random Forest model
"""

import numpy as np
from schemas import StudentInput, PredictionResult

# Cluster names mapping
CLUSTER_NAMES = {
    0: "Balanced Achiever",
    1: "Sleep-Deprived Grinder",
    2: "Overloaded & Burned Out"
}

# Feature order (must match training pipeline)
FEATURE_ORDER = [
    'attendance_pct',
    'study_hours',
    'sleep_hrs',
    'screen_time',
    'activity_days',
    'extracurricular',
    'part_time_work',
    'accommodation',
    'phone_midnight',
    'stress_level',
    'overwhelmed',
    'social_support',
    'skips_meals',
    'life_satisfaction'
]

def predict_performance(
    student_input: StudentInput,
    scaler,
    model,
    kmeans,
    pca
) -> PredictionResult:
    """
    Predict performance category and cluster assignment for a student.
    
    Args:
        student_input: StudentInput schema with 14 features
        scaler: StandardScaler fitted on training data
        model: Random Forest classifier
        kmeans: K-Means model
        pca: PCA transformer
    
    Returns:
        PredictionResult with label, confidence, cluster info
    """
    
    # Extract features in correct order
    x = np.array([[getattr(student_input, f) for f in FEATURE_ORDER]])
    
    # Scale features
    x_scaled = scaler.transform(x)
    
    # Predict label
    label = int(model.predict(x_scaled)[0])
    
    # Get prediction confidence
    proba = model.predict_proba(x_scaled)[0]
    confidence = float(proba[label])
    
    # PCA transformation
    x_pca = pca.transform(x_scaled)
    
    # Cluster assignment
    cluster_id = int(kmeans.predict(x_pca)[0])
    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    
    # Return prediction result
    return PredictionResult(
        label=label,
        label_text="Performing" if label == 1 else "At Risk",
        confidence=round(confidence, 3),
        cluster_id=cluster_id,
        cluster_name=cluster_name
    )
