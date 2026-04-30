export interface StudentInput {
  safety: number;
  basic_needs: number;
  living_conditions: number;
  anxiety_level: number;
  depression: number;
  self_esteem: number;
  peer_pressure: number;
  study_load: number;
  future_career_concerns: number;
  teacher_student_relationship: number;
  social_support: number;
  mental_health_history: number;
  headache: number;
  blood_pressure: number;
  sleep_quality: number;
  breathing_problem: number;
  extracurricular_activities: number;
  stress_level: number;
}

export interface PredictionResult {
  predicted_score: number;
  performance_band: "At Risk" | "Average" | "Performing";
  model_name: string;
  model_r2: number;
  model_rmse: number;
}

export interface AllModelsResult {
  input_summary: Record<string, number>;
  predictions: Record<string, PredictionResult | null>;
  selected_model: string;
}

export interface ModelInfo {
  name: string;
  test_r2: number;
  rmse: number;
}

export interface ModelsListResponse {
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  message: string;
  models_available: string[];
  task: string;
}

export const DEFAULT_STUDENT_INPUT: StudentInput = {
  safety: 3,
  basic_needs: 3,
  living_conditions: 3,
  anxiety_level: 10,
  depression: 13,
  self_esteem: 15,
  peer_pressure: 2,
  study_load: 3,
  future_career_concerns: 3,
  teacher_student_relationship: 3,
  social_support: 3,
  mental_health_history: 0,
  headache: 2,
  blood_pressure: 2,
  sleep_quality: 2,
  breathing_problem: 0,
  extracurricular_activities: 2,
  stress_level: 1,
};

export const MODELS_LIST = [
  { name: "AdaBoost", recommended: true },
  { name: "GradientBoosting", recommended: false },
  { name: "Ridge", recommended: false },
  { name: "LinearRegression", recommended: false },
  { name: "Lasso", recommended: false },
  { name: "ElasticNet", recommended: false },
];