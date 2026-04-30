import {
  StudentInput,
  PredictionResult,
  AllModelsResult,
  ModelsListResponse,
  HealthResponse,
} from "../types/api";

const API_BASE = "http://localhost:8001";

function apiUrl(path: string): string {
  return `${API_BASE.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(apiUrl("health"));
  if (!response.ok) throw new Error("API not reachable");
  return response.json();
}

export async function fetchModels(): Promise<ModelsListResponse> {
  const response = await fetch(apiUrl("models"));
  if (!response.ok) throw new Error("Failed to fetch models");
  return response.json();
}

export async function predictWithModel(
  data: StudentInput,
  modelName: string
): Promise<PredictionResult> {
  const response = await fetch(
    apiUrl(`predict?model_name=${modelName}`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }
  );
  if (!response.ok) throw new Error(`Prediction failed: ${response.statusText}`);
  return response.json();
}

export async function predictAllModels(
  data: StudentInput
): Promise<AllModelsResult> {
  const response = await fetch(apiUrl("predict-all"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error(`Prediction failed: ${response.statusText}`);
  return response.json();
}