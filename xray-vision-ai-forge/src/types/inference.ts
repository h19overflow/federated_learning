/**
 * TypeScript types for Inference API
 *
 * These types match the Pydantic schemas from:
 * federated_pneumonia_detection/src/api/endpoints/schema/inference_schemas.py
 */

export type PredictionClass = "NORMAL" | "PNEUMONIA";

export interface InferencePrediction {
  predicted_class: PredictionClass;
  confidence: number; // 0.0 to 1.0
  pneumonia_probability: number; // 0.0 to 1.0
  normal_probability: number; // 0.0 to 1.0
}

export interface RiskAssessment {
  risk_level: string; // LOW, MODERATE, HIGH, CRITICAL
  false_negative_risk: string;
  factors: string[];
}

export interface ClinicalInterpretation {
  summary: string;
  confidence_explanation: string;
  risk_assessment: RiskAssessment;
  recommendations: string[];
  disclaimer: string;
}

export interface InferenceResponse {
  success: boolean;
  prediction: InferencePrediction;
  clinical_interpretation?: ClinicalInterpretation;
  heatmap_base64?: string;
  model_version: string;
  processing_time_ms: number;
}

export interface InferenceError {
  success: false;
  error: string;
  detail: string;
}

export interface HealthCheckResponse {
  status: string; // healthy, degraded, unhealthy
  model_loaded: boolean;
  gpu_available: boolean;
  model_version?: string;
}

// Batch prediction types
export interface SingleImageResult {
  filename: string;
  success: boolean;
  prediction?: InferencePrediction;
  clinical_interpretation?: ClinicalInterpretation;
  error?: string;
  processing_time_ms: number;
}

export interface BatchSummaryStats {
  total_images: number;
  successful: number;
  failed: number;
  normal_count: number;
  pneumonia_count: number;
  avg_confidence: number;
  avg_processing_time_ms: number;
  high_risk_count: number;
}

export interface BatchInferenceResponse {
  success: boolean;
  results: SingleImageResult[];
  summary: BatchSummaryStats;
  model_version: string;
  total_processing_time_ms: number;
}
