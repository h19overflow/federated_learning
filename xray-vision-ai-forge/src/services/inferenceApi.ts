/**
 * Inference API Service
 *
 * Handles communication with the pneumonia detection inference endpoint.
 * Follows patterns from api.ts for consistency.
 */

import {
  InferenceResponse,
  InferenceError,
  HealthCheckResponse,
  BatchInferenceResponse,
  HeatmapResponse,
  BatchHeatmapResponse,
} from "@/types/inference";
import { getEnv } from "@/utils/env";
import { sanitizeFilename } from "@/utils/validation";

// Configuration
const env = getEnv();
const API_BASE_URL = env.VITE_API_BASE_URL;
const API_TIMEOUT = env.VITE_API_TIMEOUT; // 30 seconds default for inference

// API Error class (matching api.ts pattern)
export class InferenceApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: unknown,
  ) {
    super(message);
    this.name = "InferenceApiError";
  }
}

/**
 * Handle API response and throw error if not ok
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;
    let errorData: { detail?: string; message?: string } | undefined;

    try {
      errorData = await response.json() as { detail?: string; message?: string };
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      // If parsing JSON fails, use status text
    }

    throw new InferenceApiError(errorMessage, response.status, errorData);
  }

  try {
    return await response.json();
  } catch {
    // If no JSON body, return empty object
    return {} as T;
  }
}

/**
 * Create fetch request with timeout
 */
function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = API_TIMEOUT,
): Promise<Response> {
  return Promise.race([
    fetch(url, options),
    new Promise<Response>((_, reject) =>
      setTimeout(() => reject(new Error("Request timeout")), timeout),
    ),
  ]);
}

/**
 * Check inference service health
 */
export async function checkInferenceHealth(): Promise<HealthCheckResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/inference/health`,
    {},
    5000, // Short timeout for health check
  );

  return handleResponse<HealthCheckResponse>(response);
}

/**
 * Predict pneumonia from X-ray image
 *
 * @param file - X-ray image file (PNG or JPEG)
 * @param includeClinicalInterpretation - Whether to include AI clinical interpretation
 * @returns Prediction results with optional clinical interpretation
 */
export async function predictImage(
  file: File,
  includeClinicalInterpretation: boolean = true,
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const url = new URL(`${API_BASE_URL}/api/inference/predict`);
  url.searchParams.append(
    "include_clinical_interpretation",
    String(includeClinicalInterpretation),
  );

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: "POST",
      body: formData,
    },
    API_TIMEOUT,
  );

  return handleResponse<InferenceResponse>(response);
}

/**
 * Predict pneumonia from multiple X-ray images (batch processing)
 *
 * @param files - Array of X-ray image files (PNG or JPEG) - max 50 images
 * @param includeClinicalInterpretation - Whether to include AI clinical interpretation
 * @returns Batch prediction results with summary statistics
 */
export async function batchPredictImages(
  files: File[],
  includeClinicalInterpretation: boolean = false,
): Promise<BatchInferenceResponse> {
  const formData = new FormData();

  // Append all files with the same field name 'files'
  files.forEach((file) => {
    formData.append("files", file);
  });

  const url = new URL(`${API_BASE_URL}/api/inference/predict-batch`);
  url.searchParams.append(
    "include_clinical_interpretation",
    String(includeClinicalInterpretation),
  );

  // Longer timeout for batch processing (2 minutes)
  const batchTimeout = 120000;

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: "POST",
      body: formData,
    },
    batchTimeout,
  );

  return handleResponse<BatchInferenceResponse>(response);
}

/**
 * Generate PDF report for batch predictions
 *
 * @param batchResult - Batch inference response data
 * @returns PDF blob for download
 */
export async function generateBatchPdfReport(
  batchResult: BatchInferenceResponse,
): Promise<Blob> {
  const requestBody = {
    results: batchResult.results.map((r) => ({
      filename: r.filename,
      success: r.success,
      prediction: r.prediction
        ? {
            predicted_class: r.prediction.predicted_class,
            confidence: r.prediction.confidence,
            pneumonia_probability: r.prediction.pneumonia_probability,
            normal_probability: r.prediction.normal_probability,
          }
        : null,
      error: r.error || null,
    })),
    summary: {
      total_images: batchResult.summary.total_images,
      successful: batchResult.summary.successful,
      failed: batchResult.summary.failed,
      pneumonia_count: batchResult.summary.pneumonia_count,
      normal_count: batchResult.summary.normal_count,
      avg_confidence: batchResult.summary.avg_confidence,
      high_risk_count: batchResult.summary.high_risk_count || 0,
    },
    model_version: batchResult.model_version || "unknown",
  };

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/reports/batch`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    },
    60000, // 60 seconds for PDF generation
  );

  if (!response.ok) {
    let errorMessage = `PDF Generation Error: ${response.status}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Use status text
    }
    throw new InferenceApiError(errorMessage, response.status);
  }

  return response.blob();
}

/**
 * Download a blob as a file
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = sanitizeFilename(filename);
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Generate GradCAM heatmap for a single X-ray image
 *
 * @param file - X-ray image file (PNG or JPEG)
 * @param colormap - Colormap for visualization (jet, hot, viridis)
 * @param alpha - Heatmap overlay transparency (0.1-0.9)
 * @returns Heatmap response with base64-encoded overlay
 */
export async function generateHeatmap(
  file: File,
  colormap: string = "jet",
  alpha: number = 0.4,
): Promise<HeatmapResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const url = new URL(`${API_BASE_URL}/api/inference/heatmap`);
  url.searchParams.append("colormap", colormap);
  url.searchParams.append("alpha", String(alpha));

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: "POST",
      body: formData,
    },
    API_TIMEOUT,
  );

  return handleResponse<HeatmapResponse>(response);
}

/**
 * Generate GradCAM heatmaps for multiple X-ray images
 *
 * @param files - Array of X-ray image files (PNG or JPEG) - max 50 images
 * @param colormap - Colormap for visualization (jet, hot, viridis)
 * @param alpha - Heatmap overlay transparency (0.1-0.9)
 * @returns Batch heatmap response with results for each image
 */
export async function generateBatchHeatmaps(
  files: File[],
  colormap: string = "jet",
  alpha: number = 0.4,
): Promise<BatchHeatmapResponse> {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });

  const url = new URL(`${API_BASE_URL}/api/inference/heatmap-batch`);
  url.searchParams.append("colormap", colormap);
  url.searchParams.append("alpha", String(alpha));

  // Longer timeout for batch heatmap generation (3 minutes)
  const batchTimeout = 180000;

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: "POST",
      body: formData,
    },
    batchTimeout,
  );

  return handleResponse<BatchHeatmapResponse>(response);
}

/**
 * Generate PDF report for batch predictions with heatmaps
 *
 * @param batchResult - Batch inference response data
 * @param heatmaps - Map of filename to heatmap base64 data
 * @param originalImages - Map of filename to original image base64 data
 * @returns PDF blob for download
 */
export async function generateBatchPdfReportWithHeatmaps(
  batchResult: BatchInferenceResponse,
  heatmaps: Map<string, string>,
  originalImages: Map<string, string>,
): Promise<Blob> {
  const requestBody = {
    results: batchResult.results.map((r) => ({
      filename: r.filename,
      success: r.success,
      prediction: r.prediction
        ? {
            predicted_class: r.prediction.predicted_class,
            confidence: r.prediction.confidence,
            pneumonia_probability: r.prediction.pneumonia_probability,
            normal_probability: r.prediction.normal_probability,
          }
        : null,
      error: r.error || null,
      heatmap_base64: heatmaps.get(r.filename) || null,
      original_image_base64: originalImages.get(r.filename) || null,
    })),
    summary: {
      total_images: batchResult.summary.total_images,
      successful: batchResult.summary.successful,
      failed: batchResult.summary.failed,
      pneumonia_count: batchResult.summary.pneumonia_count,
      normal_count: batchResult.summary.normal_count,
      avg_confidence: batchResult.summary.avg_confidence,
      high_risk_count: batchResult.summary.high_risk_count || 0,
    },
    model_version: batchResult.model_version || "unknown",
    include_heatmaps: true,
  };

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/reports/batch-with-heatmaps`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    },
    120000, // 2 minutes for PDF generation with heatmaps
  );

  if (!response.ok) {
    let errorMessage = `PDF Generation Error: ${response.status}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Use status text
    }
    throw new InferenceApiError(errorMessage, response.status);
  }

  return response.blob();
}

// Export API module
export default {
  checkHealth: checkInferenceHealth,
  predict: predictImage,
  batchPredict: batchPredictImages,
  generateBatchPdfReport,
  generateHeatmap,
  generateBatchHeatmaps,
  generateBatchPdfReportWithHeatmaps,
  downloadBlob,
};
