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
} from '@/types/inference';

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8001';
const API_TIMEOUT = 30000; // 30 seconds for inference

// API Error class (matching api.ts pattern)
export class InferenceApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'InferenceApiError';
  }
}

/**
 * Handle API response and throw error if not ok
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;
    let errorData: any;

    try {
      errorData = await response.json();
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
  timeout: number = API_TIMEOUT
): Promise<Response> {
  return Promise.race([
    fetch(url, options),
    new Promise<Response>((_, reject) =>
      setTimeout(() => reject(new Error('Request timeout')), timeout)
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
    5000 // Short timeout for health check
  );

  return handleResponse<HealthCheckResponse>(response);
}

/**
 * Predict pneumonia from X-ray image
 *
 * @param file - X-ray image file (PNG or JPEG)
 * @param includeClinicalInterpretation - Whether to include AI clinical interpretation
 * @param includeHeatmap - Whether to include GradCAM heatmap visualization
 * @returns Prediction results with optional clinical interpretation and heatmap
 */
export async function predictImage(
  file: File,
  includeClinicalInterpretation: boolean = false,
  includeHeatmap: boolean = true
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`${API_BASE_URL}/api/inference/predict`);
  url.searchParams.append('include_clinical_interpretation', String(includeClinicalInterpretation));
  url.searchParams.append('include_heatmap', String(includeHeatmap));

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: 'POST',
      body: formData,
    },
    API_TIMEOUT
  );

  return handleResponse<InferenceResponse>(response);
}

/**
 * Generate GradCAM heatmap for a single image (on-demand)
 *
 * Used in batch mode to generate heatmaps when viewing individual results.
 *
 * @param file - X-ray image file (PNG or JPEG)
 * @returns Heatmap as base64-encoded PNG
 */
export async function generateHeatmap(file: File): Promise<{ heatmap_base64: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/inference/heatmap`,
    {
      method: 'POST',
      body: formData,
    },
    API_TIMEOUT
  );

  return handleResponse<{ heatmap_base64: string }>(response);
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
  includeClinicalInterpretation: boolean = false
): Promise<BatchInferenceResponse> {
  const formData = new FormData();

  // Append all files with the same field name 'files'
  files.forEach((file) => {
    formData.append('files', file);
  });

  const url = new URL(`${API_BASE_URL}/api/inference/predict-batch`);
  url.searchParams.append('include_clinical_interpretation', String(includeClinicalInterpretation));

  // Longer timeout for batch processing (2 minutes)
  const batchTimeout = 120000;

  const response = await fetchWithTimeout(
    url.toString(),
    {
      method: 'POST',
      body: formData,
    },
    batchTimeout
  );

  return handleResponse<BatchInferenceResponse>(response);
}

/**
 * Generate a PDF report for a single X-ray prediction
 *
 * @param file - X-ray image file (PNG or JPEG)
 * @param includeHeatmap - Include GradCAM heatmap in report
 * @param includeClinical - Include clinical interpretation
 * @returns PDF blob for download
 */
export async function generatePdfReport(
  file: File,
  includeHeatmap: boolean = true,
  includeClinical: boolean = false
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('include_heatmap', String(includeHeatmap));
  formData.append('include_clinical', String(includeClinical));

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/inference/report/single`,
    {
      method: 'POST',
      body: formData,
    },
    60000 // 60 second timeout for PDF generation
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new InferenceApiError(
      errorData.detail || 'Failed to generate PDF report',
      response.status,
      errorData
    );
  }

  return response.blob();
}

/**
 * Generate a summary PDF report for batch predictions
 *
 * @param results - Array of prediction results
 * @param summary - Summary statistics
 * @returns PDF blob for download
 */
export async function generateBatchPdfReport(
  results: any[],
  summary: any
): Promise<Blob> {
  const formData = new FormData();
  formData.append('results_json', JSON.stringify(results));
  formData.append('summary_json', JSON.stringify(summary));

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/inference/report/batch-summary`,
    {
      method: 'POST',
      body: formData,
    },
    60000 // 60 second timeout for PDF generation
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new InferenceApiError(
      errorData.detail || 'Failed to generate batch PDF report',
      response.status,
      errorData
    );
  }

  return response.blob();
}

/**
 * Generate a comprehensive PDF report with image appendix
 *
 * @param files - Array of image files to include in appendix
 * @param results - Array of prediction results
 * @param summary - Summary statistics
 * @param includeHeatmaps - Whether to generate heatmaps (default: true)
 * @param maxAppendixImages - Maximum images to include in appendix (default: 10)
 * @returns PDF blob for download
 */
export async function generateBatchPdfReportWithImages(
  files: File[],
  results: any[],
  summary: any,
  includeHeatmaps: boolean = true,
  maxAppendixImages: number = 10
): Promise<Blob> {
  const formData = new FormData();

  // Add files
  files.slice(0, maxAppendixImages).forEach((file) => {
    formData.append('files', file);
  });

  formData.append('results_json', JSON.stringify(results));
  formData.append('summary_json', JSON.stringify(summary));
  formData.append('include_heatmaps', String(includeHeatmaps));
  formData.append('max_appendix_images', String(maxAppendixImages));

  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/inference/report/batch-with-images`,
    {
      method: 'POST',
      body: formData,
    },
    180000 // 3 minute timeout for report with images/heatmaps
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new InferenceApiError(
      errorData.detail || 'Failed to generate batch PDF report with images',
      response.status,
      errorData
    );
  }

  return response.blob();
}

// Export API module
export default {
  checkHealth: checkInferenceHealth,
  predict: predictImage,
  generateHeatmap,
  generatePdfReport,
  generateBatchPdfReport,
  generateBatchPdfReportWithImages,
  batchPredict: batchPredictImages,
};
