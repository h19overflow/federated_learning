/**
 * API Client Service for backend communication.
 *
 * Provides type-safe HTTP methods for all backend API endpoints.
 * Handles request/response formatting, error handling, and configuration.
 *
 * Dependencies:
 * - Environment variables for API base URL
 * - Type definitions from @/types/api
 */

import {
  ConfigurationUpdateRequest,
  ConfigurationResponse,
  TrainingStartResponse,
  TrainingStatusResponse,
  TrainingResults,
  UploadProgress,
  DatasetSummary,
  ExperimentLogs,
  ExperimentResults,
  ExperimentMetricsResponse,
  ExperimentArtifactsResponse,
  ExperimentListResponse,
  ComparisonResults,
} from '@/types/api';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = 'http://127.0.0.1:8001';
const API_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT) || 300000; // 5 minutes default

// ============================================================================
// Error Handling
// ============================================================================

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'ApiError';
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

    throw new ApiError(errorMessage, response.status, errorData);
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

// ============================================================================
// Configuration Endpoints
// ============================================================================

export const configurationApi = {
  /**
   * Get current backend configuration
   */
  async getCurrentConfiguration(): Promise<{ config: any; config_path: string }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/config/current`);
    return handleResponse<{ config: any; config_path: string }>(response);
  },

  /**
   * Update backend configuration
   */
  async setConfiguration(
    config: ConfigurationUpdateRequest
  ): Promise<ConfigurationResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/config/update`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    return handleResponse<ConfigurationResponse>(response);
  },
};

// ============================================================================
// Dataset Upload Endpoints
// ============================================================================

export const datasetApi = {
  /**
   * Upload dataset ZIP file
   */
  async uploadDataset(
    file: File,
    onProgress?: (progress: UploadProgress) => void
  ): Promise<{ message: string, dataset_summary?: DatasetSummary }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // Track upload progress
      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            onProgress({
              loaded: e.loaded,
              total: e.total,
              percentage: (e.loaded / e.total) * 100,
            });
          }
        });
      }

      // Handle completion
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch (e) {
            reject(new ApiError('Failed to parse response'));
          }
        } else {
          reject(new ApiError(`Upload failed: ${xhr.statusText}`, xhr.status));
        }
      });

      // Handle errors
      xhr.addEventListener('error', () => {
        reject(new ApiError('Upload failed: Network error'));
      });

      xhr.addEventListener('abort', () => {
        reject(new ApiError('Upload aborted'));
      });

      // Note: Actual upload will be part of training request
      // This is a placeholder for future separate upload endpoint
      xhr.open('POST', `${API_BASE_URL}/upload/dataset`);

      const formData = new FormData();
      formData.append('file', file);

      xhr.send(formData);
    });
  },
};

// ============================================================================
// Training Experiment Endpoints
// ============================================================================

export const experimentsApi = {
  /**
   * Start centralized training
   */
  async startCentralizedTraining(
    dataZip: File,
    experimentName: string = 'pneumonia_centralized',
    checkpointDir: string = 'results/centralized/checkpoints',
    logsDir: string = 'results/centralized/logs',
    csvFilename: string = 'stage2_train_metadata.csv'
  ): Promise<TrainingStartResponse> {
    const formData = new FormData();
    formData.append('data_zip', dataZip);
    formData.append('experiment_name', experimentName);
    formData.append('checkpoint_dir', checkpointDir);
    formData.append('logs_dir', logsDir);
    formData.append('csv_filename', csvFilename);

    const response = await fetchWithTimeout(
      `${API_BASE_URL}/experiments/centralized/train`,
      {
        method: 'POST',
        body: formData,
      },
      600000 // 10 minute timeout for upload
    );

    return handleResponse<TrainingStartResponse>(response);
  },

  /**
   * Start federated training
   */
  async startFederatedTraining(
    dataZip: File,
    experimentName: string = 'pneumonia_federated',
    csvFilename: string = 'stage2_train_metadata.csv',
    numServerRounds: number = 3
  ): Promise<TrainingStartResponse> {
    const formData = new FormData();
    formData.append('data_zip', dataZip);
    formData.append('experiment_name', experimentName);
    formData.append('csv_filename', csvFilename);
    formData.append('num_server_rounds', numServerRounds.toString());

    const response = await fetchWithTimeout(
      `${API_BASE_URL}/experiments/federated/train`,
      {
        method: 'POST',
        body: formData,
      },
      600000 // 10 minute timeout for upload
    );

    return handleResponse<TrainingStartResponse>(response);
  },

  /**
   * Get training status
   */
  async getTrainingStatus(experimentId: string): Promise<TrainingStatusResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/experiments/status/${experimentId}`
    );

    return handleResponse<TrainingStatusResponse>(response);
  },

  /**
   * List all experiments
   */
  async listExperiments(): Promise<{ experiments: TrainingStatusResponse[] }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/experiments/list`
    );

    return handleResponse<{ experiments: TrainingStatusResponse[] }>(response);
  },
};

// ============================================================================
// Logging Endpoints
// ============================================================================

export const loggingApi = {
  /**
   * Get experiment logs
   */
  async getExperimentLogs(
    experimentId: string,
    lines: number = 100
  ): Promise<ExperimentLogs> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/logs/experiments/${experimentId}?lines=${lines}`
    );

    return handleResponse<ExperimentLogs>(response);
  },

  /**
   * Tail experiment logs (get last N lines)
   */
  async tailExperimentLogs(
    experimentId: string,
    lines: number = 50
  ): Promise<ExperimentLogs> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/logs/experiments/${experimentId}/tail?lines=${lines}`
    );

    return handleResponse<ExperimentLogs>(response);
  },

  /**
   * List available experiment logs
   */
  async listAvailableLogs(): Promise<{ experiments: string[] }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/logs/experiments`
    );

    return handleResponse<{ experiments: string[] }>(response);
  },
};

// ============================================================================
// Results Endpoints
// ============================================================================

export const resultsApi = {
  /**
   * List all training runs with summary information
   */
  async listRuns(): Promise<{ runs: any[]; total: number }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/runs/list`
    );
    return handleResponse<{ runs: any[]; total: number }>(response);
  },

  /**
   * Get complete experiment results including metrics, history, and metadata
   */
  async getExperimentResults(experimentId: string): Promise<ExperimentResults> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/results/experiments/${experimentId}`
    );
    return handleResponse<ExperimentResults>(response);
  },

  /**
   * Get training run results by run_id (from WebSocket)
   */
  async getRunMetrics(runId: number): Promise<ExperimentResults> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/runs/${runId}/metrics`
    );
    return handleResponse<ExperimentResults>(response);
  },

  /**
   * Get federated round metrics for a run
   */
  async getFederatedRounds(runId: number): Promise<{
    is_federated: boolean;
    num_rounds: number;
    num_clients: number;
    rounds: Array<{
      round: number;
      metrics: Record<string, number>;
    }>;
  }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/runs/${runId}/federated-rounds`
    );
    return handleResponse<{
      is_federated: boolean;
      num_rounds: number;
      num_clients: number;
      rounds: Array<{
        round: number;
        metrics: Record<string, number>;
      }>;
    }>(response);
  },

  /**
   * Get server-side evaluation metrics for a federated run
   */
  async getServerEvaluation(runId: number): Promise<{
    run_id: number;
    is_federated: boolean;
    has_server_evaluation: boolean;
    evaluations: Array<{
      round: number;
      loss: number;
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
      auroc?: number;
      confusion_matrix?: {
        true_positives: number;
        true_negatives: number;
        false_positives: number;
        false_negatives: number;
      };
      num_samples?: number;
      evaluation_time?: string;
    }>;
    summary: {
      total_rounds: number;
      latest_round: number;
      latest_metrics: Record<string, number>;
      best_accuracy: { value: number | null; round: number | null };
      best_recall: { value: number | null; round: number | null };
      best_f1_score: { value: number | null; round: number | null };
    };
  }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/runs/${runId}/server-evaluation`
    );
    return handleResponse(response);
  },

  /**
   * Get experiment metrics only (lighter weight than full results)
   */
  async getExperimentMetrics(experimentId: string): Promise<ExperimentMetricsResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/results/experiments/${experimentId}/metrics`
    );
    return handleResponse<ExperimentMetricsResponse>(response);
  },

  /**
   * Get experiment artifacts (files, checkpoints, logs)
   */
  async getExperimentArtifacts(
    experimentId: string,
    artifactType?: string
  ): Promise<ExperimentArtifactsResponse> {
    const url = artifactType
      ? `${API_BASE_URL}/results/experiments/${experimentId}/artifacts?artifact_type=${artifactType}`
      : `${API_BASE_URL}/results/experiments/${experimentId}/artifacts`;

    const response = await fetchWithTimeout(url);
    return handleResponse<ExperimentArtifactsResponse>(response);
  },

  /**
   * List all experiments with optional filtering
   */
  async listExperiments(
    status?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<ExperimentListResponse> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());

    const response = await fetchWithTimeout(
      `${API_BASE_URL}/results/experiments?${params.toString()}`
    );
    return handleResponse<ExperimentListResponse>(response);
  },

  /**
   * Get comparison results between centralized and federated training
   */
  async getComparisonResults(comparisonId: string): Promise<ComparisonResults> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/results/comparison/${comparisonId}`
    );
    return handleResponse<ComparisonResults>(response);
  },

  /**
   * Download training run artifacts by run_id
   */
  async downloadRunArtifact(
    runId: number,
    format: 'json' | 'csv' | 'summary'
  ): Promise<Blob> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/runs/${runId}/download/${format}`
    );

    if (!response.ok) {
      throw new ApiError(`Download failed: ${response.statusText}`, response.status);
    }

    return response.blob();
  },

  /**
   * Trigger download in browser for run artifacts
   */
  async triggerRunDownload(
    runId: number,
    format: 'json' | 'csv' | 'summary'
  ): Promise<void> {
    const blob = await this.downloadRunArtifact(runId, format);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;

    // Let the backend set the filename via Content-Disposition header
    // But provide a fallback
    a.download = `run_${runId}_${format}`;

    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },

  /**
   * @deprecated Use triggerRunDownload instead for run-based downloads
   * Download training artifacts (legacy experiment-based)
   */
  async downloadArtifact(
    experimentId: string,
    artifactType: 'metrics_json' | 'metrics_csv' | 'summary' | 'logs' | 'checkpoint'
  ): Promise<Blob> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/results/experiments/${experimentId}/download/${artifactType}`
    );

    if (!response.ok) {
      throw new ApiError(`Download failed: ${response.statusText}`, response.status);
    }

    return response.blob();
  },

  /**
   * @deprecated Use triggerRunDownload instead for run-based downloads
   * Trigger download in browser (legacy experiment-based)
   */
  async triggerDownload(
    experimentId: string,
    artifactType: 'metrics_json' | 'metrics_csv' | 'summary' | 'logs' | 'checkpoint',
    filename?: string
  ): Promise<void> {
    const blob = await this.downloadArtifact(experimentId, artifactType);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || `${experimentId}_${artifactType}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check API health
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/`, {}, 5000);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get API base URL
 */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

// Export all API modules
export default {
  configuration: configurationApi,
  dataset: datasetApi,
  experiments: experimentsApi,
  logging: loggingApi,
  results: resultsApi,
  checkHealth: checkApiHealth,
  getBaseUrl: getApiBaseUrl,
};
