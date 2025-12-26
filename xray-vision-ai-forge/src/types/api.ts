/**
 * TypeScript type definitions for backend API integration.
 *
 * These types match the backend Pydantic schemas and API response formats,
 * providing type safety for all frontend-backend communication.
 */

// ============================================================================
// Configuration Types (matching backend schemas.py)
// ============================================================================

export interface SystemConfig {
  img_size?: [number, number];
  image_extension?: string;
  batch_size?: number;
  sample_fraction?: number;
  validation_split?: number;
  seed?: number;
}

export interface PathsConfig {
  base_path?: string;
  main_images_folder?: string;
  images_subfolder?: string;
  metadata_filename?: string;
}

export interface ColumnsConfig {
  patient_id?: string;
  target?: string;
  filename?: string;
}

export interface ExperimentConfigBackend {
  // Model parameters
  learning_rate?: number;
  epochs?: number;
  batch_size?: number;
  weight_decay?: number;
  freeze_backbone?: boolean;
  dropout_rate?: number;
  fine_tune_layers_count?: number;
  num_classes?: number;
  monitor_metric?: 'val_loss' | 'val_acc' | 'val_f1' | 'val_auroc';

  // Training parameters
  early_stopping_patience?: number;
  reduce_lr_patience?: number;
  reduce_lr_factor?: number;
  min_lr?: number;
  validation_split?: number;

  // Federated Learning parameters
  num_rounds?: number;
  num_clients?: number;
  clients_per_round?: number;
  local_epochs?: number;

  // System parameters
  device?: 'auto' | 'cpu' | 'cuda' | 'mps';
  num_workers?: number;

  // Image processing parameters
  color_mode?: 'RGB' | 'L';
  use_imagenet_norm?: boolean;
  augmentation_strength?: number;
  use_custom_preprocessing?: boolean;
  validate_images_on_init?: boolean;
  pin_memory?: boolean;
  persistent_workers?: boolean;
  prefetch_factor?: number;

  // Custom preprocessing parameters
  contrast_stretch?: boolean;
  adaptive_histogram?: boolean;
  edge_enhancement?: boolean;
  lower_percentile?: number;
  upper_percentile?: number;
  clip_limit?: number;
  edge_strength?: number;
}

export interface OutputConfig {
  checkpoint_dir?: string;
  results_dir?: string;
  log_dir?: string;
}

export interface LoggingConfig {
  level?: string;
  format?: string;
  file_logging?: boolean;
}

export interface ConfigurationUpdateRequest {
  system?: SystemConfig;
  paths?: PathsConfig;
  columns?: ColumnsConfig;
  experiment?: ExperimentConfigBackend;
  output?: OutputConfig;
  logging?: LoggingConfig;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiResponse<T = any> {
  message?: string;
  data?: T;
  error?: string;
  status?: string;
}

export interface ConfigurationResponse {
  message: string;
  updated_fields: number;
  fields: string[];
}

export interface TrainingStartResponse {
  message: string;
  experiment_name: string;
  experiment_id?: string;
  checkpoint_dir?: string;
  logs_dir?: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
}

export interface TrainingStatusResponse {
  experiment_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress?: number;
  current_epoch?: number;
  total_epochs?: number;
  message?: string;
}

export interface DatasetSummary {
  totalImages: number;
  classes: string[];
  classDistribution?: Record<string, number>;
}

// ============================================================================
// Training Metrics Types
// ============================================================================

export interface EpochMetrics {
  loss: number;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auroc?: number;
  [key: string]: number | undefined;
}

export interface TrainingHistory {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  [key: string]: number;
}

export interface FinalMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  confusionMatrix?: number[][];
}

export interface TrainingResults {
  status: 'completed' | 'failed';
  experiment_name: string;
  final_metrics?: FinalMetrics;
  training_history?: TrainingHistory[];
  best_checkpoint_path?: string;
  error?: string;
  error_type?: string;
}

// ============================================================================
// Results API Types
// ============================================================================

export interface ExperimentResults {
  experiment_id: string;
  status: string;
  metadata: ExperimentMetadata;
  final_metrics: ResultsFinalMetrics;
  confusion_matrix: ConfusionMatrix | null;
  training_history: ResultsTrainingHistoryEntry[];
  total_epochs: number;
}

export interface ExperimentMetadata {
  experiment_name: string;
  start_time: string;
  end_time: string;
  total_epochs: number;
  best_epoch: number;
  best_val_recall: number;
  best_val_loss: number;
  training_duration_seconds?: number;
  training_duration_formatted?: string;
  max_epochs?: number;
  num_devices?: number;
  accelerator?: string;
}

export interface ResultsFinalMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc: number;
  loss: number;
}

export interface ConfusionMatrix {
  true_positives: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  // Summary statistics calculated from CM values
  sensitivity: number;      // TP / (TP + FN) - true positive rate
  specificity: number;      // TN / (TN + FP) - true negative rate
  precision_cm: number;     // TP / (TP + FP)
  accuracy_cm: number;      // (TP + TN) / Total
  f1_cm: number;            // 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
}

export interface ResultsTrainingHistoryEntry {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface ExperimentMetricsResponse {
  experiment_id: string;
  final_metrics: ResultsFinalMetrics;
  best_metrics: {
    best_val_recall: number;
    best_val_loss: number;
    best_epoch: number;
  };
  total_epochs: number;
}

export interface ExperimentArtifact {
  type: string;
  filename: string;
  path: string;
  size_bytes: number;
  format: string;
}

export interface ExperimentArtifactsResponse {
  experiment_id: string;
  artifact_type: string | null;
  artifacts: ExperimentArtifact[];
  total_count: number;
}

export interface ExperimentListItem {
  experiment_id: string;
  experiment_name: string;
  status: string;
  start_time: string;
  end_time: string;
  total_epochs: number;
  best_val_recall: number;
}

export interface ExperimentListResponse {
  experiments: ExperimentListItem[];
  total: number;
  limit: number;
  offset: number;
  returned: number;
}

export interface ComparisonMetricEntry {
  centralized: number;
  federated: number;
  difference: number;
  percent_difference: number;
}

export interface ComparisonResults {
  comparison_id: string;
  centralized_results: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc: number;
    loss: number;
    training_history: ResultsTrainingHistoryEntry[];
  };
  federated_results: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc: number;
    loss: number;
    training_history: ResultsTrainingHistoryEntry[];
  };
  comparison_metrics: {
    [key: string]: ComparisonMetricEntry;
  };
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export type WebSocketMessageType =
  | 'connected'
  | 'training_start'
  | 'training_end'
  | 'training_mode'
  | 'round_metrics'
  | 'epoch_start'
  | 'epoch_end'
  | 'round_start'
  | 'round_end'
  | 'local_epoch'
  | 'client_training_start'
  | 'client_progress'
  | 'client_complete'
  | 'early_stopping'
  | 'status'
  | 'error'
  | 'pong';

export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  data: T;
  timestamp: string;
}

export interface EpochStartData {
  epoch: number;
  total_epochs: number;
}

export interface EpochEndData {
  epoch: number;
  phase: 'train' | 'val' | 'test';
  metrics: EpochMetrics;
}

export interface RoundStartData {
  run_id?: number | null;
  round: number;
  total_rounds: number;
  client_id?: string;
  experiment_name?: string;
  local_epochs?: number;
  status?: string;
  timestamp?: string;
}

export interface RoundEndData {
  round: number;
  client_id?: string;
  fit_metrics: EpochMetrics;
  eval_metrics: EpochMetrics;
}

export interface LocalEpochData {
  run_id?: number | null;
  round: number;
  local_epoch: number;
  client_id: string;
  metrics: EpochMetrics;
  timestamp?: string;
  status?: string;
  experiment_name?: string;
}

export interface ClientTrainingStartData {
  run_id?: number | null;
  client_id: string;
  round: number;
  status: string;
  local_epochs: number;
  num_samples: number;
  experiment_name?: string;
  timestamp?: string;
}

export interface ClientProgressData {
  run_id?: number | null;
  round: number;
  client_id: string;
  local_epoch: number;
  metrics: {
    train_loss: number;
    learning_rate: number;
    num_samples: number;
    batch_count?: number;
    batches_processed?: number;
    epoch_progress?: string;
    overall_progress_percent?: number;
    total_batches?: number;
    [key: string]: any;
  };
  timestamp?: string;
  status?: string;
  experiment_name?: string;
}

export interface ClientCompleteData {
  run_id?: number | null;
  client_id: string;
  status: string;
  total_rounds: number;
  total_local_epochs: number;
  best_round?: number;
  best_val_accuracy?: number;
  best_val_loss?: number;
  total_samples_trained?: number;
  training_duration?: string;
}

export interface StatusData {
  status: 'started' | 'running' | 'completed' | 'failed';
  message?: string;
}

export interface ErrorData {
  error: string;
  error_type?: string;
  traceback?: string;
}

export interface TrainingStartData {
  run_id: number;
  experiment_name: string;
  max_epochs: number;
  training_mode: string;
}

export interface TrainingEndData {
  run_id: number;
  status: 'completed' | 'failed';
  experiment_name: string;
  best_epoch?: number;
  best_val_recall?: number;
  total_epochs: number;
  training_duration?: string;
  reason?: string;
}

export interface EarlyStoppingData {
  epoch: number;
  best_metric_value: number;
  metric_name: string;
  patience: number;
  reason?: string;
  timestamp?: string;
}

export interface TrainingModeData {
  is_federated: boolean;
  num_rounds: number;
  num_clients: number;
}

export interface RoundMetricsData {
  round: number;
  total_rounds: number;
  metrics: {
    loss?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    auroc?: number;
  };
}

// ============================================================================
// Experiment Types
// ============================================================================

export interface ExperimentRequest {
  data_zip: File;
  checkpoint_dir?: string;
  logs_dir?: string;
  experiment_name?: string;
  csv_filename?: string;
}

export interface CentralizedExperimentRequest extends ExperimentRequest {
  checkpoint_dir?: string;
  logs_dir?: string;
}

export interface FederatedExperimentRequest extends ExperimentRequest {
  // Federated specific fields if any
}

// ============================================================================
// Log Types
// ============================================================================

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  context?: Record<string, any>;
}

export interface ExperimentLogs {
  experiment_id: string;
  logs: LogEntry[];
  total_lines: number;
}

// ============================================================================
// Upload Types
// ============================================================================

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

export interface UploadResponse {
  message: string;
  dataset_summary?: DatasetSummary;
  dataset_id?: string;
}
// ============================================================================
// Chat & Knowledge Base Types
// ============================================================================

export interface KnowledgeBaseDocument {
  source: string;
  paper_id?: string;
  display_name: string;
  type: 'arxiv' | 'uploaded';
  chunk_count: number;
}

export interface KnowledgeBaseResponse {
  documents: KnowledgeBaseDocument[];
  total_count: number;
}
