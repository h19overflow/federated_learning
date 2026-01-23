/**
 * Run and experiment summary types.
 * Shared across SavedExperiments, ChatSidebar, and other components.
 */

export interface FederatedInfo {
  num_rounds: number;
  num_clients: number;
  has_server_evaluation: boolean;
  best_accuracy?: number;
  best_recall?: number;
  latest_round?: number;
  latest_accuracy?: number;
}

export interface RunSummary {
  id: number;
  training_mode: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  best_val_recall: number;
  best_val_accuracy: number;
  metrics_count: number;
  run_description: string | null;
  federated_info: FederatedInfo | null;
  final_epoch_stats?: {
    sensitivity: number;
    specificity: number;
    precision_cm: number;
    accuracy_cm: number;
    f1_cm: number;
  } | null;
}
