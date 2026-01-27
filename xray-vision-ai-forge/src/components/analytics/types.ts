/**
 * Analytics type definitions for training run analytics and statistics.
 *
 * This module contains all TypeScript interfaces and types used for
 * analytics data structures, including mode statistics, top performing runs,
 * and aggregated analytics data.
 */

/**
 * Statistics for a specific training mode (centralized or federated).
 *
 * Contains aggregated metrics across all training runs of a particular mode,
 * including average performance metrics and duration.
 */
export interface ModeStatistics {
  /** Total number of runs for this training mode */
  count: number;
  /** Average accuracy across all runs (0-1 scale) */
  avg_accuracy: number;
  /** Average precision across all runs (0-1 scale) */
  avg_precision: number;
  /** Average recall across all runs (0-1 scale) */
  avg_recall: number;
  /** Average F1 score across all runs (0-1 scale) */
  avg_f1: number;
  /** Average duration in minutes across all runs */
  avg_duration_minutes: number;
}

/**
 * Represents a single top-performing training run.
 *
 * Contains detailed metrics for individual training runs that are
 * displayed in the top performers table.
 */
export interface TopRun {
  /** Unique identifier for the training run */
  run_id: number;
  /** Training mode type: 'centralized' or 'federated' */
  training_mode: string;
  /** Best accuracy achieved during the run (0-1 scale) */
  best_accuracy: number;
  /** Best precision achieved during the run (0-1 scale) */
  best_precision: number;
  /** Best recall achieved during the run (0-1 scale) */
  best_recall: number;
  /** Best F1 score achieved during the run (0-1 scale) */
  best_f1: number;
  /** Total duration of the run in minutes */
  duration_minutes: number;
  /** ISO 8601 timestamp of when the run started */
  start_time: string;
  /** Current status: 'completed', 'failed', or 'in_progress' */
  status: string;
}

/**
 * Complete analytics data structure returned from the API.
 *
 * Aggregates all analytics information including total run counts,
 * success rates, mode-specific statistics, and top performing runs.
 */
export interface AnalyticsData {
  /** Total number of training runs across all modes */
  total_runs: number;
  /** Overall success rate (0-1 scale) across all runs */
  success_rate: number;
  /** Aggregated statistics for centralized training mode */
  centralized: ModeStatistics;
  /** Aggregated statistics for federated training mode */
  federated: ModeStatistics;
  /** Array of top performing runs based on selected metrics */
  top_runs: TopRun[];
}

/**
 * Metric type used for sorting top runs table.
 *
 * Defines which performance metric to use when ordering the top runs list.
 */
export type SortMetric =
  | "best_accuracy"
  | "best_precision"
  | "best_recall"
  | "best_f1";

/**
 * Sort direction for ordering table data.
 *
 * Determines whether to sort in ascending or descending order.
 */
export type SortDirection = "asc" | "desc";
