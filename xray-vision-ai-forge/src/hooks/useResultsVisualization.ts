import { useState, useEffect, useMemo } from "react";
import { toast } from "sonner";
import api from "@/services/api";
import type {
  ExperimentResults,
  ComparisonResults,
  ResultsTrainingHistoryEntry,
} from "@/types/api";
import { ExperimentConfiguration } from "@/types/experiment";

interface UseResultsVisualizationProps {
  config: ExperimentConfiguration;
  runId: number;
}

interface TrainingHistoryData {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  valPrecision?: number;
  valRecall?: number;
  valF1?: number;
  valAuroc?: number;
}

interface ConfusionMatrixData {
  name: string;
  value: number;
}

interface ConfusionMatrix2D {
  [0]: [number, number];
  [1]: [number, number];
}

interface MetricsChartData {
  name: string;
  value: number;
}

interface MetricsBarData {
  metric: string;
  value: number;
}

interface ComparisonBarData {
  metric: string;
  centralized: number;
  federated: number;
}

interface ComparisonMetricsData {
  name: string;
  centralized: number;
  federated: number;
  difference: string;
}

interface ServerEvaluationData {
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
  summary: any;
}

export const useResultsVisualization = ({
  config,
  runId,
}: UseResultsVisualizationProps) => {
  const [activeTab, setActiveTab] = useState("metrics");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [centralizedResults, setCentralizedResults] =
    useState<ExperimentResults | null>(null);
  const [federatedResults, setFederatedResults] =
    useState<ExperimentResults | null>(null);
  const [comparisonData, setComparisonData] =
    useState<ComparisonResults | null>(null);
  const [serverEvaluation, setServerEvaluation] =
    useState<ServerEvaluationData | null>(null);

  const showComparison = config.trainingMode === "both";

  // Fetch results on mount
  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      setError(null);

      try {
        const results = await api.results.getRunMetrics(runId);

        if (config.trainingMode === "centralized") {
          setCentralizedResults(results);
        } else if (config.trainingMode === "federated") {
          setFederatedResults(results);

          // Fetch server evaluation data for federated runs
          try {
            const serverEval = await api.results.getServerEvaluation(runId);
            if (serverEval.has_server_evaluation) {
              setServerEvaluation(serverEval);
              console.log(
                "[useResultsVisualization] Server evaluation loaded:",
                serverEval,
              );
            }
          } catch (serverEvalErr: any) {
            console.warn(
              "[useResultsVisualization] Failed to fetch server evaluation:",
              serverEvalErr,
            );
            // Non-critical error, continue without server evaluation
          }
        } else if (config.trainingMode === "both") {
          setCentralizedResults(results);
          console.warn(
            "Comparison mode requires separate run_ids for centralized and federated",
          );
        }
      } catch (err: any) {
        console.error("Error fetching results:", err);
        setError(err.message || "Failed to load results");
        toast.error("Failed to load results");
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [runId, config.trainingMode]);

  // Get active results based on training mode
  const activeResults = useMemo(() => {
    if (config.trainingMode === "centralized") return centralizedResults;
    if (config.trainingMode === "federated") return federatedResults;
    return centralizedResults || federatedResults;
  }, [config.trainingMode, centralizedResults, federatedResults]);

  // Transform training history data
  const trainingHistoryData = useMemo((): TrainingHistoryData[] => {
    if (!activeResults?.training_history) return [];

    return activeResults.training_history.map(
      (entry: ResultsTrainingHistoryEntry) => ({
        epoch: entry.epoch,
        trainLoss: entry.train_loss,
        valLoss: entry.val_loss,
        trainAcc: entry.train_acc,
        valAcc: entry.val_acc,
        valPrecision: (entry as any).val_precision,
        valRecall: (entry as any).val_recall,
        valF1: (entry as any).val_f1,
        valAuroc: (entry as any).val_auroc,
      }),
    );
  }, [activeResults]);

  // Transform confusion matrix data for pie chart
  const confusionMatrixData = useMemo((): ConfusionMatrixData[] => {
    if (!activeResults?.confusion_matrix) return [];

    const cm = activeResults.confusion_matrix;
    return [
      { name: "True Positives", value: cm.true_positives },
      { name: "True Negatives", value: cm.true_negatives },
      { name: "False Positives", value: cm.false_positives },
      { name: "False Negatives", value: cm.false_negatives },
    ];
  }, [activeResults]);

  // Transform confusion matrix data for grid display
  const confusionMatrix = useMemo((): ConfusionMatrix2D | null => {
    if (!activeResults?.confusion_matrix) return null;

    const cm = activeResults.confusion_matrix;
    return [
      [cm.true_negatives, cm.false_positives],
      [cm.false_negatives, cm.true_positives],
    ] as ConfusionMatrix2D;
  }, [activeResults]);

  // Transform metrics for chart (used by primary metrics display)
  // Uses BEST validation metrics from metadata for consistency with cards display
  const metricsChartData = useMemo((): MetricsChartData[] => {
    if (!activeResults?.metadata) {
      console.warn(
        "[useResultsVisualization] No metadata in activeResults:",
        activeResults,
      );
      return [];
    }

    const metadata = activeResults.metadata;
    console.log("[useResultsVisualization] metricsChartData metadata:", {
      best_val_accuracy: metadata.best_val_accuracy,
      best_val_precision: metadata.best_val_precision,
      best_val_recall: metadata.best_val_recall,
      best_val_f1: metadata.best_val_f1,
      best_val_auroc: metadata.best_val_auroc,
    });

    return [
      { name: "Accuracy", value: metadata.best_val_accuracy || 0 },
      { name: "Precision", value: metadata.best_val_precision || 0 },
      { name: "Recall", value: metadata.best_val_recall || 0 },
      { name: "F1-Score", value: metadata.best_val_f1 || 0 },
      { name: "AUC", value: metadata.best_val_auroc || 0 },
    ];
  }, [activeResults]);

  // Transform metrics for bar chart
  const metricsBarData = useMemo((): MetricsBarData[] => {
    if (!activeResults?.final_metrics) return [];

    const metrics = activeResults.final_metrics;
    return [
      { metric: "Accuracy", value: metrics.accuracy },
      { metric: "Precision", value: metrics.precision },
      { metric: "Recall", value: metrics.recall },
      { metric: "F1 Score", value: metrics.f1_score },
      { metric: "AUC", value: metrics.auc },
    ];
  }, [activeResults]);

  // Transform comparison data for bar chart
  const comparisonBarData = useMemo((): ComparisonBarData[] => {
    if (!showComparison || !centralizedResults || !federatedResults) return [];

    const cMetrics = centralizedResults.final_metrics;
    const fMetrics = federatedResults.final_metrics;

    return [
      {
        metric: "Accuracy",
        centralized: cMetrics.accuracy,
        federated: fMetrics.accuracy,
      },
      {
        metric: "Precision",
        centralized: cMetrics.precision,
        federated: fMetrics.precision,
      },
      {
        metric: "Recall",
        centralized: cMetrics.recall,
        federated: fMetrics.recall,
      },
      {
        metric: "F1 Score",
        centralized: cMetrics.f1_score,
        federated: fMetrics.f1_score,
      },
      { metric: "AUC", centralized: cMetrics.auc, federated: fMetrics.auc },
    ];
  }, [showComparison, centralizedResults, federatedResults]);

  // Transform comparison metrics for table display
  const comparisonMetricsData = useMemo((): ComparisonMetricsData[] => {
    if (!comparisonData) return [];

    return Object.entries(comparisonData.comparison_metrics).map(
      ([metric, data]) => ({
        name:
          metric.charAt(0).toUpperCase() + metric.slice(1).replace("_", " "),
        centralized: data.centralized,
        federated: data.federated,
        difference: data.difference.toFixed(3),
      }),
    );
  }, [comparisonData]);

  // Centralized-specific data transformations
  // Uses BEST validation metrics from metadata for consistency
  const centralizedMetricsData = useMemo((): MetricsChartData[] => {
    if (!centralizedResults?.metadata) return [];

    const metadata = centralizedResults.metadata;
    return [
      { name: "Accuracy", value: metadata.best_val_accuracy || 0 },
      { name: "Precision", value: metadata.best_val_precision || 0 },
      { name: "Recall", value: metadata.best_val_recall || 0 },
      { name: "F1-Score", value: metadata.best_val_f1 || 0 },
      { name: "AUC", value: metadata.best_val_auroc || 0 },
    ];
  }, [centralizedResults]);

  const centralizedHistoryData = useMemo((): TrainingHistoryData[] => {
    if (!centralizedResults?.training_history) return [];

    return centralizedResults.training_history.map(
      (entry: ResultsTrainingHistoryEntry) => ({
        epoch: entry.epoch,
        trainLoss: entry.train_loss,
        valLoss: entry.val_loss,
        trainAcc: entry.train_acc,
        valAcc: entry.val_acc,
        valPrecision: (entry as any).val_precision,
        valRecall: (entry as any).val_recall,
        valF1: (entry as any).val_f1,
        valAuroc: (entry as any).val_auroc,
      }),
    );
  }, [centralizedResults]);

  const centralizedConfusionMatrix = useMemo((): ConfusionMatrix2D | null => {
    if (!centralizedResults?.confusion_matrix) return null;

    const cm = centralizedResults.confusion_matrix;
    return [
      [cm.true_negatives, cm.false_positives],
      [cm.false_negatives, cm.true_positives],
    ] as ConfusionMatrix2D;
  }, [centralizedResults]);

  // Federated-specific data transformations
  const federatedMetricsData = useMemo((): MetricsChartData[] => {
    if (!federatedResults?.final_metrics) return [];

    const metrics = federatedResults.final_metrics;
    return [
      { name: "Accuracy", value: metrics.accuracy },
      { name: "Precision", value: metrics.precision },
      { name: "Recall", value: metrics.recall },
      { name: "F1-Score", value: metrics.f1_score },
      { name: "AUC", value: metrics.auc },
    ];
  }, [federatedResults]);

  const federatedHistoryData = useMemo((): TrainingHistoryData[] => {
    if (!federatedResults?.training_history) return [];

    return federatedResults.training_history.map(
      (entry: ResultsTrainingHistoryEntry) => ({
        epoch: entry.epoch,
        trainLoss: entry.train_loss,
        valLoss: entry.val_loss,
        trainAcc: entry.train_acc,
        valAcc: entry.val_acc,
        valPrecision: (entry as any).val_precision,
        valRecall: (entry as any).val_recall,
        valF1: (entry as any).val_f1,
        valAuroc: (entry as any).val_auroc,
      }),
    );
  }, [federatedResults]);

  const federatedConfusionMatrix = useMemo((): ConfusionMatrix2D | null => {
    if (!federatedResults?.confusion_matrix) return null;

    const cm = federatedResults.confusion_matrix;
    return [
      [cm.true_negatives, cm.false_positives],
      [cm.false_negatives, cm.true_positives],
    ] as ConfusionMatrix2D;
  }, [federatedResults]);

  // Get confusion matrix cell color
  const getConfusionMatrixColor = (name: string): string => {
    const colorMap: Record<string, string> = {
      "True Positives": "#0A9396",
      "True Negatives": "#94D2BD",
      "False Positives": "#E9C46A",
      "False Negatives": "#E76F51",
    };
    return colorMap[name] || "#888";
  };

  // Handle download
  const handleDownload = async (format: "json" | "csv" | "summary") => {
    try {
      const formatLabels: Record<string, string> = {
        json: "Metrics JSON",
        csv: "Metrics CSV",
        summary: "Summary Report",
      };

      await api.results.triggerRunDownload(runId, format);
      toast.success(`${formatLabels[format]} download started`);
    } catch (err: any) {
      console.error("Download error:", err);
      toast.error(`Failed to download ${format}`);
    }
  };

  // Server evaluation data transformations
  const serverEvaluationChartData = useMemo(() => {
    if (!serverEvaluation?.evaluations) return [];

    return serverEvaluation.evaluations.map((evaluation) => ({
      round: evaluation.round,
      loss: evaluation.loss,
      accuracy: evaluation.accuracy,
      precision: evaluation.precision,
      recall: evaluation.recall,
      f1_score: evaluation.f1_score,
      auroc: evaluation.auroc,
    }));
  }, [serverEvaluation]);

  const serverEvaluationLatestMetrics = useMemo(() => {
    if (
      !serverEvaluation?.evaluations ||
      serverEvaluation.evaluations.length === 0
    )
      return null;

    const latest =
      serverEvaluation.evaluations[serverEvaluation.evaluations.length - 1];
    return [
      { name: "Accuracy", value: latest.accuracy || 0 },
      { name: "Precision", value: latest.precision || 0 },
      { name: "Recall", value: latest.recall || 0 },
      { name: "F1-Score", value: latest.f1_score || 0 },
      { name: "AUC", value: latest.auroc || 0 },
    ];
  }, [serverEvaluation]);

  const serverEvaluationConfusionMatrix = useMemo(() => {
    if (
      !serverEvaluation?.evaluations ||
      serverEvaluation.evaluations.length === 0
    )
      return null;

    const latest =
      serverEvaluation.evaluations[serverEvaluation.evaluations.length - 1];
    if (!latest.confusion_matrix) return null;

    const cm = latest.confusion_matrix;
    return [
      [cm.true_negatives, cm.false_positives],
      [cm.false_negatives, cm.true_positives],
    ] as ConfusionMatrix2D;
  }, [serverEvaluation]);

  return {
    // State
    activeTab,
    setActiveTab,
    loading,
    error,

    // Raw data
    centralizedResults,
    federatedResults,
    comparisonData,
    activeResults,
    showComparison,

    // Active results transformed data
    trainingHistoryData,
    confusionMatrixData,
    confusionMatrix,
    metricsChartData,
    metricsBarData,

    // Comparison data
    comparisonBarData,
    comparisonMetricsData,

    // Centralized-specific data
    centralizedMetricsData,
    centralizedHistoryData,
    centralizedConfusionMatrix,

    // Federated-specific data
    federatedMetricsData,
    federatedHistoryData,
    federatedConfusionMatrix,

    // Server evaluation data
    serverEvaluation,
    serverEvaluationChartData,
    serverEvaluationLatestMetrics,
    serverEvaluationConfusionMatrix,

    // Utilities
    getConfusionMatrixColor,
    handleDownload,
  };
};
