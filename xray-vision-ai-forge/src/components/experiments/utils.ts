import { RunSummary } from "@/types/runs";

export const formatDate = (dateString: string | null): string => {
  if (!dateString) return "Unknown date";
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateString;
  }
};

export const formatShortDate = (dateString: string | null): string => {
  const formatted = formatDate(dateString);
  return formatted.split(",")[0];
};

export const shouldShowCentralizedMetrics = (run: RunSummary): boolean => {
  return run.best_val_recall > 0 || run.best_val_accuracy > 0;
};

export const shouldShowFederatedEvalMetrics = (
  federatedInfo?: RunSummary["federated_info"]
): boolean => {
  if (!federatedInfo?.has_server_evaluation) return false;
  return federatedInfo.best_accuracy !== null || federatedInfo.best_recall !== null;
};

export const hasValidAccuracy = (value: number | null | undefined): boolean => {
  return value !== null && value !== undefined;
};

export const hasValidRecall = (value: number | null | undefined): boolean => {
  return value !== null && value !== undefined;
};

export const hasNoMetrics = (run: RunSummary): boolean => {
  return run.best_val_recall === 0 && run.best_val_accuracy === 0;
};
