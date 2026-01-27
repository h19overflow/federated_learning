/**
 * Container panel for real-time training observability.
 * Shows accuracy and F1 metrics with summary cards and chart.
 */

import React, { memo } from "react";
import { Activity, Percent, Zap, CheckCircle, XCircle } from "lucide-react";
import BatchMetricsChart from "./BatchMetricsChart";
import type { BatchMetricsDataPoint } from "@/types/api";
import type { ConfusionMatrixData } from "@/hooks/useTrainingMetrics";

interface TrainingObservabilityPanelProps {
  batchMetrics: BatchMetricsDataPoint[];
  currentLoss: number | null;
  currentAccuracy: number | null;
  currentF1: number | null;
  confusionMatrix: ConfusionMatrixData | null;
  isReceiving: boolean;
  trainingMode: "centralized" | "federated";
}

const StatCard = memo(function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3 bg-white rounded-xl px-4 py-3 border border-[hsl(168_20%_88%)] shadow-sm">
      <div className={`p-2 rounded-lg ${color}`}>
        <Icon className="h-4 w-4 text-white" />
      </div>
      <div>
        <p className="text-xs text-[hsl(215_15%_50%)]">{label}</p>
        <p className="text-sm font-semibold text-[hsl(215_20%_25%)]">{value}</p>
      </div>
    </div>
  );
});

const ConfusionMatrixCard = memo(function ConfusionMatrixCard({
  confusionMatrix,
}: {
  confusionMatrix: ConfusionMatrixData;
}) {
  // Calculate total and percentages
  const total = confusionMatrix.tp + confusionMatrix.tn + confusionMatrix.fp + confusionMatrix.fn;
  const tpPercent = ((confusionMatrix.tp / total) * 100).toFixed(1);
  const tnPercent = ((confusionMatrix.tn / total) * 100).toFixed(1);
  const fpPercent = ((confusionMatrix.fp / total) * 100).toFixed(1);
  const fnPercent = ((confusionMatrix.fn / total) * 100).toFixed(1);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-[hsl(215_15%_50%)]">
          Confusion Matrix (Epoch {confusionMatrix.epoch})
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {/* True Positives */}
        <div className="bg-green-50 rounded-lg p-3 border border-green-100">
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <span className="text-xs font-semibold text-green-700">TP</span>
          </div>
          <p className="text-lg font-bold text-green-800">{formatNumber(confusionMatrix.tp)}</p>
          <p className="text-xs text-green-600">{tpPercent}%</p>
        </div>
        {/* True Negatives */}
        <div className="bg-blue-50 rounded-lg p-3 border border-blue-100">
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle className="h-4 w-4 text-blue-600" />
            <span className="text-xs font-semibold text-blue-700">TN</span>
          </div>
          <p className="text-lg font-bold text-blue-800">{formatNumber(confusionMatrix.tn)}</p>
          <p className="text-xs text-blue-600">{tnPercent}%</p>
        </div>
        {/* False Positives */}
        <div className="bg-red-50 rounded-lg p-3 border border-red-100">
          <div className="flex items-center gap-2 mb-1">
            <XCircle className="h-4 w-4 text-red-600" />
            <span className="text-xs font-semibold text-red-700">FP</span>
          </div>
          <p className="text-lg font-bold text-red-800">{formatNumber(confusionMatrix.fp)}</p>
          <p className="text-xs text-red-600">{fpPercent}%</p>
        </div>
        {/* False Negatives */}
        <div className="bg-orange-50 rounded-lg p-3 border border-orange-100">
          <div className="flex items-center gap-2 mb-1">
            <XCircle className="h-4 w-4 text-orange-600" />
            <span className="text-xs font-semibold text-orange-700">FN</span>
          </div>
          <p className="text-lg font-bold text-orange-800">{formatNumber(confusionMatrix.fn)}</p>
          <p className="text-xs text-orange-600">{fnPercent}%</p>
        </div>
      </div>
    </div>
  );
});

const TrainingObservabilityPanel = memo(function TrainingObservabilityPanel({
  batchMetrics,
  currentLoss,
  currentAccuracy,
  currentF1,
  confusionMatrix,
  isReceiving,
  trainingMode,
}: TrainingObservabilityPanelProps) {
  const formatPercent = (val: number | null) =>
    val !== null ? `${(val * 100).toFixed(1)}%` : "—";
  const formatLoss = (val: number | null) =>
    val !== null ? val.toFixed(4) : "—";
  const formatNumber = (val: number) => val.toLocaleString();

  return (
    <div className="space-y-4 p-4 bg-[hsl(168_25%_97%)] rounded-2xl border border-[hsl(168_20%_90%)]">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-[hsl(172_63%_28%)]" />
          <h3 className="text-lg font-semibold text-[hsl(215_20%_25%)]">
            Live Metrics
          </h3>
          {isReceiving && (
            <span className="flex items-center gap-1 text-xs text-[hsl(172_63%_35%)]">
              <span className="w-2 h-2 rounded-full bg-[hsl(172_63%_35%)] animate-pulse" />
              Receiving
            </span>
          )}
        </div>
        <span className="text-xs px-2 py-1 rounded-full bg-[hsl(210_60%_95%)] text-[hsl(210_60%_40%)]">
          {trainingMode === "federated" ? "Federated" : "Centralized"}
        </span>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard
          label="Loss"
          value={formatLoss(currentLoss)}
          icon={Activity}
          color="bg-[hsl(0_65%_55%)]"
        />
        <StatCard
          label="Accuracy"
          value={formatPercent(currentAccuracy)}
          icon={Percent}
          color="bg-[hsl(172_63%_35%)]"
        />
        <StatCard
          label="F1 Score"
          value={formatPercent(currentF1)}
          icon={Zap}
          color="bg-[hsl(210_60%_50%)]"
        />
      </div>

      {/* Confusion Matrix */}
      {confusionMatrix && (
        <div className="bg-white rounded-xl px-4 py-3 border border-[hsl(168_20%_88%)] shadow-sm">
          <ConfusionMatrixCard confusionMatrix={confusionMatrix} />
        </div>
      )}

      {/* Chart */}
      <BatchMetricsChart data={batchMetrics} height={250} />
    </div>
  );
});

export default TrainingObservabilityPanel;
