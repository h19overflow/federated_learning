/**
 * Container panel for real-time training observability.
 * Shows accuracy and F1 metrics with summary cards and chart.
 */

import React, { memo } from "react";
import { Activity, Percent, Zap } from "lucide-react";
import BatchMetricsChart from "./BatchMetricsChart";
import type { BatchMetricsDataPoint } from "@/types/api";

interface TrainingObservabilityPanelProps {
  batchMetrics: BatchMetricsDataPoint[];
  currentLoss: number | null;
  currentAccuracy: number | null;
  currentF1: number | null;
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

const TrainingObservabilityPanel = memo(function TrainingObservabilityPanel({
  batchMetrics,
  currentLoss,
  currentAccuracy,
  currentF1,
  isReceiving,
  trainingMode,
}: TrainingObservabilityPanelProps) {
  const formatPercent = (val: number | null) =>
    val !== null ? `${(val * 100).toFixed(1)}%` : "—";
  const formatLoss = (val: number | null) =>
    val !== null ? val.toFixed(4) : "—";

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

      {/* Chart */}
      <BatchMetricsChart data={batchMetrics} height={250} />
    </div>
  );
});

export default TrainingObservabilityPanel;
