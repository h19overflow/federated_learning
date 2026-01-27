import { memo } from "react";
import { Calendar, Zap, Users, CheckCircle2, BarChart, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MetricIndicator } from "./MetricIndicator";
import { SummaryStatisticsPreview } from "./SummaryStatisticsPreview";
import { TrainingModeBadge } from "./TrainingModeBadge";
import {
  formatDate,
  shouldShowCentralizedMetrics,
  shouldShowFederatedEvalMetrics,
  hasValidAccuracy,
  hasValidRecall,
  hasNoMetrics,
} from "./utils";
import { RunSummary } from "@/types/runs";

interface DetailedCardProps {
  run: RunSummary;
  onViewResults: (id: number) => void;
}

export const DetailedCard = memo(({ run, onViewResults }: DetailedCardProps) => {
  const isFederated = run.training_mode === "federated";
  const federatedInfo = run.federated_info;

  return (
    <>
      {/* Card Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-[hsl(172_43%_15%)] text-base truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300">
            {run.run_description || `Run #${run.id}`}
          </h3>
          <div className="flex items-center gap-1.5 mt-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300">
            <Calendar className="h-3 w-3 flex-shrink-0" />
            <span className="text-[10px]">{formatDate(run.start_time)}</span>
          </div>
        </div>
        <TrainingModeBadge isFederated={isFederated} />
      </div>

      {/* Metrics Section */}
      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_15%_97%)] rounded-lg p-2.5 mb-2.5 border border-[hsl(168_20%_94%)] flex-grow transition-all duration-300 group-hover:border-[hsl(168_20%_88%)]">
        {/* Centralized Metrics */}
        {!isFederated && shouldShowCentralizedMetrics(run) && (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <MetricIndicator label="Best Val Accuracy" value={run.best_val_accuracy * 100} />
              <MetricIndicator label="Best Val Recall" value={run.best_val_recall * 100} variant="secondary" />
            </div>
            <SummaryStatisticsPreview stats={run.final_epoch_stats} />
          </div>
        )}

        {/* Federated Metrics */}
        {isFederated && federatedInfo && (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gradient-to-br from-[hsl(210_60%_40%)]/5 to-[hsl(210_60%_40%)]/0 rounded-lg p-2 border border-[hsl(210_60%_40%)]/20 backdrop-blur-sm transition-all duration-300 hover:shadow-md">
                <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">Rounds</p>
                <div className="flex items-baseline gap-0.5">
                  <p className="text-base font-semibold text-[hsl(172_43%_20%)]">{federatedInfo.num_rounds}</p>
                  <Zap className="h-3 w-3 text-[hsl(210_60%_40%)]" />
                </div>
              </div>
              <div className="bg-gradient-to-br from-[hsl(152_60%_35%)]/5 to-[hsl(152_60%_35%)]/0 rounded-lg p-2 border border-[hsl(152_60%_35%)]/20 backdrop-blur-sm transition-all duration-300 hover:shadow-md">
                <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">Clients</p>
                <div className="flex items-baseline gap-0.5">
                  <p className="text-base font-semibold text-[hsl(172_43%_20%)]">{federatedInfo.num_clients}</p>
                  <Users className="h-3 w-3 text-[hsl(152_60%_35%)]" />
                </div>
              </div>
            </div>

            {shouldShowFederatedEvalMetrics(federatedInfo) && (
              <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
                <p className="text-[10px] text-[hsl(215_15%_55%)] mb-1.5 uppercase tracking-wide font-medium">
                  Server Evaluation
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {hasValidAccuracy(federatedInfo.best_accuracy) && (
                    <MetricIndicator label="Best Val Accuracy" value={(federatedInfo.best_accuracy ?? 0) * 100} />
                  )}
                  {hasValidRecall(federatedInfo.best_recall) && (
                    <MetricIndicator label="Best Val Recall" value={(federatedInfo.best_recall ?? 0) * 100} variant="secondary" />
                  )}
                </div>
              </div>
            )}

            {!federatedInfo.has_server_evaluation && (
              <div className="flex items-center gap-1.5 pt-2 border-t border-[hsl(168_20%_92%)]">
                <div className="w-1 h-1 rounded-full bg-[hsl(35_70%_50%)]" />
                <p className="text-[10px] text-[hsl(35_70%_45%)] font-medium">No server evaluation data</p>
              </div>
            )}
            <SummaryStatisticsPreview stats={run.final_epoch_stats} />
          </div>
        )}

        {!isFederated && hasNoMetrics(run) && (
          <div className="text-center py-2">
            <p className="text-sm text-[hsl(215_15%_55%)]">No metrics available</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-auto pt-2.5 border-t border-[hsl(210_15%_92%)] group-hover:border-[hsl(172_63%_22%)]/20 transition-colors duration-300">
        <div className="flex items-center gap-1">
          <CheckCircle2 className="h-3 w-3 text-[hsl(152_60%_35%)]" />
          <span className="text-[10px] text-[hsl(215_15%_55%)] font-medium">{run.metrics_count} metrics</span>
        </div>
        <Button
          onClick={() => onViewResults(run.id)}
          className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-3 py-1.5 text-xs shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.03] font-medium flex items-center gap-1"
        >
          <BarChart className="h-3.5 w-3.5" />
          View Results
          <ChevronRight className="h-3.5 w-3.5 opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
        </Button>
      </div>
    </>
  );
});

DetailedCard.displayName = "DetailedCard";
