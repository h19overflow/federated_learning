import { memo } from "react";
import { MetricIndicator } from "./MetricIndicator";
import { RunSummary } from "@/types/runs";

interface SummaryStatisticsPreviewProps {
  stats?: RunSummary["final_epoch_stats"];
}

export const SummaryStatisticsPreview = memo(({ stats }: SummaryStatisticsPreviewProps) => {
  if (!stats) return null;

  return (
    <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
      <p className="text-[10px] text-[hsl(215_15%_55%)] mb-1.5 uppercase tracking-wide font-medium">
        Final Epoch Statistics
      </p>
      <div className="grid grid-cols-2 gap-1.5 text-xs">
        <MetricIndicator label="Recall" value={stats.sensitivity * 100} />
        <MetricIndicator label="Specificity" value={stats.specificity * 100} variant="secondary" />
        <MetricIndicator label="Precision" value={stats.precision_cm * 100} variant="accent" />
        <MetricIndicator label="F1 Score" value={stats.f1_cm * 100} />
      </div>
    </div>
  );
});

SummaryStatisticsPreview.displayName = "SummaryStatisticsPreview";
