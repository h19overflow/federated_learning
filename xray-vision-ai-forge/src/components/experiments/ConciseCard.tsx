import { memo } from "react";
import { Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TrainingModeBadge } from "./TrainingModeBadge";
import { formatDate } from "./utils";
import { RunSummary } from "@/types/runs";

interface ConciseCardProps {
  run: RunSummary;
  onViewResults: (id: number) => void;
}

export const ConciseCard = memo(({ run, onViewResults }: ConciseCardProps) => {
  const isFederated = run.training_mode === "federated";
  const federatedInfo = run.federated_info;
  const bestAccuracy = isFederated ? federatedInfo?.best_accuracy ?? 0 : run.best_val_accuracy;

  return (
    <>
      <div className="flex-1">
        <h3 className="font-semibold text-[hsl(172_43%_15%)] text-sm truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300 mb-1.5">
          {run.run_description || `Run #${run.id}`}
        </h3>
        <div className="flex items-center gap-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300 mb-2">
          <Calendar className="h-3 w-3 flex-shrink-0" />
          <span className="text-[9px]">{formatDate(run.start_time)}</span>
        </div>

        <TrainingModeBadge isFederated={isFederated} />

        {/* Key Metric */}
        <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_15%_97%)] rounded-lg p-2 border border-[hsl(168_20%_94%)] transition-all duration-300 group-hover:border-[hsl(168_20%_88%)] mt-2">
          <p className="text-[8px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">Best Accuracy</p>
          <div className="flex items-baseline gap-0.5">
            <p className="text-sm font-semibold text-[hsl(172_43%_20%)]">{(bestAccuracy * 100).toFixed(1)}</p>
            <p className="text-[9px] text-[hsl(215_15%_50%)]">%</p>
          </div>
        </div>
      </div>

      <Button
        onClick={() => onViewResults(run.id)}
        className="w-full bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-2 py-1.5 text-xs shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.02] font-medium mt-2"
      >
        View Results
      </Button>
    </>
  );
});

ConciseCard.displayName = "ConciseCard";
