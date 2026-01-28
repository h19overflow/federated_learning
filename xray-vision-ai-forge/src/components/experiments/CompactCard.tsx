import { memo } from "react";
import { Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TrainingModeBadge } from "./TrainingModeBadge";
import { formatShortDate } from "./utils";
import { RunSummary } from "@/types/runs";

interface CompactCardProps {
  run: RunSummary;
  onViewResults: (id: number) => void;
}

export const CompactCard = memo(({ run, onViewResults }: CompactCardProps) => {
  return (
    <>
      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-[hsl(172_43%_15%)] text-xs truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300 mb-1">
          {run.run_description || `Run #${run.id}`}
        </h3>
        <div className="flex items-center gap-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300 mb-1.5">
          <Calendar className="h-2.5 w-2.5 flex-shrink-0" />
          <span className="text-[8px]">{formatShortDate(run.start_time)}</span>
        </div>
        <TrainingModeBadge isFederated={run.training_mode === "federated"} compact />
      </div>

      <Button
        onClick={() => onViewResults(run.id)}
        className="w-full bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-2 py-1 text-[10px] shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.02] font-medium mt-2"
      >
        View
      </Button>
    </>
  );
});

CompactCard.displayName = "CompactCard";
