import React from "react";
import { Button } from "@/components/ui/button";
import { BarChart, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { RunContext } from "./types";

interface RunContextBadgeProps {
  selectedRun: RunContext;
  onClear: () => void;
}

export const RunContextBadge: React.FC<RunContextBadgeProps> = ({
  selectedRun,
  onClear,
}) => {
  return (
    <div className="px-4 pt-4 pb-2">
      <div className="bg-[hsl(172_40%_95%)] border border-[hsl(172_30%_88%)] rounded-2xl p-4 flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center flex-shrink-0">
          <BarChart className="h-5 w-5 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold text-[hsl(172_43%_20%)]">
                Run #{selectedRun.runId}
              </p>
              <span
                className={cn(
                  "text-[10px] font-medium px-2 py-0.5 rounded-full",
                  selectedRun.status === "completed"
                    ? "bg-green-100 text-green-700"
                    : "bg-amber-100 text-amber-700",
                )}
              >
                {selectedRun.status === "completed"
                  ? "Completed"
                  : "In Progress"}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              aria-label="Clear run selection"
              className="h-6 w-6 p-0 rounded-lg hover:bg-[hsl(172_30%_88%)] text-[hsl(215_15%_45%)]"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="flex items-center gap-2 mb-2">
            <p className="text-xs text-[hsl(215_15%_50%)]">
              {selectedRun.trainingMode} training
            </p>
            {selectedRun.bestRecall !== undefined &&
              selectedRun.bestRecall > 0 && (
                <span className="text-xs font-medium text-[hsl(152_60%_35%)]">
                  Best Recall: {(selectedRun.bestRecall * 100).toFixed(2)}%
                </span>
              )}
          </div>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-end gap-0.5 h-4">
              {[40, 65, 55, 75, 80, 70].map((height, i) => (
                <div
                  key={i}
                  className="w-1 rounded-sm bg-[hsl(172_63%_35%)]"
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
            <button
              onClick={() => console.log("View Report clicked")}
              className="text-[10px] font-medium text-[hsl(172_63%_22%)] hover:underline"
            >
              View Report →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
