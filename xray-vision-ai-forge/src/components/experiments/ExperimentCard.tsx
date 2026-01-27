import { memo } from "react";
import { DetailedCard } from "./DetailedCard";
import { ConciseCard } from "./ConciseCard";
import { CompactCard } from "./CompactCard";
import { ViewMode } from "./useExperiments";
import { RunSummary } from "@/types/runs";

interface ExperimentCardProps {
  run: RunSummary;
  viewMode: ViewMode;
  index: number;
  onViewResults: (id: number) => void;
}

const viewModePadding = {
  detailed: "p-3",
  concise: "p-2.5",
  compact: "p-2",
};

const viewModeGridCols = {
  detailed: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
  concise: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
  compact: "grid-cols-1 md:grid-cols-3 lg:grid-cols-5",
};

export const ExperimentCard = memo(({ run, viewMode, index, onViewResults }: ExperimentCardProps) => {
  const isFederated = run.training_mode === "federated";

  return (
    <div
      key={`${run.id}-${viewMode}`}
      className={`group relative bg-white rounded-xl border border-[hsl(210_15%_92%)] hover:border-[hsl(172_63%_22%)]/30 hover:shadow-2xl hover:shadow-[hsl(172_40%_85%)]/20 transition-all duration-500 hover:-translate-y-1 flex flex-col h-full overflow-hidden ${viewModePadding[viewMode]}`}
      style={{
        animation: "fadeIn 0.4s ease-out forwards, scaleIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) forwards",
        animationDelay: `${index * 0.04}s`,
        opacity: 0,
        transformOrigin: "center",
      }}
    >
      {/* Gradient accent bar */}
      <div
        className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${
          isFederated
            ? "from-[hsl(210_60%_40%)] via-[hsl(210_60%_40%)]/50 to-transparent"
            : "from-[hsl(152_60%_35%)] via-[hsl(152_60%_35%)]/50 to-transparent"
        }`}
      />

      {viewMode === "detailed" && <DetailedCard run={run} onViewResults={onViewResults} />}
      {viewMode === "concise" && <ConciseCard run={run} onViewResults={onViewResults} />}
      {viewMode === "compact" && <CompactCard run={run} onViewResults={onViewResults} />}
    </div>
  );
});

export { viewModeGridCols };
ExperimentCard.displayName = "ExperimentCard";
