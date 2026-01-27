import { Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  onCreate: () => void;
}

export const EmptyState = ({ onCreate }: EmptyStateProps) => (
  <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
    <div className="relative w-24 h-24 mb-8">
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-[hsl(172_63%_22%)]/10 to-[hsl(168_25%_96%)]/50 blur-xl" />
      <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(168_25%_96%)] to-[hsl(210_15%_95%)] flex items-center justify-center border border-[hsl(168_20%_90%)] shadow-lg">
        <svg className="w-12 h-12 text-[hsl(172_63%_35%)]" viewBox="0 0 40 40" fill="none">
          <rect x="6" y="10" width="28" height="24" rx="3" stroke="currentColor" strokeWidth="2" />
          <path d="M6 16h28" stroke="currentColor" strokeWidth="2" />
          <circle cx="11" cy="13" r="1.5" fill="currentColor" />
          <circle cx="16" cy="13" r="1.5" fill="currentColor" />
          <path d="M14 26l5-5 4 4 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </div>
    <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-3">No experiments yet</h3>
    <p className="text-[hsl(215_15%_50%)] mb-10 max-w-sm text-center leading-relaxed">
      Start your first training run to see it appear here. Create a new experiment to begin.
    </p>
    <Button
      onClick={onCreate}
      className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-xl px-8 py-3 text-base shadow-lg shadow-[hsl(172_63%_22%)]/25 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/35 transition-all duration-300 hover:-translate-y-1 font-semibold flex items-center gap-2"
    >
      <Zap className="h-5 w-5" />
      Create New Experiment
    </Button>
  </div>
);
