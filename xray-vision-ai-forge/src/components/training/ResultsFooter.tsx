import React from "react";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";

interface ResultsFooterProps {
  onReset: () => void;
  handleDownload: (format: string) => void;
}

const ResultsFooter: React.FC<ResultsFooterProps> = ({
  onReset,
}) => {
  return (
    <div className="px-6 py-5 md:px-8 md:py-6 bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] border-t border-[hsl(172_63%_15%)]">
      <div className="flex items-center justify-between gap-4">
        {/* Helper Text */}
        <p className="text-sm text-white/70 hidden sm:block">
          Ready to run another experiment?
        </p>

        {/* Start New Button */}
        <Button
          onClick={onReset}
          className="group relative overflow-hidden bg-white hover:bg-[hsl(168_25%_98%)] text-[hsl(172_63%_22%)] font-semibold text-sm px-6 py-5 h-auto rounded-xl shadow-lg shadow-black/10 transition-all duration-300 hover:shadow-xl hover:shadow-black/15 hover:-translate-y-0.5 ml-auto"
        >
          <span className="relative flex items-center gap-2">
            <RotateCcw className="h-4 w-4 transition-transform group-hover:-rotate-180 duration-500" />
            <span>Start New Experiment</span>
          </span>
        </Button>
      </div>
    </div>
  );
};

export default ResultsFooter;
