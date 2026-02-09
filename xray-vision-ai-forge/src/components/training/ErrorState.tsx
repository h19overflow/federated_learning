import React from "react";
import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorStateProps {
  error: string;
  onReset: () => void;
}

const ErrorState: React.FC<ErrorStateProps> = ({ error, onReset }) => {
  return (
    <div className="space-y-8" style={{ animation: "fadeIn 0.5s ease-out" }}>
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(0_50%_90%)] shadow-lg p-12">
          <div className="flex flex-col items-center justify-center space-y-5">
            <div className="p-4 rounded-2xl bg-[hsl(0_60%_95%)]">
              <AlertCircle className="h-10 w-10 text-[hsl(0_72%_51%)]" />
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold text-[hsl(0_72%_40%)]">
                Failed to Load Results
              </p>
              <p className="text-[hsl(215_15%_50%)] mt-1">{error}</p>
            </div>
            <Button
              onClick={onReset}
              variant="outline"
              className="rounded-xl border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
            >
              Start New Experiment
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ErrorState;
