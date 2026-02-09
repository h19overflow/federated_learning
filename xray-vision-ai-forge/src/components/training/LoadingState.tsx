import React from "react";
import { Loader2 } from "lucide-react";

interface LoadingStateProps {}

const LoadingState: React.FC<LoadingStateProps> = () => {
  return (
    <div className="space-y-8" style={{ animation: "fadeIn 0.5s ease-out" }}>
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 p-12">
          <div className="flex flex-col items-center justify-center space-y-5">
            <div className="p-4 rounded-2xl bg-[hsl(172_40%_94%)]">
              <Loader2 className="h-10 w-10 text-[hsl(172_63%_28%)] animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold text-[hsl(172_43%_20%)]">
                Loading Results
              </p>
              <p className="text-[hsl(215_15%_50%)] mt-1">
                Analyzing experiment data...
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingState;
