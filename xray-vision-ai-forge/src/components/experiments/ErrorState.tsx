import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

export const ErrorState = ({ message, onRetry }: ErrorStateProps) => (
  <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
    <div className="relative w-20 h-20 mb-8">
      <div className="absolute inset-0 rounded-3xl bg-[hsl(0_72%_51%)]/10 blur-xl" />
      <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(0_60%_95%)] to-[hsl(0_50%_93%)] flex items-center justify-center border border-[hsl(0_60%_85%)] shadow-lg">
        <AlertCircle className="h-9 w-9 text-[hsl(0_72%_51%)]" />
      </div>
    </div>
    <p className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-2">Failed to Load Experiments</p>
    <p className="text-sm text-[hsl(215_15%_50%)] mb-8 max-w-md text-center leading-relaxed">{message}</p>
    <Button
      onClick={onRetry}
      className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-xl px-6 py-2.5 shadow-lg shadow-[hsl(172_63%_22%)]/25 hover:shadow-xl transition-all duration-300 font-medium"
    >
      Try Again
    </Button>
  </div>
);
