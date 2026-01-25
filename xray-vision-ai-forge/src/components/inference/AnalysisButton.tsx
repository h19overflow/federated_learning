/**
 * AnalysisButton Component
 *
 * Primary action button for analysis with clear loading state.
 * Supports single and batch analysis with improved visual feedback.
 */

import React from "react";
import { Button } from "@/components/ui/button";
import { Sparkles, RefreshCw } from "lucide-react";

interface AnalysisButtonProps {
  onClick: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: "analyze" | "retry";
  imageCount?: number;
  className?: string;
}

export const AnalysisButton: React.FC<AnalysisButtonProps> = ({
  onClick,
  loading = false,
  disabled = false,
  variant = "analyze",
  imageCount = 1,
  className = "w-full",
}) => {
  if (variant === "retry") {
    return (
      <Button
        onClick={onClick}
        variant="outline"
        size="lg"
        disabled={disabled || loading}
        className={`${className} py-6 rounded-xl border border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(172_30%_96%)] hover:border-[hsl(172_40%_75%)] transition-colors duration-200`}
      >
        <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} style={{ animationDuration: "1.5s" }} />
        Analyze Another {imageCount > 1 ? "Batch" : "Image"}
      </Button>
    );
  }

  // Analyze variant
  const buttonText =
    imageCount === 1
      ? "Analyze Image"
      : `Analyze ${imageCount} Image${imageCount !== 1 ? "s" : ""}`;

  return (
    <Button
      onClick={onClick}
      disabled={disabled || loading}
      size="lg"
      className={`${className} bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white py-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed`}
    >
      {loading ? (
        <>
          <div className="mr-2 h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" style={{ animationDuration: "1.5s" }} />
          <span className="text-sm font-medium">
            {imageCount > 1
              ? `Analyzing ${imageCount} images...`
              : "Analyzing..."}
          </span>
        </>
      ) : (
        <>
          <Sparkles className="mr-2 h-4 w-4" />
          <span className="text-sm font-medium">{buttonText}</span>
        </>
      )}
    </Button>
  );
};

export default AnalysisButton;
