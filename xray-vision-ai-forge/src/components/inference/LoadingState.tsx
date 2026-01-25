/**
 * LoadingState Component
 *
 * Minimal loading indicator with subtle spinner animation.
 * Clean progress feedback without heavy visual elements.
 */

import React from "react";
import { Progress } from "@/components/ui/progress";

interface LoadingStateProps {
  title?: string;
  description?: string;
  showProgress?: boolean;
  variant?: "default" | "compact";
  minHeight?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  title = "Processing...",
  description = "Please wait while we analyze your image",
  showProgress = true,
  variant = "default",
  minHeight = "min-h-[400px]",
}) => {
  if (variant === "compact") {
    return (
      <div className={`flex flex-col items-center justify-center ${minHeight} space-y-3`}>
        {/* Minimal spinner */}
        <div className="relative w-12 h-12">
          <svg className="w-full h-full" viewBox="0 0 50 50">
            <circle cx="25" cy="25" r="20" fill="none" stroke="hsl(172 30% 85%)" strokeWidth="2" />
            <circle
              cx="25"
              cy="25"
              r="20"
              fill="none"
              stroke="hsl(172 63% 28%)"
              strokeWidth="2"
              strokeDasharray="31.4 125.6"
              strokeLinecap="round"
              style={{
                animation: "spin 1.5s linear infinite",
                transformOrigin: "center",
              }}
            />
          </svg>
        </div>

        {/* Text content */}
        <div className="text-center space-y-1">
          <h3 className="text-base font-medium text-[hsl(172_43%_15%)]">
            {title}
          </h3>
          {description && (
            <p className="text-xs text-[hsl(215_15%_45%)]">{description}</p>
          )}
        </div>

        <style>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          @media (prefers-reduced-motion: reduce) {
            circle { animation: none !important; }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center justify-center ${minHeight} space-y-8`}>
      {/* Minimal spinner */}
      <div className="relative w-16 h-16">
        <svg className="w-full h-full" viewBox="0 0 50 50">
          <circle cx="25" cy="25" r="20" fill="none" stroke="hsl(172 30% 85%)" strokeWidth="2" />
          <circle
            cx="25"
            cy="25"
            r="20"
            fill="none"
            stroke="hsl(172 63% 28%)"
            strokeWidth="2"
            strokeDasharray="31.4 125.6"
            strokeLinecap="round"
            style={{
              animation: "spin 1.5s linear infinite",
              transformOrigin: "center",
            }}
          />
        </svg>
      </div>

      {/* Text content */}
      <div className="text-center space-y-2">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
          {title}
        </h3>
        {description && (
          <p className="text-sm text-[hsl(215_15%_45%)]">{description}</p>
        )}
      </div>

      {/* Progress bar */}
      {showProgress && (
        <div className="w-full max-w-sm">
          <Progress
            value={undefined}
            className="h-2 bg-[hsl(168_25%_96%)]"
          />
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (prefers-reduced-motion: reduce) {
          circle { animation: none !important; }
        }
      `}</style>
    </div>
  );
};

export default LoadingState;
