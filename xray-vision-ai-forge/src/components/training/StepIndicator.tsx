import React from "react";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

interface StepIndicatorProps {
  currentStep: number;
  steps: {
    id: number;
    name: string;
    description: string;
  }[];
}

/**
 * Step indicator component for multi-step flows
 * Redesigned with Clinical Clarity theme - teal accents
 */
const StepIndicator = ({ currentStep, steps }: StepIndicatorProps) => {
  return (
    <div className="w-full py-8">
      <div className="flex items-center justify-center">
        {steps.map((step, i) => (
          <React.Fragment key={step.id}>
            <div className="flex flex-col items-center">
              {/* Step circle */}
              <div
                className={cn(
                  "relative w-12 h-12 flex items-center justify-center rounded-full transition-all duration-300",
                  "border-2 font-semibold",
                  currentStep > step.id
                    ? "bg-[hsl(172_63%_28%)] border-[hsl(172_63%_28%)] text-white"
                    : currentStep === step.id
                      ? "bg-[hsl(172_63%_22%)] border-[hsl(172_63%_22%)] text-white shadow-lg shadow-[hsl(172_63%_22%)]/30"
                      : "bg-white border-[hsl(210_15%_85%)] text-[hsl(215_15%_55%)]",
                )}
                style={{
                  animation:
                    currentStep === step.id
                      ? "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
                      : undefined,
                }}
              >
                {currentStep > step.id ? (
                  <Check className="w-5 h-5" />
                ) : (
                  <span className="text-sm">{step.id}</span>
                )}

                {/* Active glow effect */}
                {currentStep === step.id && (
                  <div className="absolute inset-0 rounded-full bg-[hsl(172_63%_35%)] animate-ping opacity-20" />
                )}
              </div>

              {/* Step label */}
              <div className="mt-3 text-center">
                <span
                  className={cn(
                    "text-xs font-semibold transition-colors",
                    currentStep >= step.id
                      ? "text-[hsl(172_43%_20%)]"
                      : "text-[hsl(215_15%_55%)]",
                  )}
                >
                  {step.name}
                </span>
              </div>
            </div>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className="relative w-16 mx-2 -mt-6">
                {/* Background track */}
                <div className="h-0.5 w-full bg-[hsl(210_15%_90%)] rounded-full" />
                {/* Progress fill */}
                <div
                  className={cn(
                    "absolute top-0 left-0 h-0.5 rounded-full transition-all duration-500",
                    "bg-gradient-to-r from-[hsl(172_63%_35%)] to-[hsl(172_63%_28%)]",
                  )}
                  style={{
                    width: currentStep > step.id ? "100%" : "0%",
                  }}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default StepIndicator;
