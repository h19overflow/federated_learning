import React from "react";
import { Check, Database, Settings, Cpu, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * ProgressIndicator Component
 * Minimalist navigation with subtle green accents matching Landing theme
 */

interface Step {
  id: number;
  name: string;
  description: string;
  icon: React.ReactNode;
}

interface ProgressIndicatorProps {
  currentStep: number;
  steps: { id: number; name: string; description: string }[];
}

const stepIcons: Record<number, React.ReactNode> = {
  1: <Database className="w-4 h-4" />,
  2: <Settings className="w-4 h-4" />,
  3: <Cpu className="w-4 h-4" />,
  4: <BarChart3 className="w-4 h-4" />,
};

const ProgressIndicator = ({ currentStep, steps }: ProgressIndicatorProps) => {
  const enrichedSteps: Step[] = steps.map((step) => ({
    ...step,
    icon: stepIcons[step.id],
  }));

  return (
    <div className="w-full py-8 md:py-10">
      <div className="relative max-w-3xl mx-auto">
        {/* Background Track */}
        <div className="absolute top-6 left-14 right-14 h-0.5 bg-[hsl(172_20%_90%)] rounded-full" />

        {/* Progress Fill */}
        <div
          className="absolute top-6 left-14 h-0.5 bg-gradient-to-r from-[hsl(172_50%_45%)] to-[hsl(172_63%_35%)] rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${Math.min(((currentStep - 1) / (steps.length - 1)) * (100 - 28), 100 - 28)}%`,
          }}
        />

        {/* Steps */}
        <div className="relative flex items-start justify-between">
          {enrichedSteps.map((step) => {
            const isCompleted = currentStep > step.id;
            const isCurrent = currentStep === step.id;

            return (
              <div
                key={step.id}
                className="flex flex-col items-center gap-3"
              >
                {/* Step Circle */}
                <div className="relative">
                  {/* Subtle glow for current step */}
                  {isCurrent && (
                    <div className="absolute inset-0 rounded-full bg-[hsl(172_50%_45%)] blur-md opacity-20" />
                  )}

                  <div
                    className={cn(
                      "relative flex items-center justify-center w-12 h-12 rounded-full",
                      "border-2 transition-all duration-500",
                      isCompleted
                        ? "bg-[hsl(172_63%_28%)] border-[hsl(172_63%_28%)] text-white shadow-md shadow-[hsl(172_63%_28%)]/20"
                        : isCurrent
                          ? "bg-white border-[hsl(172_50%_55%)] text-[hsl(172_63%_28%)] shadow-lg shadow-[hsl(172_50%_45%)]/15"
                          : "bg-white border-[hsl(172_20%_88%)] text-[hsl(172_20%_65%)]"
                    )}
                  >
                    {isCompleted ? (
                      <Check className="w-5 h-5" />
                    ) : (
                      <span className={cn("transition-transform duration-300", isCurrent && "scale-110")}>
                        {step.icon}
                      </span>
                    )}
                  </div>

                  {/* Current indicator dot */}
                  {isCurrent && (
                    <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-[hsl(172_50%_50%)]" />
                  )}
                </div>

                {/* Labels */}
                <div className="text-center">
                  <p
                    className={cn(
                      "text-sm font-semibold transition-colors duration-300",
                      isCurrent
                        ? "text-[hsl(172_63%_22%)]"
                        : isCompleted
                          ? "text-[hsl(172_43%_28%)]"
                          : "text-[hsl(172_15%_55%)]"
                    )}
                  >
                    {step.name}
                  </p>
                  <p
                    className={cn(
                      "text-xs mt-1 hidden md:block transition-colors duration-300 max-w-[120px]",
                      isCurrent
                        ? "text-[hsl(172_40%_45%)]"
                        : "text-[hsl(172_15%_55%)]"
                    )}
                  >
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ProgressIndicator;
