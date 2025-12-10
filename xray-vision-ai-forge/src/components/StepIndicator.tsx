
import React from 'react';
import { cn } from '@/lib/utils';

interface StepIndicatorProps {
  currentStep: number;
  steps: {
    id: number;
    name: string;
    description: string;
  }[];
}

const StepIndicator = ({ currentStep, steps }: StepIndicatorProps) => {
  return (
    <div className="w-full py-6">
      <div className="flex items-center justify-center">
        {steps.map((step, i) => (
          <div key={step.id}>
            <div className="relative">
              <div 
                className={cn(
                  "step-indicator w-12 h-12 flex items-center justify-center rounded-full transition-all border-2",
                  currentStep >= step.id
                    ? "bg-medical text-white border-medical"
                    : "bg-white text-muted-foreground border-muted",
                  currentStep === step.id && "active animate-pulse-glow"
                )}
              >
                <span className="text-sm font-medium">{step.id}</span>
              </div>
              <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 w-max">
                <span 
                  className={cn(
                    "text-xs font-medium",
                    currentStep >= step.id ? "text-medical-dark" : "text-muted-foreground"
                  )}
                >
                  {step.name}
                </span>
              </div>
            </div>
            {i < steps.length - 1 && (
              <div 
                className={cn(
                  "w-12 h-0.5 transition-all",
                  currentStep > i ? "bg-medical" : "bg-muted"
                )}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default StepIndicator;
