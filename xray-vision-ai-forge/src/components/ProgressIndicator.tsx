import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * ProgressIndicator Component
 * Displays current progress with breadcrumb-style navigation
 * Redesigned with Clinical Clarity theme - teal accents, no green
 *
 * Props:
 * - currentStep: Current step number (1-4)
 * - steps: Array of step definitions with id, name, description
 */

interface Step {
  id: number;
  name: string;
  description: string;
}

interface ProgressIndicatorProps {
  currentStep: number;
  steps: Step[];
}

const ProgressIndicator = ({ currentStep, steps }: ProgressIndicatorProps) => {
  return (
    <div className="w-full py-8">
      <div className="flex items-center justify-center gap-2">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            {/* Step Circle */}
            <div className="flex flex-col items-center gap-3">
              <div
                className={cn(
                  'flex items-center justify-center w-12 h-12 rounded-full',
                  'border-2 transition-all duration-300 font-semibold',
                  currentStep > step.id
                    ? 'bg-[hsl(172_63%_28%)] border-[hsl(172_63%_28%)] text-white shadow-md shadow-[hsl(172_63%_28%)]/25'
                    : currentStep === step.id
                      ? 'bg-[hsl(172_63%_22%)] text-white border-[hsl(172_63%_22%)] shadow-lg shadow-[hsl(172_63%_22%)]/30 scale-105'
                      : 'bg-white text-[hsl(215_15%_55%)] border-[hsl(210_15%_85%)]'
                )}
              >
                {currentStep > step.id ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <span className="text-sm">{step.id}</span>
                )}
              </div>

              {/* Step labels */}
              <div className="text-center">
                <p className={cn(
                  'text-sm font-semibold transition-colors',
                  currentStep === step.id
                    ? 'text-[hsl(172_63%_25%)]'
                    : currentStep > step.id
                      ? 'text-[hsl(172_43%_25%)]'
                      : 'text-[hsl(172_43%_20%)]'
                )}>
                  {step.name}
                </p>
                <p className="text-xs text-[hsl(215_15%_55%)] mt-0.5">
                  {step.description}
                </p>
              </div>
            </div>

            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div className="relative h-1 w-10 mx-2 -mt-10 rounded-full overflow-hidden bg-[hsl(210_15%_92%)]">
                <div
                  className={cn(
                    'absolute inset-y-0 left-0 rounded-full transition-all duration-500',
                    'bg-gradient-to-r from-[hsl(172_63%_35%)] to-[hsl(172_63%_28%)]'
                  )}
                  style={{
                    width: currentStep > step.id ? '100%' : '0%'
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

export default ProgressIndicator;
