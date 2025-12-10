import React from 'react';
import { CheckCircle2, Circle } from 'lucide-react';

/**
 * ProgressIndicator Component
 * Displays current progress with breadcrumb-style navigation
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
    <div className="w-full py-6">
      <div className="flex items-center justify-center gap-2">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            {/* Step Circle */}
            <div className="flex flex-col items-center gap-2">
              <div
                className={`
                  flex items-center justify-center w-12 h-12 rounded-full
                  border-2 transition-all duration-300 font-semibold
                  ${
                    currentStep > step.id
                      ? 'bg-green-500 border-green-500 text-white shadow-md'
                      : currentStep === step.id
                        ? 'bg-medical text-white border-medical shadow-lg scale-105'
                        : 'bg-white text-gray-400 border-gray-300'
                  }
                `}
              >
                {currentStep > step.id ? (
                  <CheckCircle2 className="w-6 h-6" />
                ) : (
                  <span>{step.id}</span>
                )}
              </div>
              <div className="text-center">
                <p className={`text-sm font-semibold ${currentStep === step.id ? 'text-medical' : 'text-gray-900'}`}>
                  {step.name}
                </p>
                <p className="text-xs text-gray-500">{step.description}</p>
              </div>
            </div>

            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div
                className={`
                  h-1 w-8 mx-2 rounded-full transition-all duration-300
                  ${currentStep > step.id ? 'bg-green-500' : 'bg-gray-300'}
                `}
              />
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default ProgressIndicator;
