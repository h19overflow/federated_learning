import React from 'react';

/**
 * StepContent Wrapper Component
 * Simple wrapper that passes through children - animations are handled by individual components
 *
 * Props:
 * - children: The content to render for the current step
 * - stepKey: Unique key for React reconciliation (forces remount on step change)
 * - isLoading: Optional loading state to disable interactions
 */

interface StepContentProps {
  children: React.ReactNode;
  stepKey: number;
  isLoading?: boolean;
}

const StepContent = ({ children, stepKey, isLoading = false }: StepContentProps) => {
  // Using key prop on the wrapper ensures React remounts the content on step change
  // This triggers the individual component's entrance animations naturally
  return (
    <div
      key={stepKey}
      className={`step-content-container ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      {children}
    </div>
  );
};

export default StepContent;
