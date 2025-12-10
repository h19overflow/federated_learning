import React, { useEffect, useState } from 'react';

/**
 * StepContent Wrapper Component
 * Provides smooth fade-in/fade-out animations when switching between steps
 *
 * Props:
 * - children: The content to render for the current step
 * - stepKey: Unique key to trigger re-animation when step changes
 */

interface StepContentProps {
  children: React.ReactNode;
  stepKey: number;
  isLoading?: boolean;
}

const StepContent = ({ children, stepKey, isLoading = false }: StepContentProps) => {
  const [isVisible, setIsVisible] = useState(true);
  const [displayKey, setDisplayKey] = useState(stepKey);

  useEffect(() => {
    // Trigger fade-out animation
    setIsVisible(false);

    // After fade-out completes, update content and fade-in
    const timer = setTimeout(() => {
      setDisplayKey(stepKey);
      setIsVisible(true);
    }, 150); // Half of fade-out animation duration

    return () => clearTimeout(timer);
  }, [stepKey]);

  return (
    <div
      key={displayKey}
      className={`step-content-container transition-all duration-300 ${
        isVisible ? 'step-content-enter' : 'step-content-exit'
      } ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      {children}
    </div>
  );
};

export default StepContent;
