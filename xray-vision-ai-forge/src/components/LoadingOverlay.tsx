import React from 'react';
import { Loader2 } from 'lucide-react';

/**
 * LoadingOverlay Component
 * Displays a smooth loading state during transitions
 *
 * Props:
 * - isVisible: Whether to show the loading overlay
 * - message: Optional message to display
 * - variant: 'overlay' (full screen) or 'inline' (within container)
 */

interface LoadingOverlayProps {
  isVisible: boolean;
  message?: string;
  variant?: 'overlay' | 'inline';
}

const LoadingOverlay = ({
  isVisible,
  message = 'Loading...',
  variant = 'inline'
}: LoadingOverlayProps) => {
  if (!isVisible) return null;

  if (variant === 'overlay') {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-medical animate-spin" />
          <p className="text-gray-700 font-medium">{message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center py-12">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="w-6 h-6 text-medical animate-spin" />
        <p className="text-gray-600 text-sm font-medium">{message}</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;
