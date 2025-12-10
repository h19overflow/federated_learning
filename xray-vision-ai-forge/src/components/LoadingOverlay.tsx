import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * LoadingOverlay Component
 * Displays a smooth loading state during transitions
 * Redesigned with Clinical Clarity theme - subtle blur, teal accent
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
      <div
        className="fixed inset-0 bg-[hsl(172_30%_15%)]/40 backdrop-blur-sm flex items-center justify-center z-50"
        style={{ animation: 'fadeIn 0.2s ease-out' }}
      >
        <div
          className="bg-white/95 backdrop-blur-xl rounded-2xl p-8 flex flex-col items-center gap-5 shadow-xl shadow-[hsl(172_40%_85%)]/30 border border-[hsl(168_20%_92%)]"
          style={{ animation: 'fadeIn 0.3s ease-out' }}
        >
          <div className="relative">
            {/* Glow effect */}
            <div className="absolute inset-0 rounded-full bg-[hsl(172_63%_35%)] animate-ping opacity-20" />
            <Loader2 className="w-10 h-10 text-[hsl(172_63%_28%)] animate-spin relative z-10" />
          </div>
          <div className="text-center">
            <p className="text-[hsl(172_43%_20%)] font-medium text-lg">{message}</p>
            <p className="text-[hsl(215_15%_55%)] text-sm mt-1">Please wait...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex items-center justify-center py-16"
      style={{ animation: 'fadeIn 0.3s ease-out' }}
    >
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          {/* Subtle glow effect */}
          <div className="absolute inset-0 rounded-full bg-[hsl(172_63%_35%)] animate-ping opacity-15" />
          <div className="p-3 rounded-full bg-[hsl(172_40%_94%)]">
            <Loader2 className="w-6 h-6 text-[hsl(172_63%_28%)] animate-spin" />
          </div>
        </div>
        <p className="text-[hsl(215_15%_45%)] text-sm font-medium">{message}</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;
