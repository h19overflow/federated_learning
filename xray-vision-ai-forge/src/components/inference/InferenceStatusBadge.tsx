/**
 * InferenceStatusBadge Component
 *
 * Subtle status indicator for inference service health.
 * Minimal animations with clear online/offline states.
 */

import React, { useEffect, useRef, useState } from "react";
import { Activity, AlertCircle } from "lucide-react";
import { checkInferenceHealth } from "@/services/inferenceApi";
import { HealthCheckResponse } from "@/types/inference";

export const InferenceStatusBadge: React.FC = () => {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await checkInferenceHealth();
        setHealth(response);
      } catch (err) {
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
    intervalRef.current = window.setInterval(checkHealth, 30000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  const isHealthy = health?.status === "healthy" && health?.model_loaded;

  if (loading) {
    return (
      <div className="fixed top-20 right-6 z-40 px-3 py-2 rounded-lg bg-white/95 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-[hsl(215_15%_70%)]" style={{ animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite" }} />
          <span className="text-xs font-medium text-[hsl(215_15%_50%)]">
            Checking...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed top-20 right-6 z-40 px-3 py-2 rounded-lg bg-white/95 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-sm">
      <div className="flex items-center gap-2">
        {/* Status indicator */}
        {isHealthy ? (
          <>
            <div className="relative w-2 h-2">
              <div className="absolute inset-0 w-2 h-2 rounded-full bg-[hsl(152_60%_42%)]" />
              <div
                className="absolute inset-0 w-2 h-2 rounded-full bg-[hsl(152_60%_42%)]"
                style={{
                  animation: "pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
                  opacity: 0.5,
                }}
              />
            </div>
            <Activity className="w-3 h-3 text-[hsl(152_60%_35%)]" />
            <span className="text-xs font-medium text-[hsl(152_60%_30%)]">
              Online
            </span>
          </>
        ) : (
          <>
            <div className="w-2 h-2 rounded-full bg-red-500" />
            <AlertCircle className="w-3 h-3 text-red-600" />
            <span className="text-xs font-medium text-red-700">
              Offline
            </span>
          </>
        )}

        {/* GPU indicator */}
        {isHealthy && health?.gpu_available && (
          <div className="ml-1 pl-2 border-l border-[hsl(168_20%_85%)]">
            <span className="text-xs text-[hsl(172_43%_25%)] font-medium">
              GPU
            </span>
          </div>
        )}
      </div>

      {/* Model version */}
      {isHealthy && health?.model_version && (
        <div className="mt-1 pt-1 border-t border-[hsl(168_20%_90%)]">
          <span className="text-[10px] text-[hsl(215_15%_55%)]">
            {health.model_version}
          </span>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes pulse-subtle {
          0%, 100% { transform: scale(1); opacity: 0.5; }
          50% { transform: scale(1.2); opacity: 0.2; }
        }
        @media (prefers-reduced-motion: reduce) {
          [style*="animation"] { animation: none !important; }
        }
      `}</style>
    </div>
  );
};

export default InferenceStatusBadge;
