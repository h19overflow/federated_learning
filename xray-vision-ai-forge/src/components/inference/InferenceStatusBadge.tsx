/**
 * InferenceStatusBadge Component
 *
 * Small badge showing inference service health status.
 * Green pulsing dot when healthy, red when unavailable.
 */

import React, { useEffect, useState } from "react";
import { Activity, AlertCircle } from "lucide-react";
import { checkInferenceHealth } from "@/services/inferenceApi";
import { HealthCheckResponse } from "@/types/inference";

export const InferenceStatusBadge: React.FC = () => {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await checkInferenceHealth();
        setHealth(response);
        setError(false);
      } catch (err) {
        setError(true);
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    // Check health immediately
    checkHealth();

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, []);

  const isHealthy = health?.status === "healthy" && health?.model_loaded;

  if (loading) {
    return (
      <div className="fixed top-20 right-6 z-40 px-4 py-2 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-lg">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[hsl(215_15%_70%)] animate-pulse" />
          <span className="text-xs font-medium text-[hsl(215_15%_50%)]">
            Checking...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed top-20 right-6 z-40 px-4 py-2 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-lg">
      <div className="flex items-center gap-2">
        {/* Status indicator */}
        {isHealthy ? (
          <>
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-[hsl(152_60%_42%)]" />
              <div className="absolute inset-0 w-2 h-2 rounded-full bg-[hsl(152_60%_42%)] animate-ping opacity-75" />
            </div>
            <Activity className="w-3.5 h-3.5 text-[hsl(152_60%_35%)]" />
            <span className="text-xs font-medium text-[hsl(152_60%_30%)]">
              Service Online
            </span>
          </>
        ) : (
          <>
            <div className="w-2 h-2 rounded-full bg-red-500" />
            <AlertCircle className="w-3.5 h-3.5 text-red-600" />
            <span className="text-xs font-medium text-red-700">
              Service Offline
            </span>
          </>
        )}

        {/* GPU indicator */}
        {isHealthy && health?.gpu_available && (
          <div className="ml-2 pl-2 border-l border-[hsl(168_20%_85%)]">
            <span className="text-xs text-[hsl(172_43%_25%)] font-medium">
              GPU
            </span>
          </div>
        )}
      </div>

      {/* Model version (on hover or always visible) */}
      {isHealthy && health?.model_version && (
        <div className="mt-1 pt-1 border-t border-[hsl(168_20%_90%)]">
          <span className="text-[10px] text-[hsl(215_15%_55%)]">
            {health.model_version}
          </span>
        </div>
      )}
    </div>
  );
};

export default InferenceStatusBadge;
