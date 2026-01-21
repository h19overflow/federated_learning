/**
 * PredictionResult Component
 *
 * Displays prediction results with visual confidence indicators.
 * Color-coded for NORMAL (green) vs PNEUMONIA (amber/red).
 */

import React from "react";
import { CheckCircle2, AlertTriangle, Activity } from "lucide-react";
import { InferencePrediction } from "@/types/inference";

interface PredictionResultProps {
  prediction: InferencePrediction;
  modelVersion: string;
  processingTimeMs: number;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  modelVersion,
  processingTimeMs,
}) => {
  const isPneumonia = prediction.predicted_class === "PNEUMONIA";
  const confidence = prediction.confidence * 100;

  // Color scheme based on prediction
  const colors = isPneumonia
    ? {
        primary: "hsl(35 70% 45%)", // Amber
        bg: "hsl(35 60% 95%)",
        border: "hsl(35 60% 80%)",
        icon: AlertTriangle,
        label: "Pneumonia Detected",
      }
    : {
        primary: "hsl(152 60% 42%)", // Green
        bg: "hsl(152 50% 95%)",
        border: "hsl(152 50% 80%)",
        icon: CheckCircle2,
        label: "Normal",
      };

  const Icon = colors.icon;

  return (
    <div className="space-y-6">
      {/* Main prediction card */}
      <div
        className="p-8 rounded-3xl border-2 shadow-xl transition-all duration-500"
        style={{
          backgroundColor: colors.bg,
          borderColor: colors.border,
        }}
      >
        {/* Prediction icon and label */}
        <div className="flex items-center gap-4 mb-6">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-md"
            style={{ backgroundColor: "white" }}
          >
            <Icon className="w-8 h-8" style={{ color: colors.primary }} />
          </div>
          <div>
            <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
              {colors.label}
            </h3>
            <p className="text-[hsl(215_15%_45%)]">
              Prediction: {prediction.predicted_class}
            </p>
          </div>
        </div>

        {/* Confidence meter */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-[hsl(172_43%_20%)]">
              Confidence Level
            </span>
            <span
              className="text-lg font-bold"
              style={{ color: colors.primary }}
            >
              {confidence.toFixed(1)}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="relative h-4 bg-white rounded-full overflow-hidden shadow-inner">
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${confidence}%`,
                backgroundColor: colors.primary,
              }}
            />
          </div>
        </div>

        {/* Probability breakdown */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-[hsl(172_43%_20%)] uppercase tracking-wide">
            Probability Breakdown
          </h4>

          {/* Pneumonia probability */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-[hsl(215_15%_40%)]">Pneumonia</span>
            <div className="flex items-center gap-3 flex-1 ml-4">
              <div className="flex-1 h-2 bg-white rounded-full overflow-hidden shadow-inner">
                <div
                  className="h-full bg-[hsl(35_70%_50%)] transition-all duration-700"
                  style={{
                    width: `${prediction.pneumonia_probability * 100}%`,
                  }}
                />
              </div>
              <span className="text-sm font-medium text-[hsl(172_43%_20%)] w-14 text-right">
                {(prediction.pneumonia_probability * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Normal probability */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-[hsl(215_15%_40%)]">Normal</span>
            <div className="flex items-center gap-3 flex-1 ml-4">
              <div className="flex-1 h-2 bg-white rounded-full overflow-hidden shadow-inner">
                <div
                  className="h-full bg-[hsl(152_60%_42%)] transition-all duration-700"
                  style={{ width: `${prediction.normal_probability * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium text-[hsl(172_43%_20%)] w-14 text-right">
                {(prediction.normal_probability * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Metadata card */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-md">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-4 h-4 text-[hsl(172_63%_28%)]" />
            <span className="text-xs font-semibold text-[hsl(215_15%_50%)] uppercase tracking-wide">
              Model Version
            </span>
          </div>
          <p className="text-sm font-medium text-[hsl(172_43%_15%)] truncate">
            {modelVersion}
          </p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-md">
          <div className="flex items-center gap-2 mb-1">
            <svg
              className="w-4 h-4 text-[hsl(172_63%_28%)]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <span className="text-xs font-semibold text-[hsl(215_15%_50%)] uppercase tracking-wide">
              Processing Time
            </span>
          </div>
          <p className="text-sm font-medium text-[hsl(172_43%_15%)]">
            {processingTimeMs.toFixed(0)} ms
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
