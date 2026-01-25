/**
 * PredictionResult Component
 *
 * Displays prediction results with prominent confidence indicators.
 * Color-coded for NORMAL (green) vs PNEUMONIA (amber/red).
 * Minimal animations, clear medical UI hierarchy.
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
        textColor: "hsl(35 70% 35%)",
      }
    : {
        primary: "hsl(152 60% 42%)", // Green
        bg: "hsl(152 50% 95%)",
        border: "hsl(152 50% 80%)",
        icon: CheckCircle2,
        label: "Normal",
        textColor: "hsl(152 60% 30%)",
      };

  const Icon = colors.icon;

  return (
    <div className="space-y-6">
      {/* Main prediction card - prominent display */}
      <div
        className="p-8 rounded-3xl border-2 shadow-lg"
        style={{
          backgroundColor: colors.bg,
          borderColor: colors.border,
        }}
      >
        {/* Prediction icon and label */}
        <div className="flex items-center gap-4 mb-8">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-md flex-shrink-0"
            style={{ backgroundColor: "white" }}
          >
            <Icon className="w-8 h-8" style={{ color: colors.primary }} />
          </div>
          <div>
            <h2
              className="text-3xl font-bold mb-1"
              style={{ color: colors.textColor }}
            >
              {colors.label}
            </h2>
            <p className="text-sm text-[hsl(215_15%_45%)]">
              {prediction.predicted_class}
            </p>
          </div>
        </div>

        {/* Confidence meter - prominent */}
        <div className="mb-8 p-6 bg-white/60 rounded-2xl">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-[hsl(172_43%_20%)]">
              Confidence Score
            </span>
            <span
              className="text-3xl font-bold"
              style={{ color: colors.primary }}
            >
              {confidence.toFixed(1)}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="relative h-3 bg-white rounded-full overflow-hidden shadow-inner">
            <div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                width: `${confidence}%`,
                backgroundColor: colors.primary,
              }}
            />
          </div>
        </div>

        {/* Probability breakdown */}
        <div className="space-y-4">
          <h4 className="text-xs font-semibold text-[hsl(172_43%_20%)] uppercase tracking-widest">
            Probability Distribution
          </h4>

          {/* Pneumonia probability */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-[hsl(215_15%_40%)]">
                Pneumonia
              </span>
              <span className="text-sm font-bold text-[hsl(172_43%_20%)]">
                {(prediction.pneumonia_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-white rounded-full overflow-hidden shadow-inner">
              <div
                className="h-full bg-[hsl(35_70%_50%)]"
                style={{
                  width: `${prediction.pneumonia_probability * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Normal probability */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-[hsl(215_15%_40%)]">
                Normal
              </span>
              <span className="text-sm font-bold text-[hsl(172_43%_20%)]">
                {(prediction.normal_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-white rounded-full overflow-hidden shadow-inner">
              <div
                className="h-full bg-[hsl(152_60%_42%)]"
                style={{ width: `${prediction.normal_probability * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Metadata cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-md">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-[hsl(172_63%_28%)]" />
            <span className="text-xs font-semibold text-[hsl(215_15%_50%)] uppercase tracking-wide">
              Model
            </span>
          </div>
          <p className="text-sm font-medium text-[hsl(172_43%_15%)] truncate">
            {modelVersion}
          </p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(168_20%_90%)] shadow-md">
          <div className="flex items-center gap-2 mb-2">
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
              Time
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
