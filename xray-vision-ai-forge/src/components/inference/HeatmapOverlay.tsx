/**
 * HeatmapOverlay Component
 *
 * Displays GradCAM heatmap visualization showing which regions
 * of the X-ray the model focused on for its prediction.
 * Helps doctors understand the AI's reasoning.
 * Clean, minimal animations for medical UI.
 */

import React, { useState } from "react";
import { Flame, ZoomIn, ZoomOut, Info } from "lucide-react";
import { InferencePrediction } from "@/types/inference";

interface HeatmapOverlayProps {
  originalImageUrl: string;
  heatmapBase64: string;
  prediction: InferencePrediction;
}

export const HeatmapOverlay: React.FC<HeatmapOverlayProps> = ({
  originalImageUrl,
  heatmapBase64,
  prediction,
}) => {
  const [isZoomed, setIsZoomed] = useState(false);
  const isPneumonia = prediction.predicted_class === "PNEUMONIA";

  const heatmapDataUrl = `data:image/png;base64,${heatmapBase64}`;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center shadow-md">
            <Flame className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
              GradCAM Heatmap
            </h3>
            <p className="text-xs text-[hsl(215_15%_45%)] uppercase tracking-wide">
              Model Attention Map
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsZoomed(!isZoomed)}
          className="p-2 rounded-lg bg-white/80 border border-[hsl(168_20%_90%)] hover:bg-white"
          title={isZoomed ? "Zoom out" : "Zoom in"}
        >
          {isZoomed ? (
            <ZoomOut className="w-5 h-5 text-[hsl(172_63%_28%)]" />
          ) : (
            <ZoomIn className="w-5 h-5 text-[hsl(172_63%_28%)]" />
          )}
        </button>
      </div>

      {/* Heatmap Image */}
      <div
        className={`relative rounded-2xl overflow-hidden border-2 shadow-md ${
          isPneumonia
            ? "border-amber-300"
            : "border-emerald-300"
        } ${isZoomed ? "cursor-zoom-out" : "cursor-zoom-in"}`}
        onClick={() => setIsZoomed(!isZoomed)}
      >
        <img
          src={heatmapDataUrl}
          alt="GradCAM heatmap visualization"
          className={`w-full ${
            isZoomed ? "scale-150" : "scale-100"
          }`}
          style={{ imageRendering: "auto" }}
        />

        {/* Color scale legend */}
        <div className="absolute bottom-3 right-3 bg-black/70 backdrop-blur-sm rounded-lg p-2 flex items-center gap-2">
          <div className="w-24 h-3 rounded-full bg-gradient-to-r from-blue-500 via-yellow-400 to-red-500" />
          <div className="flex justify-between text-[10px] text-white w-24">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* Info card */}
      <div className="p-4 rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div className="space-y-2">
            <p className="text-sm text-[hsl(215_20%_35%)]">
              <strong>Interpretation:</strong> Warmer colors (red/orange)
              indicate regions that strongly influenced the model's{" "}
              {isPneumonia ? "pneumonia" : "normal"} prediction.
              {isPneumonia && (
                <span className="block mt-1">
                  These areas may show opacity patterns, consolidation, or
                  infiltrates typical of pneumonia.
                </span>
              )}
            </p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              GradCAM (Gradient-weighted Class Activation Mapping) provides
              explainable AI insights for clinical review.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeatmapOverlay;
