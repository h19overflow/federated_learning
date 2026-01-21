/**
 * HeatmapComparisonView Component
 *
 * Displays original X-ray and GradCAM heatmap overlay side by side
 * with zoom controls and a color legend explaining the heatmap.
 */

import React, { useState } from "react";
import { ZoomIn, ZoomOut, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface HeatmapComparisonViewProps {
  /** URL or base64 data URL of the original X-ray image */
  originalImageUrl: string;
  /** Base64-encoded heatmap overlay image (without data URL prefix) */
  heatmapBase64: string;
  /** Prediction class for contextual information */
  predictionClass?: "NORMAL" | "PNEUMONIA";
  /** Optional custom class name */
  className?: string;
}

export const HeatmapComparisonView: React.FC<HeatmapComparisonViewProps> = ({
  originalImageUrl,
  heatmapBase64,
  predictionClass,
  className = "",
}) => {
  const [isZoomed, setIsZoomed] = useState(false);

  // Format heatmap as data URL if needed
  const heatmapDataUrl = heatmapBase64.startsWith("data:")
    ? heatmapBase64
    : `data:image/png;base64,${heatmapBase64}`;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
            GradCAM Visualization
          </h4>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-[hsl(215_15%_50%)]"
                >
                  <Info className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right" className="max-w-xs">
                <p className="text-sm">
                  GradCAM highlights regions that influenced the model's
                  prediction. Warmer colors (red/orange) indicate higher
                  activation.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsZoomed(!isZoomed)}
          className="rounded-xl border-[hsl(172_30%_85%)] text-[hsl(172_43%_25%)]"
        >
          {isZoomed ? (
            <>
              <ZoomOut className="h-4 w-4 mr-1" />
              Reset
            </>
          ) : (
            <>
              <ZoomIn className="h-4 w-4 mr-1" />
              Zoom
            </>
          )}
        </Button>
      </div>

      {/* Side-by-side images */}
      <div
        className={`grid grid-cols-2 gap-4 transition-all duration-300 ${
          isZoomed ? "scale-110 origin-center" : ""
        }`}
      >
        {/* Original Image */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-[hsl(215_15%_45%)] text-center">
            Original X-Ray
          </p>
          <div className="relative rounded-xl overflow-hidden border-2 border-[hsl(172_30%_88%)] bg-white shadow-md">
            <img
              src={originalImageUrl}
              alt="Original X-ray"
              className="w-full h-auto object-contain max-h-[300px]"
            />
          </div>
        </div>

        {/* Heatmap Overlay */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-[hsl(215_15%_45%)] text-center">
            GradCAM Heatmap
          </p>
          <div className="relative rounded-xl overflow-hidden border-2 border-[hsl(172_30%_88%)] bg-white shadow-md">
            <img
              src={heatmapDataUrl}
              alt="GradCAM heatmap overlay"
              className="w-full h-auto object-contain max-h-[300px]"
            />
          </div>
        </div>
      </div>

      {/* Color Legend */}
      <div className="p-3 rounded-xl bg-[hsl(168_25%_96%)] border border-[hsl(172_30%_88%)]">
        <div className="flex items-center gap-4">
          <span className="text-sm text-[hsl(215_15%_45%)]">Activation:</span>
          <div className="flex items-center gap-2 flex-1">
            <span className="text-xs text-[hsl(215_15%_50%)]">Low</span>
            <div className="flex-1 h-3 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500" />
            <span className="text-xs text-[hsl(215_15%_50%)]">High</span>
          </div>
        </div>
      </div>

      {/* Interpretation hint */}
      {predictionClass && (
        <div className="p-3 rounded-xl bg-white/80 border border-[hsl(172_30%_88%)]">
          <p className="text-sm text-[hsl(215_15%_45%)]">
            {predictionClass === "PNEUMONIA" ? (
              <>
                <span className="font-medium text-amber-700">
                  Pneumonia detected:
                </span>{" "}
                Red/orange regions highlight areas where the model identified
                patterns consistent with pneumonia, such as consolidation or
                infiltrates.
              </>
            ) : (
              <>
                <span className="font-medium text-emerald-700">
                  Normal finding:
                </span>{" "}
                The heatmap shows regions the model examined. Low activation
                across lung fields supports the normal classification.
              </>
            )}
          </p>
        </div>
      )}
    </div>
  );
};

export default HeatmapComparisonView;
