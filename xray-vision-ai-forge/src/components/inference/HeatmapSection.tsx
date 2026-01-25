/**
 * Heatmap Section Component
 * Displays GradCAM heatmap visualization with clean toggle controls
 * Minimal animations, clear medical UI
 */

import React from "react";
import { Button } from "@/components/ui/button";
import { Flame, X } from "lucide-react";
import HeatmapComparisonView from "@/components/inference/HeatmapComparisonView";
import { HeatmapResponse } from "@/types/inference";

interface HeatmapSectionProps {
  showHeatmap: boolean;
  heatmap: HeatmapResponse | null;
  heatmapLoading: boolean;
  imageUrl: string | null;
  predictionClass: string;
  onToggleHeatmap: (show: boolean) => void;
  onGenerateHeatmap: () => void;
}

const HeatmapSection: React.FC<HeatmapSectionProps> = ({
  showHeatmap,
  heatmap,
  heatmapLoading,
  imageUrl,
  predictionClass,
  onToggleHeatmap,
  onGenerateHeatmap,
}) => {
  return (
    <div className="result-card">
      {showHeatmap && heatmap && imageUrl ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center shadow-md">
                <Flame className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
                  GradCAM Visualization
                </h3>
                <p className="text-xs text-[hsl(215_15%_45%)]">
                  Model attention map
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onToggleHeatmap(false)}
              className="rounded-lg text-[hsl(215_15%_45%)] hover:bg-red-50"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          <HeatmapComparisonView
            originalImageUrl={imageUrl}
            heatmapBase64={heatmap.heatmap_base64}
            predictionClass={predictionClass}
          />
        </div>
      ) : (
        <div className="p-6 rounded-2xl bg-gradient-to-br from-white to-[hsl(168_25%_97%)] border border-[hsl(172_30%_88%)] shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-1">
                GradCAM Heatmap
              </h3>
              <p className="text-sm text-[hsl(215_15%_45%)]">
                Visualize which regions influenced the model's prediction
              </p>
            </div>
            <Button
              onClick={onGenerateHeatmap}
              disabled={heatmapLoading}
              className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl whitespace-nowrap"
            >
              {heatmapLoading ? (
                <>
                  <div className="w-4 h-4 mr-2 border-2 border-white/30 border-t-white rounded-full" />
                  Generating...
                </>
              ) : (
                <>
                  <Flame className="w-4 h-4 mr-2" />
                  Generate Heatmap
                </>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapSection;
