/**
 * HeatmapComparisonView Component
 *
 * Displays original X-ray and GradCAM heatmap overlay side by side
 * with zoom controls and a color legend explaining the heatmap.
 * Clean, minimal animations for medical UI.
 */

import React, { useState, useRef, useEffect } from "react";
import { ZoomIn, ZoomOut, Info, Maximize2, RotateCcw } from "lucide-react";
import { motion, AnimatePresence, useAnimation } from "framer-motion";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Slider } from "@/components/ui/slider";

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
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const containerRef = useRef<HTMLDivElement>(null);
  const controls = useAnimation();

  /**
   * Sync animation controls with zoom level changes
   * This ensures that both slider and mouse wheel zoom are animated smoothly
   */
  useEffect(() => {
    if (isModalOpen) {
      controls.start({ 
        scale: zoomLevel, 
        opacity: 1,
        transition: { type: "spring", damping: 25, stiffness: 120 }
      });
    }
  }, [zoomLevel, isModalOpen, controls]);

  /**
   * Handle mouse wheel zoom
   * Scroll up = Zoom in
   * Scroll down = Zoom out
   */
  const handleWheel = (e: React.WheelEvent) => {
    // Sensitivity factor for smoother zooming
    const zoomStep = 0.15;
    const delta = e.deltaY > 0 ? -zoomStep : zoomStep;

    setZoomLevel((prev) => {
      const newZoom = Math.min(Math.max(prev + delta, 0.5), 4);
      // Round to 2 decimal places to avoid floating point issues
      return Math.round(newZoom * 100) / 100;
    });
  };

  // Format heatmap as data URL if needed
  const heatmapDataUrl = heatmapBase64.startsWith("data:")
    ? heatmapBase64
    : `data:image/png;base64,${heatmapBase64}`;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-base font-semibold text-[hsl(172_43%_15%)]">
            Side-by-Side Comparison
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
                  Left: Original X-ray. Right: GradCAM heatmap showing model attention.
                  Warmer colors indicate higher activation.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsModalOpen(true)}
          className="rounded-lg border-[hsl(172_30%_85%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(172_30%_95%)] transition-colors"
        >
          <Maximize2 className="h-4 w-4 mr-1" />
          Full Screen Zoom
        </Button>
      </div>

      {/* Side-by-side images (Preview) */}
      <div className="grid grid-cols-2 gap-4">
        {/* Original Image */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide text-center">
            Original X-Ray
          </p>
          <div 
            className="relative rounded-xl overflow-hidden border-2 border-[hsl(172_30%_88%)] bg-white shadow-md cursor-zoom-in group"
            onClick={() => setIsModalOpen(true)}
          >
            <img
              src={originalImageUrl}
              alt="Original X-ray"
              className="w-full h-auto object-contain max-h-[350px] transition-transform duration-500 group-hover:scale-105"
            />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
              <ZoomIn className="text-white h-8 w-8 drop-shadow-md" />
            </div>
          </div>
        </div>

        {/* Heatmap Overlay */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide text-center">
            GradCAM Heatmap
          </p>
          <div 
            className="relative rounded-xl overflow-hidden border-2 border-[hsl(172_30%_88%)] bg-white shadow-md cursor-zoom-in group"
            onClick={() => setIsModalOpen(true)}
          >
            <img
              src={heatmapDataUrl}
              alt="GradCAM heatmap overlay"
              className="w-full h-auto object-contain max-h-[350px] transition-transform duration-500 group-hover:scale-105"
            />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
              <ZoomIn className="text-white h-8 w-8 drop-shadow-md" />
            </div>
          </div>
        </div>
      </div>

      {/* Zoom Modal */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="max-w-[98vw] w-[98vw] h-[95vh] flex flex-col p-0 overflow-hidden bg-slate-950 border-slate-800 shadow-2xl">
          <DialogHeader className="p-4 bg-slate-900/50 border-b border-slate-800 flex flex-row items-center justify-between space-y-0">
            <DialogTitle className="text-slate-100 flex items-center gap-2">
              <Maximize2 className="h-5 w-5 text-teal-400" />
              Diagnostic Zoom View
            </DialogTitle>
          </DialogHeader>
          
          <div 
            ref={containerRef}
            onWheel={handleWheel}
            className="flex-1 relative overflow-hidden bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-slate-900 to-black flex items-center justify-center cursor-grab active:cursor-grabbing"
          >
            <motion.div
              drag
              dragConstraints={containerRef}
              dragElastic={0.1}
              initial={{ scale: 0.9, opacity: 0, x: 0, y: 0 }}
              animate={controls}
              className="flex gap-4 md:gap-12 p-12 items-center"
            >
              {/* Original Image in Modal */}
              <div className="flex flex-col gap-3 items-center">
                <span className="px-3 py-1 rounded-full bg-slate-800/80 text-slate-300 text-[10px] font-bold uppercase tracking-widest border border-slate-700">
                  Original
                </span>
                <div className="rounded-lg overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-slate-700 bg-black">
                  <img
                    src={originalImageUrl}
                    alt="Original X-ray zoomed"
                    className="max-h-[70vh] w-auto object-contain pointer-events-none"
                  />
                </div>
              </div>

              {/* Heatmap in Modal */}
              <div className="flex flex-col gap-3 items-center">
                <span className="px-3 py-1 rounded-full bg-teal-900/40 text-teal-300 text-[10px] font-bold uppercase tracking-widest border border-teal-800/50">
                  GradCAM Heatmap
                </span>
                <div className="rounded-lg overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-slate-700 bg-black">
                  <img
                    src={heatmapDataUrl}
                    alt="GradCAM heatmap zoomed"
                    className="max-h-[70vh] w-auto object-contain pointer-events-none"
                  />
                </div>
              </div>
            </motion.div>

            {/* Floating Instructions */}
            <div className="absolute bottom-6 left-6 px-4 py-2 rounded-lg bg-black/60 backdrop-blur-md border border-white/10 text-white/60 text-xs pointer-events-none">
              Drag to pan • Scroll or use slider to zoom
            </div>
          </div>

          {/* Modal Footer Controls */}
          <div className="p-6 bg-slate-900/80 backdrop-blur-xl border-t border-slate-800 flex flex-col md:flex-row items-center justify-center gap-8">
            <div className="flex items-center gap-4 w-full max-w-md">
              <ZoomOut className="text-slate-400 h-5 w-5 shrink-0" />
              <Slider
                value={[zoomLevel]}
                min={0.5}
                max={4}
                step={0.1}
                onValueChange={([v]) => setZoomLevel(v)}
                className="flex-1"
              />
              <ZoomIn className="text-slate-400 h-5 w-5 shrink-0" />
            </div>
            
            <div className="flex items-center gap-3">
              <div className="h-8 w-[1px] bg-slate-700 hidden md:block mx-2" />
              <Button 
                variant="secondary" 
                size="sm" 
                onClick={() => {
                  setZoomLevel(1.0);
                  controls.start({ 
                    x: 0, 
                    y: 0, 
                    scale: 1.0,
                    transition: { type: "spring", damping: 25, stiffness: 120 }
                  });
                }}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset View
              </Button>
              <span className="text-slate-400 text-sm font-mono w-12 text-center">
                {Math.round(zoomLevel * 100)}%
              </span>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Color Legend */}
      <div className="p-4 rounded-xl bg-[hsl(168_25%_96%)] border border-[hsl(172_30%_88%)]">
        <div className="space-y-2">
          <p className="text-xs font-semibold text-[hsl(215_15%_45%)] uppercase tracking-wide">
            Activation Scale
          </p>
          <div className="flex items-center gap-3">
            <span className="text-xs text-[hsl(215_15%_50%)] font-medium">Low</span>
            <div className="flex-1 h-3 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500" />
            <span className="text-xs text-[hsl(215_15%_50%)] font-medium">High</span>
          </div>
        </div>
      </div>

      {/* Interpretation hint */}
      {predictionClass && (
        <div
          className="p-4 rounded-xl border-l-4"
          style={{
            backgroundColor:
              predictionClass === "PNEUMONIA"
                ? "hsl(35 60% 95%)"
                : "hsl(152 50% 95%)",
            borderColor:
              predictionClass === "PNEUMONIA"
                ? "hsl(35 70% 45%)"
                : "hsl(152 60% 42%)",
          }}
        >
          <p className="text-sm text-[hsl(215_15%_40%)] leading-relaxed">
            {predictionClass === "PNEUMONIA" ? (
              <>
                <span className="font-semibold text-amber-700">
                  Pneumonia Pattern:
                </span>{" "}
                Red/orange regions indicate areas where the model identified
                patterns consistent with pneumonia, such as consolidation or
                infiltrates.
              </>
            ) : (
              <>
                <span className="font-semibold text-emerald-700">
                  Normal Pattern:
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
