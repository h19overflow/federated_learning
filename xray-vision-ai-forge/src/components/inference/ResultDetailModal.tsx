/**
 * ResultDetailModal Component
 *
 * Modal dialog that displays full prediction details for a single image.
 * Includes large image preview, prediction results, and on-demand GradCAM heatmap.
 * Supports navigation between results with arrow keys and prev/next buttons.
 */

import React, { useEffect, useState } from 'react';
import { X, ChevronLeft, ChevronRight, Eye, EyeOff, Loader2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { SingleImageResult } from '@/types/inference';
import PredictionResult from './PredictionResult';
import HeatmapOverlay from './HeatmapOverlay';
import { generateHeatmap } from '@/services/inferenceApi';

interface ResultDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: SingleImageResult | null;
  imageUrl: string | null;
  imageFile?: File | null;
  onNext?: () => void;
  onPrevious?: () => void;
  canGoNext?: boolean;
  canGoPrevious?: boolean;
  currentIndex?: number;
  totalResults?: number;
}

export const ResultDetailModal: React.FC<ResultDetailModalProps> = ({
  isOpen,
  onClose,
  result,
  imageUrl,
  imageFile,
  onNext,
  onPrevious,
  canGoNext = false,
  canGoPrevious = false,
  currentIndex,
  totalResults,
}) => {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapBase64, setHeatmapBase64] = useState<string | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);

  // Reset heatmap state when result changes
  useEffect(() => {
    setShowHeatmap(false);
    setHeatmapBase64(null);
    setHeatmapError(null);
  }, [result?.filename]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' && canGoNext && onNext) {
        onNext();
      } else if (e.key === 'ArrowLeft' && canGoPrevious && onPrevious) {
        onPrevious();
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, canGoNext, canGoPrevious, onNext, onPrevious, onClose]);

  const handleToggleHeatmap = async () => {
    if (showHeatmap) {
      setShowHeatmap(false);
      return;
    }

    // If we already have the heatmap, just show it
    if (heatmapBase64) {
      setShowHeatmap(true);
      return;
    }

    // Generate heatmap on-demand
    if (!imageFile) {
      setHeatmapError('Image file not available');
      return;
    }

    setHeatmapLoading(true);
    setHeatmapError(null);

    try {
      const response = await generateHeatmap(imageFile);
      setHeatmapBase64(response.heatmap_base64);
      setShowHeatmap(true);
    } catch (err) {
      setHeatmapError(err instanceof Error ? err.message : 'Failed to generate heatmap');
    } finally {
      setHeatmapLoading(false);
    }
  };

  if (!result) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto bg-[hsl(168_25%_97%)] border-2 border-[hsl(172_30%_88%)]">
        <DialogHeader>
          <div className="flex items-center justify-between pr-8">
            <div>
              <DialogTitle className="text-2xl font-semibold text-[hsl(172_43%_15%)]">
                {result.filename}
              </DialogTitle>
              {currentIndex !== undefined && totalResults !== undefined && (
                <p className="text-sm text-[hsl(215_15%_45%)] mt-1">
                  Image {currentIndex + 1} of {totalResults}
                </p>
              )}
            </div>

            {/* Navigation arrows */}
            {(onPrevious || onNext) && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onPrevious}
                  disabled={!canGoPrevious}
                  className="rounded-xl border-[hsl(172_30%_85%)]"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onNext}
                  disabled={!canGoNext}
                  className="rounded-xl border-[hsl(172_30%_85%)]"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>
        </DialogHeader>

        <div className="space-y-6 mt-6">
          {/* Heatmap toggle button */}
          {result.success && result.prediction && imageFile && (
            <div className="flex items-center justify-end">
              <Button
                onClick={handleToggleHeatmap}
                disabled={heatmapLoading}
                variant="outline"
                size="sm"
                className={`flex items-center gap-2 rounded-xl border-2 transition-all duration-300 ${
                  showHeatmap
                    ? 'bg-[hsl(172_63%_28%)] text-white border-[hsl(172_63%_28%)]'
                    : 'border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]'
                }`}
              >
                {heatmapLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Generating...
                  </>
                ) : showHeatmap ? (
                  <>
                    <EyeOff className="w-4 h-4" />
                    Hide Heatmap
                  </>
                ) : (
                  <>
                    <Eye className="w-4 h-4" />
                    Show Heatmap
                  </>
                )}
              </Button>
            </div>
          )}

          {/* Heatmap error */}
          {heatmapError && (
            <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-sm text-red-700">
              {heatmapError}
            </div>
          )}

          {/* Heatmap visualization */}
          {showHeatmap && heatmapBase64 && imageUrl && result.prediction && (
            <HeatmapOverlay
              originalImageUrl={imageUrl}
              heatmapBase64={heatmapBase64}
              prediction={result.prediction}
            />
          )}

          {/* Image preview (only shown when heatmap is hidden) */}
          {imageUrl && !showHeatmap && (
            <div className="relative w-full rounded-2xl bg-white/90 backdrop-blur-sm border-2 border-[hsl(172_30%_88%)] shadow-lg overflow-hidden">
              <div className="flex items-center justify-center p-8">
                <img
                  src={imageUrl}
                  alt={result.filename}
                  className="max-w-full max-h-[400px] object-contain rounded-xl shadow-xl"
                />
              </div>
            </div>
          )}

          {/* Error state */}
          {!result.success && result.error && (
            <div className="p-6 rounded-2xl bg-red-50 border-2 border-red-200 shadow-md">
              <div className="flex items-start gap-3">
                <X className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-lg font-semibold text-red-900 mb-1">
                    Processing Failed
                  </h3>
                  <p className="text-red-700">
                    {result.error}
                  </p>
                  <p className="text-sm text-red-600 mt-2">
                    Processing time: {result.processing_time_ms.toFixed(0)} ms
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Success state - prediction results */}
          {result.success && result.prediction && (
            <div>
              <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">
                Prediction Results
              </h3>
              <PredictionResult
                prediction={result.prediction}
                modelVersion="batch-analysis"
                processingTimeMs={result.processing_time_ms}
              />
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ResultDetailModal;
