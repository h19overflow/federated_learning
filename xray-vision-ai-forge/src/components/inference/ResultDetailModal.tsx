/**
 * ResultDetailModal Component
 *
 * Modal dialog that displays full prediction details for a single image.
 * Includes large image preview, prediction results, clinical interpretation,
 * and on-demand GradCAM heatmap generation.
 * Supports navigation between results with arrow keys and prev/next buttons.
 */

import React, { useEffect, useState } from 'react';
import { X, ChevronLeft, ChevronRight, Activity, Flame } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { SingleImageResult, HeatmapResponse } from '@/types/inference';
import { generateHeatmap } from '@/services/inferenceApi';
import { useToast } from '@/hooks/use-toast';
import PredictionResult from './PredictionResult';
import ClinicalInterpretation from './ClinicalInterpretation';
import HeatmapComparisonView from './HeatmapComparisonView';

interface ResultDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: SingleImageResult | null;
  imageUrl: string | null;
  /** Original File object for heatmap generation */
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
  const [heatmapData, setHeatmapData] = useState<HeatmapResponse | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const { toast } = useToast();

  // Reset heatmap state when result changes
  useEffect(() => {
    setHeatmapData(null);
    setShowHeatmap(false);
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

  const handleGenerateHeatmap = async () => {
    if (!imageFile) {
      toast({
        title: 'Cannot Generate Heatmap',
        description: 'Image file not available for heatmap generation.',
        variant: 'destructive',
      });
      return;
    }

    setHeatmapLoading(true);
    try {
      const response = await generateHeatmap(imageFile);
      setHeatmapData(response);
      setShowHeatmap(true);
      toast({
        title: 'Heatmap Generated',
        description: `GradCAM visualization ready (${response.processing_time_ms.toFixed(0)}ms)`,
      });
    } catch (error: any) {
      toast({
        title: 'Heatmap Generation Failed',
        description: error.message || 'Failed to generate GradCAM heatmap',
        variant: 'destructive',
      });
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
          {/* Image preview or Heatmap comparison */}
          {showHeatmap && heatmapData && imageUrl ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowHeatmap(false)}
                  className="rounded-xl border-[hsl(172_30%_85%)] text-[hsl(172_43%_25%)]"
                >
                  ← Back to Image
                </Button>
              </div>
              <HeatmapComparisonView
                originalImageUrl={imageUrl}
                heatmapBase64={heatmapData.heatmap_base64}
                predictionClass={result.prediction?.predicted_class}
              />
            </div>
          ) : (
            imageUrl && (
              <div className="relative w-full rounded-2xl bg-white/90 backdrop-blur-sm border-2 border-[hsl(172_30%_88%)] shadow-lg overflow-hidden">
                <div className="flex items-center justify-center p-8">
                  <img
                    src={imageUrl}
                    alt={result.filename}
                    className="max-w-full max-h-[400px] object-contain rounded-xl shadow-xl"
                  />
                </div>
                {/* Heatmap button overlay */}
                {result.success && imageFile && (
                  <div className="absolute bottom-4 right-4">
                    <Button
                      onClick={handleGenerateHeatmap}
                      disabled={heatmapLoading}
                      className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl shadow-lg"
                    >
                      {heatmapLoading ? (
                        <>
                          <Activity className="w-4 h-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Flame className="w-4 h-4 mr-2" />
                          Show Heatmap
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </div>
            )
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
                  <p className="text-red-700">{result.error}</p>
                  <p className="text-sm text-red-600 mt-2">
                    Processing time: {result.processing_time_ms.toFixed(0)} ms
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Success state - prediction results */}
          {result.success && result.prediction && (
            <div className="space-y-6">
              {/* Prediction card */}
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

              {/* Clinical interpretation */}
              {result.clinical_interpretation && (
                <div>
                  <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">
                    Clinical Analysis
                  </h3>
                  <ClinicalInterpretation
                    interpretation={result.clinical_interpretation}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ResultDetailModal;
