/**
 * ResultDetailModal Component
 *
 * Modal dialog that displays full prediction details for a single image.
 * Includes large image preview, prediction results, and clinical interpretation.
 * Supports navigation between results with arrow keys and prev/next buttons.
 */

import React, { useEffect } from 'react';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { SingleImageResult } from '@/types/inference';
import PredictionResult from './PredictionResult';
import ClinicalInterpretation from './ClinicalInterpretation';

interface ResultDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: SingleImageResult | null;
  imageUrl: string | null;
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
  onNext,
  onPrevious,
  canGoNext = false,
  canGoPrevious = false,
  currentIndex,
  totalResults,
}) => {
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
          {/* Image preview */}
          {imageUrl && (
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
