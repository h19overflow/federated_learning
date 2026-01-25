/**
 * Batch Analysis Component
 * Complete batch image analysis workflow with upload and results
 */

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import SectionHeader from "@/components/inference/SectionHeader";
import BatchUploadZone from "@/components/inference/BatchUploadZone";
import AnalysisButton from "@/components/inference/AnalysisButton";
import LoadingState from "@/components/inference/LoadingState";
import BatchResultsSection from "@/components/inference/BatchResultsSection";
import { BatchInferenceResponse } from "@/types/inference";

interface BatchAnalysisProps {
  selectedImages: File[];
  batchResult: BatchInferenceResponse | null;
  batchLoading: boolean;
  imageUrls: Map<string, string>;
  imageFiles: Map<string, File>;
  onImagesSelect: (files: File[]) => void;
  onClear: () => void;
  onBatchPredict: () => void;
  onBatchTryAnother: () => void;
}

const BatchAnalysis: React.FC<BatchAnalysisProps> = ({
  selectedImages,
  batchResult,
  batchLoading,
  imageUrls,
  imageFiles,
  onImagesSelect,
  onClear,
  onBatchPredict,
  onBatchTryAnother,
}) => {
  return (
    <div className="space-y-8">
      <motion.div
        className="content-card batch-upload-area"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <SectionHeader
          title="Upload Multiple Images"
          description="Upload up to 500 chest X-ray images for batch analysis"
        />

        <BatchUploadZone
          onImagesSelect={onImagesSelect}
          selectedImages={selectedImages}
          onClear={onClear}
          disabled={batchLoading}
        />

        <AnimatePresence mode="wait">
          {selectedImages.length > 0 && !batchResult && (
            <motion.div
              key="analyze-btn"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6 space-y-4 overflow-hidden"
            >
              <AnalysisButton
                onClick={onBatchPredict}
                loading={batchLoading}
                disabled={batchLoading}
                imageCount={selectedImages.length}
              />
            </motion.div>
          )}

          {batchResult && (
            <motion.div
              key="retry-btn"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6 overflow-hidden"
            >
              <AnalysisButton
                onClick={onBatchTryAnother}
                variant="retry"
                imageCount={selectedImages.length}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      <AnimatePresence>
        {batchLoading && (
          <motion.div
            className="content-card batch-results"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <LoadingState
              title="Processing Batch Analysis..."
              description={`Analyzing ${selectedImages.length} chest X-ray images`}
              minHeight="py-16"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {!batchLoading && batchResult && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, type: "spring" }}
        >
          <BatchResultsSection
            batchResult={batchResult}
            imageUrls={imageUrls}
            imageFiles={imageFiles}
          />
        </motion.div>
      )}
    </div>
  );
};

export default BatchAnalysis;
