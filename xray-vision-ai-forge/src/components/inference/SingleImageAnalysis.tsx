/**
 * Single Image Analysis Component
 * Complete single image analysis workflow with upload and results
 */

import React from "react";
import { motion } from "framer-motion";
import UploadPanel from "@/components/inference/UploadPanel";
import ResultsPanel from "@/components/inference/ResultsPanel";
import { InferenceResponse, HeatmapResponse } from "@/types/inference";

interface SingleImageAnalysisProps {
  selectedImage: File | null;
  result: InferenceResponse | null;
  loading: boolean;
  singleHeatmap: HeatmapResponse | null;
  singleHeatmapLoading: boolean;
  showSingleHeatmap: boolean;
  singleImageUrlRef: React.MutableRefObject<string | null>;
  onImageSelect: (file: File) => void;
  onClear: () => void;
  onPredict: () => void;
  onTryAnother: () => void;
  onGenerateSingleHeatmap: () => void;
  onToggleHeatmap: (show: boolean) => void;
}

const SingleImageAnalysis: React.FC<SingleImageAnalysisProps> = ({
  selectedImage,
  result,
  loading,
  singleHeatmap,
  singleHeatmapLoading,
  showSingleHeatmap,
  singleImageUrlRef,
  onImageSelect,
  onClear,
  onPredict,
  onTryAnother,
  onGenerateSingleHeatmap,
  onToggleHeatmap,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <UploadPanel
          selectedImage={selectedImage}
          result={!!result}
          loading={loading}
          onImageSelect={onImageSelect}
          onClear={onClear}
          onAnalyze={onPredict}
          onTryAnother={onTryAnother}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <ResultsPanel
          loading={loading}
          result={result}
          showHeatmap={showSingleHeatmap}
          heatmap={singleHeatmap}
          heatmapLoading={singleHeatmapLoading}
          imageUrl={singleImageUrlRef.current}
          onToggleHeatmap={onToggleHeatmap}
          onGenerateHeatmap={onGenerateSingleHeatmap}
        />
      </motion.div>
    </div>
  );
};

export default SingleImageAnalysis;
