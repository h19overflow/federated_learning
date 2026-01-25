/**
 * Results Panel Component
 * Right column results display for single image analysis
 */

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import SectionHeader from "@/components/inference/SectionHeader";
import PredictionResult from "@/components/inference/PredictionResult";
import HeatmapSection from "@/components/inference/HeatmapSection";
import LoadingState from "@/components/inference/LoadingState";
import EmptyState from "@/components/inference/EmptyState";
import { InferenceResponse, HeatmapResponse } from "@/types/inference";

interface ResultsPanelProps {
  loading: boolean;
  result: InferenceResponse | null;
  showHeatmap: boolean;
  heatmap: HeatmapResponse | null;
  heatmapLoading: boolean;
  imageUrl: string | null;
  onToggleHeatmap: (show: boolean) => void;
  onGenerateHeatmap: () => void;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({
  loading,
  result,
  showHeatmap,
  heatmap,
  heatmapLoading,
  imageUrl,
  onToggleHeatmap,
  onGenerateHeatmap,
}) => {
  return (
    <div className="content-card overflow-hidden">
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ duration: 0.3 }}
          >
            <LoadingState
              title="Analyzing X-Ray..."
              description="Running AI model inference"
            />
          </motion.div>
        ) : !result ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <EmptyState
              title="Upload an X-Ray to Begin"
              description="Prediction results and visualizations will appear here"
            />
          </motion.div>
        ) : (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <SectionHeader
              title="Analysis Results"
              description="AI-powered pneumonia detection analysis"
            />

            <div className="space-y-6">
              <motion.div
                className="result-card"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2, duration: 0.4 }}
              >
                <PredictionResult
                  prediction={result.prediction}
                  modelVersion={result.model_version}
                  processingTimeMs={result.processing_time_ms}
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.4 }}
              >
                <HeatmapSection
                  showHeatmap={showHeatmap}
                  heatmap={heatmap}
                  heatmapLoading={heatmapLoading}
                  imageUrl={imageUrl}
                  predictionClass={result.prediction.predicted_class}
                  onToggleHeatmap={onToggleHeatmap}
                  onGenerateHeatmap={onGenerateHeatmap}
                />
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ResultsPanel;
