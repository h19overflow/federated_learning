/**
 * Inference Page
 *
 * AI-powered chest X-ray analysis page with drag-and-drop upload,
 * real-time prediction, and GradCAM visualization.
 * Supports both single image and batch analysis modes.
 */

import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import InferenceStatusBadge from "@/components/inference/InferenceStatusBadge";
import HeroSection, { AnalysisMode } from "@/components/inference/HeroSection";
import SingleImageAnalysis from "@/components/inference/SingleImageAnalysis";
import BatchAnalysis from "@/components/inference/BatchAnalysis";
import { useSingleImageAnalysis } from "@/hooks/useSingleImageAnalysis";
import { useBatchAnalysis } from "@/hooks/useBatchAnalysis";
import { useInferenceAnimations } from "@/hooks/useInferenceAnimations";
import { useImageUrlCleanup } from "@/hooks/useImageUrlCleanup";

const Inference = () => {
  const [mode, setMode] = useState<AnalysisMode>("single");
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);

  // Single image analysis hook
  const singleAnalysis = useSingleImageAnalysis();

  // Batch analysis hook
  const batchAnalysis = useBatchAnalysis();

  // Animation hook
  const { playHeroEntrance, cleanup: cleanupAnimations } =
    useInferenceAnimations();

  // URL cleanup hook
  const { revokeAll } = useImageUrlCleanup();

  // Play entrance animations on mount
  useEffect(() => {
    playHeroEntrance();
    return () => cleanupAnimations();
  }, []);

  // Cleanup all URLs on unmount
  useEffect(() => {
    return () => {
      singleAnalysis.singleImageUrlRef.current &&
        URL.revokeObjectURL(singleAnalysis.singleImageUrlRef.current);
      batchAnalysis.cleanupAllUrls();
      revokeAll();
    };
  }, []);

  const handleModeSwitch = (newMode: AnalysisMode) => {
    setMode(newMode);
    if (newMode === "single") {
      batchAnalysis.handleBatchClear();
    } else {
      singleAnalysis.handleClear();
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header onShowHelp={() => setShowWelcomeGuide(true)} />
      {showWelcomeGuide && (
        <WelcomeGuide
          onClose={() => setShowWelcomeGuide(false)}
          initialGuide="inference"
        />
      )}
      <InferenceStatusBadge />

      <main className="flex-1 overflow-y-auto bg-hero-gradient">
        <HeroSection mode={mode} onModeChange={handleModeSwitch} />

        <section className="pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            <AnimatePresence mode="wait">
              {mode === "single" ? (
                <motion.div
                  key="single"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <SingleImageAnalysis
                    selectedImage={singleAnalysis.selectedImage}
                    result={singleAnalysis.result}
                    loading={singleAnalysis.loading}
                    singleHeatmap={singleAnalysis.singleHeatmap}
                    singleHeatmapLoading={singleAnalysis.singleHeatmapLoading}
                    showSingleHeatmap={singleAnalysis.showSingleHeatmap}
                    singleImageUrlRef={singleAnalysis.singleImageUrlRef}
                    onImageSelect={singleAnalysis.handleImageSelect}
                    onClear={singleAnalysis.handleClear}
                    onPredict={singleAnalysis.handlePredict}
                    onTryAnother={singleAnalysis.handleTryAnother}
                    onGenerateSingleHeatmap={
                      singleAnalysis.handleGenerateSingleHeatmap
                    }
                    onToggleHeatmap={singleAnalysis.toggleHeatmapVisibility}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="batch"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <BatchAnalysis
                    selectedImages={batchAnalysis.selectedImages}
                    batchResult={batchAnalysis.batchResult}
                    batchLoading={batchAnalysis.batchLoading}
                    imageUrls={batchAnalysis.imageUrls}
                    imageFiles={batchAnalysis.imageFiles}
                    onImagesSelect={batchAnalysis.handleImagesSelect}
                    onClear={batchAnalysis.handleBatchClear}
                    onBatchPredict={batchAnalysis.handleBatchPredict}
                    onBatchTryAnother={batchAnalysis.handleBatchTryAnother}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Inference;
