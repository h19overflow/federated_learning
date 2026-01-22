/**
 * Inference Page
 *
 * AI-powered chest X-ray analysis page with drag-and-drop upload,
 * real-time prediction, and GradCAM visualization.
 * Supports both single image and batch analysis modes.
 */

import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Activity,
  RefreshCw,
  Sparkles,
  Layers,
  Image as ImageIcon,
  Flame,
} from "lucide-react";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import ImageDropzone from "@/components/inference/ImageDropzone";
import PredictionResult from "@/components/inference/PredictionResult";
import InferenceStatusBadge from "@/components/inference/InferenceStatusBadge";
import BatchUploadZone from "@/components/inference/BatchUploadZone";
import BatchSummaryStats from "@/components/inference/BatchSummaryStats";
import BatchResultsGrid from "@/components/inference/BatchResultsGrid";
import BatchExportButton from "@/components/inference/BatchExportButton";
import HeatmapComparisonView from "@/components/inference/HeatmapComparisonView";
import {
  predictImage,
  batchPredictImages,
  generateHeatmap,
} from "@/services/inferenceApi";
import {
  InferenceResponse,
  BatchInferenceResponse,
  HeatmapResponse,
} from "@/types/inference";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "@/components/ui/progress";
import gsap from "gsap";

const ANIMATION_CONFIG = {
  duration: 0.8,
  staggerDelay: 0.15,
  ease: "power2.out",
} as const;

type AnalysisMode = "single" | "batch";

const Inference = () => {
  // Mode selection
  const [mode, setMode] = useState<AnalysisMode>("single");

  // Help guide state
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);

  // Single image state
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [singleHeatmap, setSingleHeatmap] = useState<HeatmapResponse | null>(
    null,
  );
  const [singleHeatmapLoading, setSingleHeatmapLoading] = useState(false);
  const [showSingleHeatmap, setShowSingleHeatmap] = useState(false);
  const singleImageUrlRef = useRef<string | null>(null);

  // Batch mode state
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [batchResult, setBatchResult] = useState<BatchInferenceResponse | null>(
    null,
  );
  const [batchLoading, setBatchLoading] = useState(false);
  const [imageUrls, setImageUrls] = useState<Map<string, string>>(new Map());
  const [imageFiles, setImageFiles] = useState<Map<string, File>>(new Map());

  const mainRef = useRef<HTMLElement>(null);
  const { toast } = useToast();

  // Single image handlers
  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setResult(null);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResult(null);
    setSingleHeatmap(null);
    setShowSingleHeatmap(false);
    // Cleanup single image blob URL
    if (singleImageUrlRef.current) {
      try {
        URL.revokeObjectURL(singleImageUrlRef.current);
      } catch (e) {
        console.error('Failed to revoke single image URL:', e);
      }
      singleImageUrlRef.current = null;
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const response = await predictImage(selectedImage);
      setResult(response);

      gsap.from(".result-card", {
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
        stagger: ANIMATION_CONFIG.staggerDelay,
      });

      toast({
        title: "Analysis Complete",
        description: `Prediction: ${response.prediction.predicted_class}`,
      });
    } catch (error: any) {
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to analyze image",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTryAnother = () => {
    handleClear();
    gsap.from(".upload-area", {
      opacity: 0,
      scale: 0.95,
      duration: 0.5,
      ease: "back.out(1.7)",
    });
  };

  const handleGenerateSingleHeatmap = async () => {
    if (!selectedImage) return;

    setSingleHeatmapLoading(true);
    try {
      const response = await generateHeatmap(selectedImage);
      setSingleHeatmap(response);
      setShowSingleHeatmap(true);
      // Create and store blob URL for the single image
      if (!singleImageUrlRef.current) {
        singleImageUrlRef.current = URL.createObjectURL(selectedImage);
      }
      toast({
        title: "Heatmap Generated",
        description: `GradCAM visualization ready (${response.processing_time_ms.toFixed(0)}ms)`,
      });
    } catch (error: any) {
      toast({
        title: "Heatmap Generation Failed",
        description: error.message || "Failed to generate GradCAM heatmap",
        variant: "destructive",
      });
    } finally {
      setSingleHeatmapLoading(false);
    }
  };

  // Batch mode handlers
  const handleImagesSelect = (files: File[]) => {
    setSelectedImages(files);
    // Cleanup old URLs
    imageUrls.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error('Failed to revoke URL:', e);
      }
    });
    const newUrls = new Map<string, string>();
    const newFiles = new Map<string, File>();
    files.forEach((file) => {
      const url = URL.createObjectURL(file);
      newUrls.set(file.name, url);
      newFiles.set(file.name, file);
    });
    setImageUrls(newUrls);
    setImageFiles(newFiles);
  };

  const handleBatchClear = () => {
    setSelectedImages([]);
    setBatchResult(null);
    // Properly revoke all URLs
    imageUrls.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error('Failed to revoke URL:', e);
      }
    });
    setImageUrls(new Map());
    setImageFiles(new Map());
  };

  const handleBatchPredict = async () => {
    if (selectedImages.length === 0) return;

    setBatchLoading(true);
    try {
      const response = await batchPredictImages(selectedImages);
      setBatchResult(response);

      gsap.from(".batch-results", {
        opacity: 0,
        y: 40,
        duration: ANIMATION_CONFIG.duration,
        ease: ANIMATION_CONFIG.ease,
        stagger: ANIMATION_CONFIG.staggerDelay,
      });

      toast({
        title: "Batch Analysis Complete",
        description: `Processed ${response.summary.total_images} images successfully`,
      });
    } catch (error: any) {
      toast({
        title: "Batch Analysis Failed",
        description: error.message || "Failed to analyze images",
        variant: "destructive",
      });
    } finally {
      setBatchLoading(false);
    }
  };

  const handleBatchTryAnother = () => {
    handleBatchClear();
    gsap.from(".batch-upload-area", {
      opacity: 0,
      scale: 0.95,
      duration: 0.5,
      ease: "back.out(1.7)",
    });
  };

  const handleModeSwitch = (newMode: AnalysisMode) => {
    setMode(newMode);
    if (newMode === "single") {
      handleBatchClear();
    } else {
      handleClear();
    }
  };

  // Cleanup blob URLs on unmount
  const imageUrlsRef = useRef<Map<string, string>>(new Map());

  // Update ref whenever imageUrls changes
  useEffect(() => {
    imageUrlsRef.current = imageUrls;
  }, [imageUrls]);

  useEffect(() => {
    return () => {
      // Cleanup batch image URLs
      imageUrlsRef.current.forEach((url) => {
        try {
          URL.revokeObjectURL(url);
        } catch (e) {
          console.error('Failed to revoke URL:', e);
        }
      });
      // Cleanup single image URL
      if (singleImageUrlRef.current) {
        try {
          URL.revokeObjectURL(singleImageUrlRef.current);
        } catch (e) {
          console.error('Failed to revoke single image URL:', e);
        }
      }
    };
  }, []);

  // GSAP entrance animations
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (prefersReducedMotion) return;

    const ctx = gsap.context(() => {
      const timeline = gsap.timeline({
        defaults: { ease: ANIMATION_CONFIG.ease },
      });

      timeline.from(".hero-badge", {
        opacity: 0,
        y: 20,
        duration: ANIMATION_CONFIG.duration,
      });

      timeline.from(
        ".hero-title",
        {
          opacity: 0,
          y: 30,
          duration: ANIMATION_CONFIG.duration,
        },
        "-=0.4",
      );

      timeline.from(
        ".hero-subtitle",
        {
          opacity: 0,
          y: 20,
          duration: ANIMATION_CONFIG.duration,
        },
        "-=0.3",
      );

      timeline.from(
        ".hero-mode-toggle",
        {
          opacity: 0,
          y: 20,
          duration: ANIMATION_CONFIG.duration,
        },
        "-=0.2",
      );

      timeline.from(
        ".content-card",
        {
          opacity: 0,
          y: 60,
          stagger: ANIMATION_CONFIG.staggerDelay,
          duration: ANIMATION_CONFIG.duration,
        },
        "-=0.2",
      );
    });

    return () => ctx.revert();
  }, []);

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

      <main ref={mainRef} className="flex-1 overflow-y-auto bg-hero-gradient">
        {/* Hero Section */}
        <section className="relative py-16 px-6 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-[hsl(172_40%_85%)] rounded-full blur-[120px] opacity-30" />
            <div className="absolute bottom-1/4 right-1/4 w-[350px] h-[350px] bg-[hsl(210_60%_90%)] rounded-full blur-[100px] opacity-25" />
          </div>

          <div className="relative z-10 max-w-4xl mx-auto text-center">
            <div className="hero-badge inline-flex items-center gap-2 px-4 py-2 mb-6 rounded-full bg-white/60 backdrop-blur-sm border border-[hsl(172_30%_85%)] shadow-sm">
              <Sparkles className="w-4 h-4 text-[hsl(172_63%_28%)]" />
              <span className="text-sm font-medium text-[hsl(172_43%_25%)]">
                AI-Powered Diagnostics
              </span>
            </div>

            <h1 className="hero-title text-4xl md:text-5xl font-semibold tracking-tight text-[hsl(172_43%_15%)] mb-4">
              Chest X-Ray Analysis
            </h1>

            <p className="hero-subtitle text-lg md:text-xl text-[hsl(215_15%_45%)] max-w-2xl mx-auto mb-8">
              Upload a chest X-ray image to detect pneumonia using our
              state-of-the-art AI model.
            </p>

            <div className="hero-mode-toggle inline-flex items-center gap-1 p-1.5 rounded-2xl bg-white/80 backdrop-blur-sm border border-[hsl(172_30%_85%)] shadow-lg">
              <button
                onClick={() => handleModeSwitch("single")}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  mode === "single"
                    ? "bg-[hsl(172_63%_28%)] text-white shadow-md"
                    : "text-[hsl(172_43%_25%)] hover:bg-white/60"
                }`}
              >
                <ImageIcon className="w-5 h-5" />
                <span>Single Image</span>
              </button>
              <button
                onClick={() => handleModeSwitch("batch")}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  mode === "batch"
                    ? "bg-[hsl(172_63%_28%)] text-white shadow-md"
                    : "text-[hsl(172_43%_25%)] hover:bg-white/60"
                }`}
              >
                <Layers className="w-5 h-5" />
                <span>Batch Analysis</span>
              </button>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <section className="pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Single Image Mode */}
            {mode === "single" && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column - Upload */}
                <div className="content-card upload-area">
                  <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-2">
                      Upload Image
                    </h2>
                    <p className="text-[hsl(215_15%_45%)]">
                      Drag and drop or click to upload a chest X-ray image
                    </p>
                  </div>

                  <ImageDropzone
                    onImageSelect={handleImageSelect}
                    selectedImage={selectedImage}
                    onClear={handleClear}
                    disabled={loading}
                  />

                  {selectedImage && !result && (
                    <div className="mt-6 space-y-4">
                      <Button
                        onClick={handlePredict}
                        disabled={loading}
                        size="lg"
                        className="w-full bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white py-7 rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 hover:shadow-xl transition-all duration-300"
                      >
                        {loading ? (
                          <>
                            <Activity className="mr-2 h-5 w-5 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Sparkles className="mr-2 h-5 w-5" />
                            Analyze Image
                          </>
                        )}
                      </Button>
                    </div>
                  )}

                  {result && (
                    <div className="mt-6">
                      <Button
                        onClick={handleTryAnother}
                        variant="outline"
                        size="lg"
                        className="w-full py-7 rounded-2xl border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] transition-all duration-300"
                      >
                        <RefreshCw className="mr-2 h-5 w-5" />
                        Analyze Another Image
                      </Button>
                    </div>
                  )}
                </div>

                {/* Right Column - Results */}
                <div className="content-card">
                  {loading && (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] space-y-6">
                      <div className="w-20 h-20 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                        <Activity className="w-10 h-10 text-[hsl(172_63%_28%)] animate-spin" />
                      </div>
                      <div className="text-center space-y-2">
                        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
                          Analyzing X-Ray...
                        </h3>
                        <p className="text-[hsl(215_15%_45%)]">
                          Running AI model inference
                        </p>
                      </div>
                      <div className="w-full max-w-md space-y-3">
                        {[...Array(3)].map((_, i) => (
                          <div
                            key={i}
                            className="h-12 bg-[hsl(168_25%_96%)] rounded-2xl animate-pulse"
                            style={{ animationDelay: `${i * 0.1}s` }}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {!loading && !result && (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] space-y-4">
                      <div className="w-20 h-20 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                        <svg
                          className="w-10 h-10 text-[hsl(172_63%_28%)]"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="17 8 12 3 7 8" />
                          <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                      </div>
                      <div className="text-center space-y-2">
                        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
                          Upload an X-Ray to Begin
                        </h3>
                        <p className="text-[hsl(215_15%_45%)] max-w-xs">
                          Prediction results and visualizations will appear here
                        </p>
                      </div>
                    </div>
                  )}

                  {!loading && result && (
                    <div>
                      <div className="mb-6">
                        <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-2">
                          Analysis Results
                        </h2>
                        <p className="text-[hsl(215_15%_45%)]">
                          AI-powered pneumonia detection analysis
                        </p>
                      </div>

                      <div className="space-y-6">
                        <div className="result-card">
                          <PredictionResult
                            prediction={result.prediction}
                            modelVersion={result.model_version}
                            processingTimeMs={result.processing_time_ms}
                          />
                        </div>

                        {/* GradCAM Heatmap Section */}
                        <div className="result-card">
                          {showSingleHeatmap &&
                          singleHeatmap &&
                          selectedImage ? (
                            <div className="space-y-4">
                              <div className="flex items-center justify-between">
                                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
                                  GradCAM Visualization
                                </h3>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setShowSingleHeatmap(false)}
                                  className="rounded-xl border-[hsl(172_30%_85%)] text-[hsl(172_43%_25%)]"
                                >
                                  Hide Heatmap
                                </Button>
                              </div>
                              {singleImageUrlRef.current && (
                                <HeatmapComparisonView
                                  originalImageUrl={singleImageUrlRef.current}
                                  heatmapBase64={singleHeatmap.heatmap_base64}
                                  predictionClass={
                                    result.prediction.predicted_class
                                  }
                                />
                              )}
                            </div>
                          ) : (
                            <div className="p-6 rounded-2xl bg-white/80 border border-[hsl(172_30%_88%)]">
                              <div className="flex items-center justify-between">
                                <div>
                                  <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
                                    GradCAM Heatmap
                                  </h3>
                                  <p className="text-sm text-[hsl(215_15%_45%)]">
                                    Visualize which regions influenced the
                                    model's prediction
                                  </p>
                                </div>
                                <Button
                                  onClick={handleGenerateSingleHeatmap}
                                  disabled={singleHeatmapLoading}
                                  className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl"
                                >
                                  {singleHeatmapLoading ? (
                                    <>
                                      <Activity className="w-4 h-4 mr-2 animate-spin" />
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
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Batch Analysis Mode */}
            {mode === "batch" && (
              <div className="space-y-8">
                <div className="content-card batch-upload-area">
                  <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-2">
                      Upload Multiple Images
                    </h2>
                    <p className="text-[hsl(215_15%_45%)]">
                      Upload up to 500 chest X-ray images for batch analysis
                    </p>
                  </div>

                  <BatchUploadZone
                    onImagesSelect={handleImagesSelect}
                    selectedImages={selectedImages}
                    onClear={handleBatchClear}
                    disabled={batchLoading}
                  />

                  {selectedImages.length > 0 && !batchResult && (
                    <div className="mt-6 space-y-4">
                      <Button
                        onClick={handleBatchPredict}
                        disabled={batchLoading}
                        size="lg"
                        className="w-full bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white py-7 rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 hover:shadow-xl transition-all duration-300"
                      >
                        {batchLoading ? (
                          <>
                            <Activity className="mr-2 h-5 w-5 animate-spin" />
                            Analyzing {selectedImages.length} images...
                          </>
                        ) : (
                          <>
                            <Sparkles className="mr-2 h-5 w-5" />
                            Analyze {selectedImages.length} Image
                            {selectedImages.length !== 1 ? "s" : ""}
                          </>
                        )}
                      </Button>
                    </div>
                  )}

                  {batchResult && (
                    <div className="mt-6">
                      <Button
                        onClick={handleBatchTryAnother}
                        variant="outline"
                        size="lg"
                        className="w-full py-7 rounded-2xl border-2 border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)] transition-all duration-300"
                      >
                        <RefreshCw className="mr-2 h-5 w-5" />
                        Analyze Another Batch
                      </Button>
                    </div>
                  )}
                </div>

                {batchLoading && (
                  <div className="content-card batch-results">
                    <div className="flex flex-col items-center justify-center py-16 space-y-6">
                      <div className="w-20 h-20 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                        <Activity className="w-10 h-10 text-[hsl(172_63%_28%)] animate-spin" />
                      </div>
                      <div className="text-center space-y-2">
                        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
                          Processing Batch Analysis...
                        </h3>
                        <p className="text-[hsl(215_15%_45%)]">
                          Analyzing {selectedImages.length} chest X-ray images
                        </p>
                      </div>
                      <div className="w-full max-w-md">
                        <Progress
                          value={undefined}
                          className="h-3 bg-[hsl(168_25%_96%)]"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {!batchLoading && batchResult && (
                  <div className="space-y-8">
                    <div className="content-card batch-results">
                      <BatchSummaryStats summary={batchResult.summary} />
                    </div>

                    <div className="content-card batch-results flex items-center justify-between p-6">
                      <div>
                        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">
                          Export Results
                        </h3>
                        <p className="text-sm text-[hsl(215_15%_45%)]">
                          Download batch analysis data in CSV, JSON, or PDF
                          format
                        </p>
                      </div>
                      <BatchExportButton
                        data={batchResult}
                        imageFiles={imageFiles}
                      />
                    </div>

                    <div className="content-card batch-results">
                      <div className="mb-6">
                        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-2">
                          Detailed Results
                        </h3>
                        <p className="text-[hsl(215_15%_45%)]">
                          Click on any image to view full prediction details
                        </p>
                      </div>
                      <BatchResultsGrid
                        results={batchResult.results}
                        imageUrls={imageUrls}
                        imageFiles={imageFiles}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Inference;
