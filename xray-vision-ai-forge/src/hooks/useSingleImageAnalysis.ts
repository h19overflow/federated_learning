/**
 * Single Image Analysis Hook
 * Manages state and handlers for single image prediction workflow
 */

import { useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { predictImage, generateHeatmap } from "@/services/inferenceApi";
import {
  InferenceResponse,
  HeatmapResponse,
} from "@/types/inference";
import gsap from "gsap";
import { resultCardAnimation } from "@/utils/animationConfigs";

export const useSingleImageAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [singleHeatmap, setSingleHeatmap] = useState<HeatmapResponse | null>(
    null,
  );
  const [singleHeatmapLoading, setSingleHeatmapLoading] = useState(false);
  const [showSingleHeatmap, setShowSingleHeatmap] = useState(false);
  const singleImageUrlRef = useRef<string | null>(null);
  const { toast } = useToast();

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
        console.error("Failed to revoke single image URL:", e);
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
        opacity: resultCardAnimation.opacity,
        y: resultCardAnimation.y,
        duration: resultCardAnimation.duration,
        ease: resultCardAnimation.ease,
        stagger: resultCardAnimation.stagger,
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

  const toggleHeatmapVisibility = () => {
    setShowSingleHeatmap(!showSingleHeatmap);
  };

  return {
    selectedImage,
    result,
    loading,
    singleHeatmap,
    singleHeatmapLoading,
    showSingleHeatmap,
    singleImageUrlRef,
    handleImageSelect,
    handleClear,
    handlePredict,
    handleTryAnother,
    handleGenerateSingleHeatmap,
    toggleHeatmapVisibility,
  };
};
