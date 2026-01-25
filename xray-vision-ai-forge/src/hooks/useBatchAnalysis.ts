/**
 * Batch Analysis Hook
 * Manages state and handlers for batch image prediction workflow
 */

import { useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { batchPredictImages } from "@/services/inferenceApi";
import { BatchInferenceResponse } from "@/types/inference";
import gsap from "gsap";
import { batchResultsAnimation, uploadAreaResetAnimation } from "@/utils/animationConfigs";

export const useBatchAnalysis = () => {
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [batchResult, setBatchResult] = useState<BatchInferenceResponse | null>(
    null,
  );
  const [batchLoading, setBatchLoading] = useState(false);
  const [imageUrls, setImageUrls] = useState<Map<string, string>>(new Map());
  const [imageFiles, setImageFiles] = useState<Map<string, File>>(new Map());
  const imageUrlsRef = useRef<Map<string, string>>(new Map());
  const { toast } = useToast();

  // Update ref whenever imageUrls changes
  const updateImageUrlsRef = (urls: Map<string, string>) => {
    imageUrlsRef.current = urls;
  };

  const handleImagesSelect = (files: File[]) => {
    setSelectedImages(files);
    // Cleanup old URLs
    imageUrls.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Failed to revoke URL:", e);
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
    updateImageUrlsRef(newUrls);
  };

  const handleBatchClear = () => {
    setSelectedImages([]);
    setBatchResult(null);
    // Properly revoke all URLs
    imageUrls.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Failed to revoke URL:", e);
      }
    });
    setImageUrls(new Map());
    setImageFiles(new Map());
    updateImageUrlsRef(new Map());
  };

  const handleBatchPredict = async () => {
    if (selectedImages.length === 0) return;

    setBatchLoading(true);
    try {
      const response = await batchPredictImages(selectedImages);
      setBatchResult(response);

      gsap.from(".batch-results", {
        opacity: batchResultsAnimation.opacity,
        y: batchResultsAnimation.y,
        duration: batchResultsAnimation.duration,
        ease: batchResultsAnimation.ease,
        stagger: batchResultsAnimation.stagger,
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
      opacity: uploadAreaResetAnimation.opacity,
      scale: uploadAreaResetAnimation.scale,
      duration: uploadAreaResetAnimation.duration,
      ease: uploadAreaResetAnimation.ease,
    });
  };

  const cleanupAllUrls = () => {
    imageUrlsRef.current.forEach((url) => {
      try {
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Failed to revoke URL:", e);
      }
    });
  };

  return {
    selectedImages,
    batchResult,
    batchLoading,
    imageUrls,
    imageFiles,
    imageUrlsRef,
    handleImagesSelect,
    handleBatchClear,
    handleBatchPredict,
    handleBatchTryAnother,
    cleanupAllUrls,
  };
};
