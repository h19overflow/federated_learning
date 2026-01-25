/**
 * Upload Panel Component
 * Left column upload area for single image analysis
 */

import React from "react";
import SectionHeader from "@/components/inference/SectionHeader";
import ImageDropzone from "@/components/inference/ImageDropzone";
import AnalysisButton from "@/components/inference/AnalysisButton";

interface UploadPanelProps {
  selectedImage: File | null;
  result: boolean;
  loading: boolean;
  onImageSelect: (file: File) => void;
  onClear: () => void;
  onAnalyze: () => void;
  onTryAnother: () => void;
}

const UploadPanel: React.FC<UploadPanelProps> = ({
  selectedImage,
  result,
  loading,
  onImageSelect,
  onClear,
  onAnalyze,
  onTryAnother,
}) => {
  return (
    <div className="content-card upload-area">
      <SectionHeader
        title="Upload Image"
        description="Drag and drop or click to upload a chest X-ray image"
      />

      <ImageDropzone
        onImageSelect={onImageSelect}
        selectedImage={selectedImage}
        onClear={onClear}
        disabled={loading}
      />

      {selectedImage && !result && (
        <div className="mt-6 space-y-4">
          <AnalysisButton
            onClick={onAnalyze}
            loading={loading}
            disabled={loading}
          />
        </div>
      )}

      {result && (
        <div className="mt-6">
          <AnalysisButton onClick={onTryAnother} variant="retry" />
        </div>
      )}
    </div>
  );
};

export default UploadPanel;
