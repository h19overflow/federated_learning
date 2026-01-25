/**
 * ImageDropzone Component
 *
 * Drag-and-drop upload zone for X-ray images with preview.
 * Accepts PNG/JPEG only with 10MB size limit.
 */

import React, { useCallback, useState } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { validateImageFile } from "@/utils/validation";

interface ImageDropzoneProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg"];

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  onImageSelect,
  selectedImage,
  onClear,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const validateFile = (file: File): string | null => {
    const result = validateImageFile(file);
    return result.valid ? null : result.error || 'Invalid file';
  };

  const handleFile = useCallback(
    (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }

      setError(null);
      onImageSelect(file);

      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        // Validate it's actually an image data URL
        if (result && result.startsWith('data:image/')) {
          setPreviewUrl(result);
        } else {
          setError('Invalid image file');
        }
      };
      reader.readAsDataURL(file);
    },
    [onImageSelect],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) {
        setIsDragging(true);
      }
    },
    [disabled],
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFile(files[0]);
      }
    },
    [disabled, handleFile],
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
    },
    [handleFile],
  );

  const handleClear = useCallback(() => {
    setPreviewUrl(null);
    setError(null);
    onClear();
  }, [onClear]);

  // Show preview if image is selected
  if (selectedImage && previewUrl) {
    return (
      <div className="relative w-full h-full min-h-[400px] rounded-3xl bg-white border-2 border-[hsl(172_35%_80%)] shadow-md overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center p-8 bg-gradient-to-br from-white via-[hsl(172_30%_98%)] to-white">
          <img
            src={previewUrl}
            alt="X-ray preview"
            className="max-w-full max-h-full object-contain rounded-2xl shadow-md"
          />
        </div>

        {/* File info overlay */}
        <div className="absolute top-4 left-4 right-4 flex items-start justify-between gap-3">
          <div className="bg-white border border-[hsl(172_30%_85%)] px-4 py-3 rounded-2xl shadow-sm">
            <div className="flex items-center gap-2 text-sm text-[hsl(172_43%_20%)]">
              <ImageIcon className="w-4 h-4 text-[hsl(172_63%_28%)] flex-shrink-0" />
              <div className="min-w-0">
                <p className="font-semibold truncate max-w-[180px]">
                  {selectedImage.name}
                </p>
                <p className="text-xs text-[hsl(215_15%_50%)]">
                  {(selectedImage.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleClear}
            disabled={disabled}
            className="bg-white border-[hsl(210_15%_85%)] text-[hsl(172_43%_20%)] hover:bg-red-50 hover:border-red-300 hover:text-red-600 rounded-xl transition-colors duration-200 flex-shrink-0"
          >
            <X className="w-4 h-4 mr-1" />
            Remove
          </Button>
        </div>
      </div>
    );
  }

  // Show dropzone
  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative w-full h-full min-h-[400px] rounded-3xl border-2 border-dashed
        transition-colors duration-200 cursor-pointer
        ${
          isDragging
            ? "border-[hsl(172_63%_28%)] bg-[hsl(172_35%_92%)]"
            : "border-[hsl(172_30%_80%)] bg-white hover:bg-[hsl(172_30%_98%)] hover:border-[hsl(172_40%_70%)]"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        type="file"
        id="xray-upload"
        className="hidden"
        accept={ACCEPTED_TYPES.join(",")}
        onChange={handleFileInput}
        disabled={disabled}
      />

      <label
        htmlFor="xray-upload"
        className="absolute inset-0 flex flex-col items-center justify-center p-8 cursor-pointer"
      >
        {/* Upload icon */}
        <div
          className={`
          mb-6 w-20 h-20 rounded-2xl flex items-center justify-center
          transition-colors duration-200
          ${
            isDragging
              ? "bg-[hsl(172_63%_28%)]"
              : "bg-[hsl(172_35%_90%)]"
          }
        `}
        >
          <Upload
            className={`
            w-10 h-10 transition-colors duration-200
            ${isDragging ? "text-white" : "text-[hsl(172_63%_28%)]"}
          `}
          />
        </div>

        {/* Instructions */}
        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-2">
          {isDragging ? "Drop image here" : "Upload X-Ray Image"}
        </h3>
        <p className="text-[hsl(215_15%_45%)] text-center mb-4 max-w-sm">
          Drag and drop your chest X-ray image here, or click to browse
        </p>

        {/* File requirements */}
        <div className="flex flex-wrap gap-2 justify-center">
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(172_30%_92%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(172_30%_85%)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
            PNG or JPEG
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(172_30%_92%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(172_30%_85%)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
            Max 10MB
          </span>
        </div>

        {/* Error message */}
        {error && (
          <div className="mt-4 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm font-medium">
            {error}
          </div>
        )}
      </label>
    </div>
  );
};

export default ImageDropzone;
