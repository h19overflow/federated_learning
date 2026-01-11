/**
 * ImageDropzone Component
 *
 * Drag-and-drop upload zone for X-ray images with preview.
 * Accepts PNG/JPEG only with 10MB size limit.
 */

import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImageDropzoneProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
  disabled?: boolean;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

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
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return 'Please upload a PNG or JPEG image file.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File size must be less than 10MB.';
    }
    return null;
  };

  const handleFile = useCallback((file: File) => {
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
      setPreviewUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, [onImageSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [disabled, handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  const handleClear = useCallback(() => {
    setPreviewUrl(null);
    setError(null);
    onClear();
  }, [onClear]);

  // Show preview if image is selected
  if (selectedImage && previewUrl) {
    return (
      <div className="relative w-full h-full min-h-[400px] rounded-3xl bg-white/90 backdrop-blur-sm border-2 border-[hsl(172_30%_88%)] shadow-lg overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center p-8">
          <img
            src={previewUrl}
            alt="X-ray preview"
            className="max-w-full max-h-full object-contain rounded-2xl shadow-xl"
          />
        </div>

        {/* File info overlay */}
        <div className="absolute top-4 left-4 right-4 flex items-start justify-between">
          <div className="bg-white/95 backdrop-blur-sm px-4 py-2 rounded-2xl shadow-md border border-[hsl(168_20%_90%)]">
            <div className="flex items-center gap-2 text-sm text-[hsl(172_43%_20%)]">
              <ImageIcon className="w-4 h-4 text-[hsl(172_63%_28%)]" />
              <span className="font-medium truncate max-w-[200px]">{selectedImage.name}</span>
              <span className="text-[hsl(215_15%_50%)]">
                ({(selectedImage.size / 1024).toFixed(1)} KB)
              </span>
            </div>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleClear}
            disabled={disabled}
            className="bg-white/95 backdrop-blur-sm border-[hsl(210_15%_88%)] hover:bg-red-50 hover:border-red-300 hover:text-red-600 rounded-xl transition-all"
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
        transition-all duration-300 cursor-pointer
        ${isDragging
          ? 'border-[hsl(172_63%_28%)] bg-[hsl(172_40%_95%)]'
          : 'border-[hsl(172_30%_85%)] bg-white/60 hover:bg-white/90 hover:border-[hsl(172_40%_75%)]'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        type="file"
        id="xray-upload"
        className="hidden"
        accept={ACCEPTED_TYPES.join(',')}
        onChange={handleFileInput}
        disabled={disabled}
      />

      <label
        htmlFor="xray-upload"
        className="absolute inset-0 flex flex-col items-center justify-center p-8 cursor-pointer"
      >
        {/* Upload icon with animation */}
        <div className={`
          mb-6 w-20 h-20 rounded-2xl flex items-center justify-center
          transition-all duration-300
          ${isDragging
            ? 'bg-[hsl(172_63%_28%)] scale-110'
            : 'bg-[hsl(172_40%_94%)]'
          }
        `}>
          <Upload className={`
            w-10 h-10 transition-colors
            ${isDragging ? 'text-white' : 'text-[hsl(172_63%_28%)]'}
          `} />
        </div>

        {/* Instructions */}
        <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-2">
          {isDragging ? 'Drop image here' : 'Upload X-Ray Image'}
        </h3>
        <p className="text-[hsl(215_15%_45%)] text-center mb-4 max-w-sm">
          Drag and drop your chest X-ray image here, or click to browse
        </p>

        {/* File requirements */}
        <div className="flex flex-wrap gap-2 justify-center">
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(168_25%_96%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(168_20%_90%)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
            PNG or JPEG
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(168_25%_96%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(168_20%_90%)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
            Max 10MB
          </span>
        </div>

        {/* Error message */}
        {error && (
          <div className="mt-4 px-4 py-2 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
            {error}
          </div>
        )}
      </label>
    </div>
  );
};

export default ImageDropzone;
