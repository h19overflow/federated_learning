/**
 * BatchUploadZone Component
 *
 * Multi-file drag-and-drop upload zone for batch X-ray analysis.
 * Accepts up to 50 images (PNG/JPEG) with thumbnail previews.
 */

import React, { useCallback, useState } from "react";
import { Upload, Image as ImageIcon, X, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { validateImageFile } from "@/utils/validation";

interface BatchUploadZoneProps {
  onImagesSelect: (files: File[]) => void;
  selectedImages: File[];
  onClear: () => void;
  disabled?: boolean;
}

const MAX_FILES = 500;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB per file
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg"];

export const BatchUploadZone: React.FC<BatchUploadZoneProps> = ({
  onImagesSelect,
  selectedImages,
  onClear,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrls, setPreviewUrls] = useState<Map<string, string>>(
    new Map(),
  );

  const validateFiles = (
    files: File[],
  ): { valid: File[]; errors: string[] } => {
    const errors: string[] = [];
    const valid: File[] = [];

    files.forEach((file) => {
      const result = validateImageFile(file);
      if (result.valid) {
        valid.push(file);
      } else {
        errors.push(`${file.name}: ${result.error}`);
      }
    });

    return { valid, errors };
  };

  const handleFiles = useCallback(
    (newFiles: File[]) => {
      const currentCount = selectedImages.length;
      const totalCount = currentCount + newFiles.length;

      if (totalCount > MAX_FILES) {
        setError(
          `Maximum ${MAX_FILES} images allowed. You selected ${totalCount} images.`,
        );
        return;
      }

      const { valid, errors } = validateFiles(newFiles);

      if (errors.length > 0) {
        setError(errors.join("\n"));
        if (valid.length === 0) return;
      } else {
        setError(null);
      }

      // Create preview URLs for new files
      const newPreviews = new Map(previewUrls);
      valid.forEach((file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const result = e.target?.result as string;
          // Validate it's actually an image data URL
          if (result && result.startsWith('data:image/')) {
            newPreviews.set(file.name, result);
            setPreviewUrls(new Map(newPreviews));
          } else {
            setError(`Invalid image file: ${file.name}`);
          }
        };
        reader.readAsDataURL(file);
      });

      onImagesSelect([...selectedImages, ...valid]);
    },
    [selectedImages, onImagesSelect, previewUrls],
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
        handleFiles(files);
      }
    },
    [disabled, handleFiles],
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFiles(Array.from(files));
      }
      // Reset input so same files can be selected again
      e.target.value = "";
    },
    [handleFiles],
  );

  const handleRemoveFile = useCallback(
    (fileName: string) => {
      const newImages = selectedImages.filter((f) => f.name !== fileName);
      const newPreviews = new Map(previewUrls);
      newPreviews.delete(fileName);
      setPreviewUrls(newPreviews);
      onImagesSelect(newImages);
    },
    [selectedImages, previewUrls, onImagesSelect],
  );

  const handleClearAll = useCallback(() => {
    setPreviewUrls(new Map());
    setError(null);
    onClear();
  }, [onClear]);

  const totalSize = selectedImages.reduce((sum, file) => sum + file.size, 0);
  const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);

   // Show file list with thumbnails if images are selected
   if (selectedImages.length > 0) {
     return (
       <div className="space-y-4">
         {/* Header with stats */}
         <div className="flex items-center justify-between p-4 rounded-2xl bg-white border border-[hsl(172_30%_85%)] shadow-sm">
           <div className="flex items-center gap-3">
             <div className="w-10 h-10 rounded-xl bg-[hsl(172_35%_90%)] flex items-center justify-center">
               <ImageIcon className="w-5 h-5 text-[hsl(172_63%_28%)]" />
             </div>
             <div>
               <h3 className="text-sm font-semibold text-[hsl(172_43%_15%)]">
                 {selectedImages.length} Image{selectedImages.length !== 1 ? "s" : ""} Selected
               </h3>
               <p className="text-xs text-[hsl(215_15%_50%)]">
                 Total size: {totalSizeMB} MB
               </p>
             </div>
           </div>

           <Button
             variant="outline"
             size="sm"
             onClick={handleClearAll}
             disabled={disabled}
             className="border-[hsl(210_15%_85%)] text-[hsl(172_43%_20%)] hover:bg-red-50 hover:border-red-300 hover:text-red-600 rounded-xl transition-colors duration-200"
           >
             <Trash2 className="w-4 h-4 mr-1.5" />
             Clear All
           </Button>
         </div>

         {/* Thumbnail grid */}
         <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[400px] overflow-y-auto p-2 rounded-2xl bg-[hsl(172_30%_98%)]">
           {selectedImages.map((file) => (
             <div
               key={file.name}
               className="relative group aspect-square rounded-xl bg-white border border-[hsl(172_30%_85%)] shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-200"
             >
               {previewUrls.get(file.name) && (
                 <img
                   src={previewUrls.get(file.name)}
                   alt={file.name}
                   className="w-full h-full object-cover"
                 />
               )}

               {/* Overlay with filename */}
               <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/60 to-transparent">
                 <p className="text-xs text-white font-medium truncate">
                   {file.name}
                 </p>
                 <p className="text-xs text-white/80">
                   {(file.size / 1024).toFixed(1)} KB
                 </p>
               </div>

               {/* Remove button - always visible for clarity */}
               <button
                 onClick={() => handleRemoveFile(file.name)}
                 disabled={disabled}
                 className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-white shadow-sm flex items-center justify-center hover:bg-red-50 hover:text-red-600 transition-colors duration-200 disabled:opacity-50"
                 title="Remove image"
               >
                 <X className="w-4 h-4" />
               </button>
             </div>
           ))}
         </div>

         {/* Add more button */}
         {selectedImages.length < MAX_FILES && (
           <div>
             <input
               type="file"
               id="batch-upload-more"
               className="hidden"
               accept={ACCEPTED_TYPES.join(",")}
               onChange={handleFileInput}
               disabled={disabled}
               multiple
             />
             <label
               htmlFor="batch-upload-more"
               className={`
                 flex items-center justify-center gap-2 p-4 rounded-2xl border-2 border-dashed
                 transition-colors duration-200 cursor-pointer
                 ${
                   disabled
                     ? "opacity-50 cursor-not-allowed"
                     : "border-[hsl(172_30%_80%)] bg-white hover:bg-[hsl(172_30%_98%)] hover:border-[hsl(172_40%_70%)]"
                 }
               `}
             >
               <Upload className="w-5 h-5 text-[hsl(172_63%_28%)]" />
               <span className="text-sm font-medium text-[hsl(172_43%_20%)]">
                 Add More Images ({MAX_FILES - selectedImages.length} remaining)
               </span>
             </label>
           </div>
         )}

         {/* Error message */}
         {error && (
           <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm font-medium whitespace-pre-line">
             {error}
           </div>
         )}
       </div>
     );
   }

  // Show dropzone
  return (
    <div>
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
          id="batch-upload"
          className="hidden"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={handleFileInput}
          disabled={disabled}
          multiple
        />

        <label
          htmlFor="batch-upload"
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
            {isDragging ? "Drop images here" : "Upload Multiple X-Ray Images"}
          </h3>
          <p className="text-[hsl(215_15%_45%)] text-center mb-4 max-w-sm">
            Drag and drop up to {MAX_FILES} chest X-ray images here, or click to browse
          </p>

          {/* File requirements */}
          <div className="flex flex-wrap gap-2 justify-center">
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(172_30%_92%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(172_30%_85%)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
              PNG or JPEG
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(172_30%_92%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(172_30%_85%)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
              Max {MAX_FILES} images
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[hsl(172_30%_92%)] text-xs text-[hsl(172_43%_25%)] border border-[hsl(172_30%_85%)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_35%)]" />
              10MB per file
            </span>
          </div>
        </label>
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm font-medium whitespace-pre-line">
          {error}
        </div>
      )}
    </div>
  );
};

export default BatchUploadZone;
