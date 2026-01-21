/**
 * BatchResultsGrid Component
 *
 * Grid gallery view for batch prediction results.
 * Features thumbnail grid with color-coded borders, filtering, sorting,
 * and detail modal on click.
 */

import React, { useState, useMemo } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ArrowUpDown,
} from "lucide-react";
import { SingleImageResult } from "@/types/inference";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import ResultDetailModal from "./ResultDetailModal";

interface BatchResultsGridProps {
  results: SingleImageResult[];
  imageUrls: Map<string, string>; // filename -> blob URL
  imageFiles?: Map<string, File>; // filename -> File object for heatmap generation
}

type FilterType = "all" | "normal" | "pneumonia" | "failed";
type SortType = "filename" | "confidence" | "time";

export const BatchResultsGrid: React.FC<BatchResultsGridProps> = ({
  results,
  imageUrls,
  imageFiles,
}) => {
  const [filter, setFilter] = useState<FilterType>("all");
  const [sortBy, setSortBy] = useState<SortType>("filename");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // Filter results
  const filteredResults = useMemo(() => {
    let filtered = [...results];

    switch (filter) {
      case "normal":
        filtered = filtered.filter(
          (r) => r.success && r.prediction?.predicted_class === "NORMAL",
        );
        break;
      case "pneumonia":
        filtered = filtered.filter(
          (r) => r.success && r.prediction?.predicted_class === "PNEUMONIA",
        );
        break;
      case "failed":
        filtered = filtered.filter((r) => !r.success);
        break;
      // 'all' - no filtering
    }

    return filtered;
  }, [results, filter]);

  // Sort results
  const sortedResults = useMemo(() => {
    const sorted = [...filteredResults];

    switch (sortBy) {
      case "filename":
        sorted.sort((a, b) => a.filename.localeCompare(b.filename));
        break;
      case "confidence":
        sorted.sort((a, b) => {
          const confA = a.prediction?.confidence ?? -1;
          const confB = b.prediction?.confidence ?? -1;
          return confB - confA; // Descending
        });
        break;
      case "time":
        sorted.sort((a, b) => a.processing_time_ms - b.processing_time_ms); // Ascending
        break;
    }

    return sorted;
  }, [filteredResults, sortBy]);

  const handleThumbnailClick = (index: number) => {
    setSelectedIndex(index);
  };

  const handleCloseModal = () => {
    setSelectedIndex(null);
  };

  const handleNext = () => {
    if (selectedIndex !== null && selectedIndex < sortedResults.length - 1) {
      setSelectedIndex(selectedIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (selectedIndex !== null && selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1);
    }
  };

  const selectedResult =
    selectedIndex !== null ? sortedResults[selectedIndex] : null;
  const selectedImageUrl = selectedResult
    ? (imageUrls.get(selectedResult.filename) ?? null)
    : null;
  const selectedImageFile =
    selectedResult && imageFiles
      ? (imageFiles.get(selectedResult.filename) ?? null)
      : null;

  // Get border color based on result
  const getBorderColor = (result: SingleImageResult) => {
    if (!result.success) {
      return "border-red-500"; // Failed - red
    }
    if (result.prediction?.predicted_class === "PNEUMONIA") {
      return "border-amber-500"; // Pneumonia - amber
    }
    return "border-green-500"; // Normal - green
  };

  // Get badge color
  const getBadgeColor = (result: SingleImageResult) => {
    if (!result.success) {
      return "bg-red-100 text-red-700 border-red-300";
    }
    if (result.prediction?.predicted_class === "PNEUMONIA") {
      return "bg-amber-100 text-amber-700 border-amber-300";
    }
    return "bg-green-100 text-green-700 border-green-300";
  };

  return (
    <div className="space-y-6">
      {/* Controls bar */}
      <div className="flex flex-col sm:flex-row gap-4 p-5 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(172_30%_88%)] shadow-md">
        {/* Filter buttons */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={filter === "all" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("all")}
            className={`rounded-xl ${
              filter === "all"
                ? "bg-[hsl(172_63%_28%)] text-white"
                : "border-[hsl(172_30%_85%)]"
            }`}
          >
            All ({results.length})
          </Button>
          <Button
            variant={filter === "normal" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("normal")}
            className={`rounded-xl ${
              filter === "normal"
                ? "bg-green-600 text-white"
                : "border-green-300 text-green-700 hover:bg-green-50"
            }`}
          >
            <CheckCircle2 className="w-4 h-4 mr-1" />
            Normal (
            {
              results.filter(
                (r) => r.success && r.prediction?.predicted_class === "NORMAL",
              ).length
            }
            )
          </Button>
          <Button
            variant={filter === "pneumonia" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("pneumonia")}
            className={`rounded-xl ${
              filter === "pneumonia"
                ? "bg-amber-600 text-white"
                : "border-amber-300 text-amber-700 hover:bg-amber-50"
            }`}
          >
            <AlertTriangle className="w-4 h-4 mr-1" />
            Pneumonia (
            {
              results.filter(
                (r) =>
                  r.success && r.prediction?.predicted_class === "PNEUMONIA",
              ).length
            }
            )
          </Button>
          <Button
            variant={filter === "failed" ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter("failed")}
            className={`rounded-xl ${
              filter === "failed"
                ? "bg-red-600 text-white"
                : "border-red-300 text-red-700 hover:bg-red-50"
            }`}
          >
            <XCircle className="w-4 h-4 mr-1" />
            Failed ({results.filter((r) => !r.success).length})
          </Button>
        </div>

        {/* Sort dropdown */}
        <div className="flex items-center gap-2 sm:ml-auto">
          <ArrowUpDown className="w-4 h-4 text-[hsl(172_63%_28%)]" />
          <Select
            value={sortBy}
            onValueChange={(value) => setSortBy(value as SortType)}
          >
            <SelectTrigger className="w-[160px] rounded-xl border-[hsl(172_30%_85%)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="filename">Sort by Name</SelectItem>
              <SelectItem value="confidence">Sort by Confidence</SelectItem>
              <SelectItem value="time">Sort by Time</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Results count */}
      <p className="text-sm text-[hsl(215_15%_45%)]">
        Showing {sortedResults.length} result
        {sortedResults.length !== 1 ? "s" : ""}
      </p>

      {/* Thumbnail grid */}
      {sortedResults.length > 0 ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {sortedResults.map((result, index) => {
            const imageUrl = imageUrls.get(result.filename);
            const borderColor = getBorderColor(result);
            const badgeColor = getBadgeColor(result);

            return (
              <div
                key={`${result.filename}-${index}`}
                onClick={() => handleThumbnailClick(index)}
                className={`
                  relative group aspect-square rounded-2xl bg-white/90 backdrop-blur-sm
                  border-4 ${borderColor} shadow-lg overflow-hidden
                  cursor-pointer hover:shadow-xl hover:scale-105
                  transition-all duration-300
                `}
              >
                {/* Image */}
                {imageUrl && (
                  <img
                    src={imageUrl}
                    alt={result.filename}
                    className="w-full h-full object-cover"
                  />
                )}

                {/* Overlay gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                {/* Info overlay */}
                <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/90 to-transparent">
                  {/* Filename */}
                  <p className="text-xs text-white font-medium truncate mb-1">
                    {result.filename}
                  </p>

                  {/* Badge */}
                  {result.success && result.prediction ? (
                    <div
                      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-semibold border ${badgeColor}`}
                    >
                      {result.prediction.predicted_class === "PNEUMONIA" ? (
                        <AlertTriangle className="w-3 h-3" />
                      ) : (
                        <CheckCircle2 className="w-3 h-3" />
                      )}
                      <span>
                        {(result.prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ) : (
                    <div
                      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-semibold border ${badgeColor}`}
                    >
                      <XCircle className="w-3 h-3" />
                      <span>Failed</span>
                    </div>
                  )}
                </div>

                {/* Processing time indicator (visible on hover) */}
                <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-white/95 backdrop-blur-sm text-xs font-medium text-[hsl(172_43%_20%)] opacity-0 group-hover:opacity-100 transition-opacity shadow-md">
                  {result.processing_time_ms.toFixed(0)} ms
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 px-6 rounded-2xl bg-white/60 backdrop-blur-sm border-2 border-dashed border-[hsl(172_30%_85%)]">
          <XCircle className="w-16 h-16 text-[hsl(215_15%_60%)] mb-4" />
          <p className="text-lg font-medium text-[hsl(172_43%_20%)]">
            No results match the selected filter
          </p>
          <p className="text-sm text-[hsl(215_15%_45%)] mt-1">
            Try selecting a different filter option
          </p>
        </div>
      )}

      {/* Detail modal */}
      <ResultDetailModal
        isOpen={selectedIndex !== null}
        onClose={handleCloseModal}
        result={selectedResult}
        imageUrl={selectedImageUrl}
        imageFile={selectedImageFile}
        onNext={handleNext}
        onPrevious={handlePrevious}
        canGoNext={
          selectedIndex !== null && selectedIndex < sortedResults.length - 1
        }
        canGoPrevious={selectedIndex !== null && selectedIndex > 0}
        currentIndex={selectedIndex ?? undefined}
        totalResults={sortedResults.length}
      />
    </div>
  );
};

export default BatchResultsGrid;
