/**
 * BatchExportButton Component
 *
 * Dropdown button that exports batch prediction results in multiple formats.
 * Supports CSV (for spreadsheet analysis), JSON (for full data export),
 * and PDF reports with optional GradCAM heatmaps.
 */

import React, { useState } from "react";
import {
  Download,
  FileSpreadsheet,
  FileJson,
  FileText,
  Loader2,
  Flame,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  BatchInferenceResponse,
  BatchHeatmapResponse,
} from "@/types/inference";
import {
  generateBatchPdfReport,
  generateBatchHeatmaps,
  generateBatchPdfReportWithHeatmaps,
  downloadBlob,
} from "@/services/inferenceApi";
import { useToast } from "@/hooks/use-toast";

interface BatchExportButtonProps {
  data: BatchInferenceResponse;
  imageFiles?: Map<string, File>;
  disabled?: boolean;
}

export const BatchExportButton: React.FC<BatchExportButtonProps> = ({
  data,
  imageFiles,
  disabled = false,
}) => {
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfHeatmapLoading, setPdfHeatmapLoading] = useState(false);
  const { toast } = useToast();

  const downloadFile = (
    content: string,
    filename: string,
    mimeType: string,
  ) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:image/png;base64,")
        const base64 = result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleExportCSV = () => {
    // CSV headers
    const headers = [
      "Filename",
      "Status",
      "Prediction",
      "Confidence",
      "Pneumonia Probability",
      "Normal Probability",
      "Risk Level",
      "Processing Time (ms)",
      "Error",
    ];

    // CSV rows
    const rows = data.results.map((result) => {
      const prediction = result.prediction;
      const riskLevel =
        result.clinical_interpretation?.risk_assessment?.risk_level ?? "N/A";

      return [
        result.filename,
        result.success ? "Success" : "Failed",
        prediction?.predicted_class ?? "N/A",
        prediction ? (prediction.confidence * 100).toFixed(2) : "N/A",
        prediction
          ? (prediction.pneumonia_probability * 100).toFixed(2)
          : "N/A",
        prediction ? (prediction.normal_probability * 100).toFixed(2) : "N/A",
        riskLevel,
        result.processing_time_ms.toFixed(2),
        result.error ?? "",
      ].map((cell) => {
        // Escape cells containing commas, quotes, or newlines
        const cellStr = String(cell);
        if (
          cellStr.includes(",") ||
          cellStr.includes('"') ||
          cellStr.includes("\n")
        ) {
          return `"${cellStr.replace(/"/g, '""')}"`;
        }
        return cellStr;
      });
    });

    // Combine headers and rows
    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, -5);
    downloadFile(csvContent, `batch-results-${timestamp}.csv`, "text/csv");
  };

  const handleExportJSON = () => {
    // Export the full BatchInferenceResponse as pretty-printed JSON
    const jsonContent = JSON.stringify(data, null, 2);

    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, -5);
    downloadFile(
      jsonContent,
      `batch-results-${timestamp}.json`,
      "application/json",
    );
  };

  const handleExportPDF = async () => {
    setPdfLoading(true);
    try {
      const pdfBlob = await generateBatchPdfReport(data);
      const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, -5);
      downloadBlob(pdfBlob, `batch-analysis-report-${timestamp}.pdf`);

      toast({
        title: "PDF Report Generated",
        description: "Your clinical report has been downloaded.",
      });
    } catch (error: any) {
      toast({
        title: "PDF Generation Failed",
        description: error.message || "Failed to generate PDF report",
        variant: "destructive",
      });
    } finally {
      setPdfLoading(false);
    }
  };

  const handleExportPDFWithHeatmaps = async () => {
    if (!imageFiles || imageFiles.size === 0) {
      toast({
        title: "No Images Available",
        description: "Image files are required for heatmap generation.",
        variant: "destructive",
      });
      return;
    }

    setPdfHeatmapLoading(true);
    try {
      // Get files for successful results only
      const successfulResults = data.results.filter((r) => r.success);
      const filesToProcess: File[] = [];

      for (const result of successfulResults) {
        const file = imageFiles.get(result.filename);
        if (file) {
          filesToProcess.push(file);
        }
      }

      if (filesToProcess.length === 0) {
        toast({
          title: "No Valid Images",
          description:
            "No matching image files found for successful predictions.",
          variant: "destructive",
        });
        setPdfHeatmapLoading(false);
        return;
      }

      // Generate heatmaps in batch
      toast({
        title: "Generating Heatmaps",
        description: `Processing ${filesToProcess.length} images...`,
      });

      const heatmapResponse = await generateBatchHeatmaps(filesToProcess);

      // Build maps of heatmap and original image base64 data
      const heatmaps = new Map<string, string>();
      const originalImages = new Map<string, string>();

      for (const heatmapItem of heatmapResponse.results) {
        if (heatmapItem.success && heatmapItem.heatmap_base64) {
          heatmaps.set(heatmapItem.filename, heatmapItem.heatmap_base64);
        }
        if (heatmapItem.success && heatmapItem.original_image_base64) {
          originalImages.set(
            heatmapItem.filename,
            heatmapItem.original_image_base64,
          );
        }
      }

      // Generate PDF with heatmaps
      toast({
        title: "Generating PDF",
        description: "Creating report with heatmap appendix...",
      });

      const pdfBlob = await generateBatchPdfReportWithHeatmaps(
        data,
        heatmaps,
        originalImages,
      );

      const timestamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, -5);
      downloadBlob(
        pdfBlob,
        `batch-analysis-report-with-heatmaps-${timestamp}.pdf`,
      );

      toast({
        title: "PDF Report Generated",
        description: `Report with ${heatmaps.size} heatmaps has been downloaded.`,
      });
    } catch (error: any) {
      toast({
        title: "PDF Generation Failed",
        description: error.message || "Failed to generate PDF with heatmaps",
        variant: "destructive",
      });
    } finally {
      setPdfHeatmapLoading(false);
    }
  };

  const hasImageFiles = imageFiles && imageFiles.size > 0;

   return (
     <DropdownMenu>
       <DropdownMenuTrigger asChild>
         <Button
           disabled={disabled}
           size="lg"
           className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-colors duration-200"
         >
           <Download className="w-5 h-5 mr-2" />
           Export Results
         </Button>
       </DropdownMenuTrigger>
       <DropdownMenuContent
         align="end"
         className="w-64 rounded-xl border-[hsl(172_30%_85%)] bg-white/95 backdrop-blur-sm shadow-xl"
       >
        <DropdownMenuItem
          onClick={handleExportCSV}
          className="cursor-pointer rounded-lg hover:bg-[hsl(168_25%_96%)] focus:bg-[hsl(168_25%_96%)] p-3"
        >
          <FileSpreadsheet className="w-5 h-5 mr-3 text-[hsl(152_60%_42%)]" />
          <div className="flex-1">
            <p className="font-medium text-[hsl(172_43%_15%)]">Export as CSV</p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              For spreadsheet analysis
            </p>
          </div>
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={handleExportJSON}
          className="cursor-pointer rounded-lg hover:bg-[hsl(168_25%_96%)] focus:bg-[hsl(168_25%_96%)] p-3"
        >
          <FileJson className="w-5 h-5 mr-3 text-[hsl(172_63%_28%)]" />
          <div className="flex-1">
            <p className="font-medium text-[hsl(172_43%_15%)]">
              Export as JSON
            </p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              Full data with metadata
            </p>
          </div>
        </DropdownMenuItem>

        <DropdownMenuSeparator className="bg-[hsl(172_20%_92%)]" />

        <DropdownMenuItem
          onClick={handleExportPDF}
          disabled={pdfLoading}
          className="cursor-pointer rounded-lg hover:bg-[hsl(168_25%_96%)] focus:bg-[hsl(168_25%_96%)] p-3"
        >
          {pdfLoading ? (
            <Loader2 className="w-5 h-5 mr-3 text-[hsl(172_63%_28%)] animate-spin" />
          ) : (
            <FileText className="w-5 h-5 mr-3 text-[hsl(200_70%_45%)]" />
          )}
          <div className="flex-1">
            <p className="font-medium text-[hsl(172_43%_15%)]">
              {pdfLoading ? "Generating..." : "Export as PDF"}
            </p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              Professional clinical report
            </p>
          </div>
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={handleExportPDFWithHeatmaps}
          disabled={pdfHeatmapLoading || !hasImageFiles}
          className="cursor-pointer rounded-lg hover:bg-[hsl(168_25%_96%)] focus:bg-[hsl(168_25%_96%)] p-3"
        >
          {pdfHeatmapLoading ? (
            <Loader2 className="w-5 h-5 mr-3 text-[hsl(25_90%_50%)] animate-spin" />
          ) : (
            <Flame className="w-5 h-5 mr-3 text-[hsl(25_90%_50%)]" />
          )}
          <div className="flex-1">
            <p className="font-medium text-[hsl(172_43%_15%)]">
              {pdfHeatmapLoading ? "Generating..." : "PDF with Heatmaps"}
            </p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              {hasImageFiles
                ? "Report with GradCAM appendix"
                : "Requires image files"}
            </p>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default BatchExportButton;
