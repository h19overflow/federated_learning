/**
 * BatchExportButton Component
 *
 * Dropdown button that exports batch prediction results in multiple formats.
 * Supports CSV (for spreadsheet analysis) and JSON (for full data export).
 */

import React from 'react';
import { Download, FileSpreadsheet, FileJson } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { BatchInferenceResponse } from '@/types/inference';

interface BatchExportButtonProps {
  data: BatchInferenceResponse;
  disabled?: boolean;
}

export const BatchExportButton: React.FC<BatchExportButtonProps> = ({
  data,
  disabled = false,
}) => {
  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExportCSV = () => {
    // CSV headers
    const headers = [
      'Filename',
      'Status',
      'Prediction',
      'Confidence',
      'Pneumonia Probability',
      'Normal Probability',
      'Risk Level',
      'Processing Time (ms)',
      'Error',
    ];

    // CSV rows
    const rows = data.results.map((result) => {
      const prediction = result.prediction;
      const riskLevel = result.clinical_interpretation?.risk_assessment?.risk_level ?? 'N/A';

      return [
        result.filename,
        result.success ? 'Success' : 'Failed',
        prediction?.predicted_class ?? 'N/A',
        prediction ? (prediction.confidence * 100).toFixed(2) : 'N/A',
        prediction ? (prediction.pneumonia_probability * 100).toFixed(2) : 'N/A',
        prediction ? (prediction.normal_probability * 100).toFixed(2) : 'N/A',
        riskLevel,
        result.processing_time_ms.toFixed(2),
        result.error ?? '',
      ].map((cell) => {
        // Escape cells containing commas, quotes, or newlines
        const cellStr = String(cell);
        if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
          return `"${cellStr.replace(/"/g, '""')}"`;
        }
        return cellStr;
      });
    });

    // Combine headers and rows
    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.join(',')),
    ].join('\n');

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadFile(csvContent, `batch-results-${timestamp}.csv`, 'text/csv');
  };

  const handleExportJSON = () => {
    // Export the full BatchInferenceResponse as pretty-printed JSON
    const jsonContent = JSON.stringify(data, null, 2);

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    downloadFile(jsonContent, `batch-results-${timestamp}.json`, 'application/json');
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          disabled={disabled}
          size="lg"
          className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-2xl shadow-lg shadow-[hsl(172_63%_22%)]/20 hover:shadow-xl transition-all duration-300"
        >
          <Download className="w-5 h-5 mr-2" />
          Export Results
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-56 rounded-xl border-[hsl(172_30%_85%)] bg-white/95 backdrop-blur-sm shadow-xl"
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
            <p className="font-medium text-[hsl(172_43%_15%)]">Export as JSON</p>
            <p className="text-xs text-[hsl(215_15%_50%)]">
              Full data with metadata
            </p>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default BatchExportButton;
