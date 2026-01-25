/**
 * Batch Results Section Component
 * Displays batch analysis results with export and detailed grid
 */

import React from "react";
import SectionHeader from "@/components/inference/SectionHeader";
import BatchSummaryStats from "@/components/inference/BatchSummaryStats";
import BatchExportButton from "@/components/inference/BatchExportButton";
import BatchResultsGrid from "@/components/inference/BatchResultsGrid";
import { BatchInferenceResponse } from "@/types/inference";

interface BatchResultsSectionProps {
  batchResult: BatchInferenceResponse;
  imageUrls: Map<string, string>;
  imageFiles: Map<string, File>;
}

const BatchResultsSection: React.FC<BatchResultsSectionProps> = ({
   batchResult,
   imageUrls,
   imageFiles,
 }) => {
   return (
     <div className="space-y-8">
       {/* Summary Stats */}
       <div className="content-card batch-results">
         <BatchSummaryStats summary={batchResult.summary} />
       </div>

       {/* Export and Details Header */}
       <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
         {/* Export Card */}
         <div className="content-card batch-results flex flex-col items-start justify-between p-6">
           <div className="mb-4">
             <h3 className="text-lg font-bold text-[hsl(172_43%_15%)]">
               Export Results
             </h3>
             <p className="text-sm text-[hsl(215_15%_45%)] mt-1">
               Download analysis data in multiple formats
             </p>
           </div>
           <BatchExportButton data={batchResult} imageFiles={imageFiles} />
         </div>

         {/* Quick Stats Card */}
         <div className="content-card batch-results p-6">
           <h3 className="text-lg font-bold text-[hsl(172_43%_15%)] mb-4">
             Quick Stats
           </h3>
           <div className="space-y-3">
             <div className="flex items-center justify-between p-3 rounded-lg bg-[hsl(168_25%_96%)]">
               <span className="text-sm text-[hsl(215_15%_45%)]">Success Rate</span>
               <span className="font-bold text-[hsl(152_60%_30%)]">
                 {batchResult.summary.total_images > 0
                   ? ((batchResult.summary.successful / batchResult.summary.total_images) * 100).toFixed(1)
                   : 0}%
               </span>
             </div>
             <div className="flex items-center justify-between p-3 rounded-lg bg-[hsl(168_25%_96%)]">
               <span className="text-sm text-[hsl(215_15%_45%)]">Avg Confidence</span>
               <span className="font-bold text-[hsl(172_63%_28%)]">
                 {(batchResult.summary.avg_confidence * 100).toFixed(1)}%
               </span>
             </div>
             <div className="flex items-center justify-between p-3 rounded-lg bg-[hsl(168_25%_96%)]">
               <span className="text-sm text-[hsl(215_15%_45%)]">Pneumonia Cases</span>
               <span className="font-bold text-[hsl(35_70%_35%)]">
                 {batchResult.summary.pneumonia_count}
               </span>
             </div>
           </div>
         </div>
       </div>

       {/* Detailed Results Grid */}
       <div className="content-card batch-results">
         <div className="mb-6">
           <h2 className="text-2xl font-bold text-[hsl(172_43%_15%)]">
             Detailed Results
           </h2>
           <p className="text-sm text-[hsl(215_15%_45%)] mt-1">
             Click on any image to view full prediction details and generate heatmaps
           </p>
         </div>
         <BatchResultsGrid
           results={batchResult.results}
           imageUrls={imageUrls}
           imageFiles={imageFiles}
         />
       </div>
     </div>
   );
 };

export default BatchResultsSection;
