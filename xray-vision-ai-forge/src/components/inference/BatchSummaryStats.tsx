/**
 * BatchSummaryStats Component
 *
 * Displays summary statistics for batch prediction results.
 * Shows total processed, success/failure counts, prediction distribution,
 * average confidence, and high-risk warnings.
 */

import React from 'react';
import { CheckCircle2, XCircle, AlertTriangle, Clock, TrendingUp } from 'lucide-react';
import { BatchSummaryStats as BatchSummaryStatsType } from '@/types/inference';
import { Progress } from '@/components/ui/progress';

interface BatchSummaryStatsProps {
  summary: BatchSummaryStatsType;
}

export const BatchSummaryStats: React.FC<BatchSummaryStatsProps> = ({ summary }) => {
  const {
    total_images,
    successful,
    failed,
    normal_count,
    pneumonia_count,
    avg_confidence,
    avg_processing_time_ms,
    high_risk_count,
  } = summary;

  const successRate = total_images > 0 ? (successful / total_images) * 100 : 0;
  const avgConfidencePercent = avg_confidence * 100;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
          <TrendingUp className="w-6 h-6 text-[hsl(172_63%_28%)]" />
        </div>
        <div>
          <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)]">
            Batch Analysis Summary
          </h3>
          <p className="text-sm text-[hsl(215_15%_45%)]">
            {total_images} image{total_images !== 1 ? 's' : ''} processed
          </p>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {/* Total images */}
        <div className="p-5 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(172_30%_88%)] shadow-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[hsl(215_15%_50%)] uppercase tracking-wide">
              Total Images
            </span>
          </div>
          <p className="text-3xl font-bold text-[hsl(172_43%_15%)]">
            {total_images}
          </p>
        </div>

        {/* Successful */}
        <div className="p-5 rounded-2xl bg-[hsl(152_50%_95%)] border-2 border-[hsl(152_50%_80%)] shadow-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[hsl(152_60%_30%)] uppercase tracking-wide">
              Successful
            </span>
            <CheckCircle2 className="w-5 h-5 text-[hsl(152_60%_42%)]" />
          </div>
          <p className="text-3xl font-bold text-[hsl(152_60%_30%)]">
            {successful}
          </p>
          <p className="text-xs text-[hsl(152_60%_35%)] mt-1">
            {successRate.toFixed(1)}% success rate
          </p>
        </div>

        {/* Failed */}
        {failed > 0 && (
          <div className="p-5 rounded-2xl bg-[hsl(0_60%_95%)] border-2 border-[hsl(0_60%_80%)] shadow-md">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-[hsl(0_70%_40%)] uppercase tracking-wide">
                Failed
              </span>
              <XCircle className="w-5 h-5 text-[hsl(0_70%_50%)]" />
            </div>
            <p className="text-3xl font-bold text-[hsl(0_70%_40%)]">
              {failed}
            </p>
          </div>
        )}

        {/* Normal count */}
        <div className="p-5 rounded-2xl bg-[hsl(152_50%_95%)] border border-[hsl(152_50%_85%)] shadow-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[hsl(152_60%_30%)] uppercase tracking-wide">
              Normal
            </span>
            <div className="w-3 h-3 rounded-full bg-[hsl(152_60%_42%)]" />
          </div>
          <p className="text-3xl font-bold text-[hsl(152_60%_30%)]">
            {normal_count}
          </p>
          <p className="text-xs text-[hsl(152_60%_35%)] mt-1">
            {successful > 0 ? ((normal_count / successful) * 100).toFixed(1) : 0}% of successful
          </p>
        </div>

        {/* Pneumonia count */}
        <div className="p-5 rounded-2xl bg-[hsl(35_60%_95%)] border border-[hsl(35_60%_85%)] shadow-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[hsl(35_70%_35%)] uppercase tracking-wide">
              Pneumonia
            </span>
            <div className="w-3 h-3 rounded-full bg-[hsl(35_70%_45%)]" />
          </div>
          <p className="text-3xl font-bold text-[hsl(35_70%_35%)]">
            {pneumonia_count}
          </p>
          <p className="text-xs text-[hsl(35_70%_40%)] mt-1">
            {successful > 0 ? ((pneumonia_count / successful) * 100).toFixed(1) : 0}% of successful
          </p>
        </div>

        {/* Processing time */}
        <div className="p-5 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(172_30%_88%)] shadow-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[hsl(215_15%_50%)] uppercase tracking-wide">
              Avg Time
            </span>
            <Clock className="w-5 h-5 text-[hsl(172_63%_28%)]" />
          </div>
          <p className="text-3xl font-bold text-[hsl(172_43%_15%)]">
            {avg_processing_time_ms.toFixed(0)}
          </p>
          <p className="text-xs text-[hsl(215_15%_45%)] mt-1">
            milliseconds
          </p>
        </div>
      </div>

      {/* Average confidence */}
      <div className="p-6 rounded-2xl bg-white/90 backdrop-blur-sm border border-[hsl(172_30%_88%)] shadow-md">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-[hsl(172_43%_20%)] uppercase tracking-wide">
            Average Confidence
          </h4>
          <span className="text-2xl font-bold text-[hsl(172_63%_28%)]">
            {avgConfidencePercent.toFixed(1)}%
          </span>
        </div>
        <Progress
          value={avgConfidencePercent}
          className="h-3 bg-[hsl(168_25%_96%)]"
        />
      </div>

      {/* High risk warning */}
      {high_risk_count > 0 && (
        <div className="p-5 rounded-2xl bg-[hsl(35_60%_95%)] border-2 border-[hsl(35_60%_80%)] shadow-lg">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center shadow-sm">
              <AlertTriangle className="w-5 h-5 text-[hsl(35_70%_45%)]" />
            </div>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-[hsl(35_70%_35%)] uppercase tracking-wide mb-1">
                High Risk Detections
              </h4>
              <p className="text-[hsl(35_70%_40%)] leading-relaxed">
                <span className="font-bold text-2xl">{high_risk_count}</span> image{high_risk_count !== 1 ? 's' : ''} flagged as high or critical risk.
                Review clinical interpretations carefully.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchSummaryStats;
