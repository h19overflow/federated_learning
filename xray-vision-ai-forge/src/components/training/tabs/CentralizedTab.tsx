import React from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp } from "lucide-react";
import { MetricCard } from "../MetricCard";
import ConfusionMatrixDisplay from "../ConfusionMatrixDisplay";
import { chartColors } from "../chartConfig";
import type { ExperimentResults } from "@/types/api";

interface MetricDataPoint {
  name: string;
  value: number;
}

interface CentralizedTabProps {
  centralizedResults: ExperimentResults | null;
  centralizedMetricsData: MetricDataPoint[];
  centralizedConfusionMatrix: number[][] | null;
}

const CentralizedTab: React.FC<CentralizedTabProps> = ({
  centralizedResults,
  centralizedMetricsData,
  centralizedConfusionMatrix,
}) => {
  if (!centralizedResults) {
    return (
      <div className="text-center py-12">
        <p className="text-[hsl(215_15%_55%)]">
          No centralized results available
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-[hsl(172_50%_96%)] to-[hsl(168_40%_95%)] rounded-xl p-5 border border-[hsl(172_40%_85%)] mb-6 shadow-sm">
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)] flex-shrink-0">
            <TrendingUp className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[hsl(172_63%_25%)] mb-1">
              Best Validation Metrics — Centralized Training
            </p>
            <p className="text-sm text-[hsl(172_43%_30%)]">
              Peak validation performance achieved during centralized training across all epochs.
            </p>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {centralizedMetricsData.map((metric, idx) => (
          <MetricCard
            key={metric.name}
            name={metric.name}
            value={metric.value}
            index={idx}
            total={centralizedMetricsData.length}
          />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">
            Performance Metrics
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={centralizedMetricsData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(210, 15%, 90%)"
                />
                <XAxis
                  dataKey="name"
                  tick={{
                    fill: "hsl(215, 15%, 45%)",
                    fontSize: 12,
                  }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(tick) => `${tick * 100}%`}
                  tick={{ fill: "hsl(215, 15%, 45%)" }}
                />
                <Tooltip
                  contentStyle={{ borderRadius: "12px" }}
                  formatter={(value: number) =>
                    `${(value * 100).toFixed(1)}%`
                  }
                />
                <Bar
                  dataKey="value"
                  name="Score"
                  radius={[4, 4, 0, 0]}
                >
                  {centralizedMetricsData.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={Object.values(chartColors)[index]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        {centralizedConfusionMatrix && (
          <ConfusionMatrixDisplay
            matrix={centralizedConfusionMatrix}
            title="Confusion Matrix"
          />
        )}
      </div>
    </div>
  );
};

export default CentralizedTab;
