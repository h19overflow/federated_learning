import React from "react";
import { TrendingUp, BarChart3 } from "lucide-react";
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
import { MetricCard } from "../MetricCard";
import ConfusionMatrixDisplay from "../ConfusionMatrixDisplay";
import { chartColors } from "../chartConfig";

interface MetricData {
  name: string;
  value: number;
}

interface MetricsTabProps {
  metricsChartData: MetricData[];
  confusionMatrix: number[][] | null;
  trainingMode: string;
}

const MetricsTab: React.FC<MetricsTabProps> = ({
  metricsChartData,
  confusionMatrix,
  trainingMode,
}) => {
  return (
    <div className="space-y-6">
      {trainingMode === "federated" && (
        <div className="bg-gradient-to-r from-[hsl(210_100%_97%)] to-[hsl(210_80%_96%)] rounded-xl p-5 border border-[hsl(210_60%_85%)] mb-6 shadow-sm">
          <div className="flex gap-3">
            <div className="p-2 rounded-lg bg-[hsl(210_60%_94%)] flex-shrink-0">
              <TrendingUp className="h-5 w-5 text-[hsl(210_60%_45%)]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">
                Best Validation Metrics — Federated Learning
              </p>
              <p className="text-sm text-[hsl(210_50%_35%)]">
                These are the highest validation metrics achieved across all
                training rounds. These represent the best performance of the
                client-side training throughout the entire federated learning
                process.
              </p>
            </div>
          </div>
        </div>
      )}
      {trainingMode === "centralized" && (
        <div className="bg-gradient-to-r from-[hsl(172_50%_96%)] to-[hsl(168_40%_95%)] rounded-xl p-5 border border-[hsl(172_40%_85%)] mb-6 shadow-sm">
          <div className="flex gap-3">
            <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)] flex-shrink-0">
              <TrendingUp className="h-5 w-5 text-[hsl(172_63%_35%)]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-[hsl(172_63%_25%)] mb-1">
                Best Validation Metrics
              </p>
              <p className="text-sm text-[hsl(172_43%_30%)]">
                These are the highest validation metrics achieved during
                training across all epochs. The model checkpoint with these
                metrics is saved for deployment.
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {metricsChartData.map((metric, idx) => (
          <MetricCard
            key={metric.name}
            name={metric.name}
            value={metric.value}
            index={idx}
            total={metricsChartData.length}
          />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
              <BarChart3 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
            </div>
            Performance Metrics
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsChartData}>
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
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="value" name="Score" radius={[4, 4, 0, 0]}>
                  {metricsChartData.map((_, index) => (
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
        {confusionMatrix && (
          <ConfusionMatrixDisplay
            matrix={confusionMatrix}
            title="Confusion Matrix"
          />
        )}
      </div>
    </div>
  );
};

export default MetricsTab;
