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

interface FederatedTabProps {
  federatedResults: ExperimentResults | null;
  federatedMetricsData: MetricDataPoint[];
  federatedConfusionMatrix: number[][] | null;
}

const FederatedTab: React.FC<FederatedTabProps> = ({
  federatedResults,
  federatedMetricsData,
  federatedConfusionMatrix,
}) => {
  if (!federatedResults) {
    return (
      <div className="text-center py-12">
        <p className="text-[hsl(215_15%_55%)]">
          No federated results available
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-[hsl(210_100%_97%)] to-[hsl(210_80%_96%)] rounded-xl p-5 border border-[hsl(210_60%_85%)] mb-6 shadow-sm">
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-[hsl(210_60%_94%)] flex-shrink-0">
            <TrendingUp className="h-5 w-5 text-[hsl(210_60%_45%)]" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">
              Final Round Metrics — Federated Learning
            </p>
            <p className="text-sm text-[hsl(210_50%_35%)]">
              Averaged client-side metrics from the final training round across all participating clients.
            </p>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {federatedMetricsData.map((metric, idx) => (
          <MetricCard
            key={metric.name}
            name={metric.name}
            value={metric.value}
            index={idx}
            total={federatedMetricsData.length}
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
              <BarChart data={federatedMetricsData}>
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
                  {federatedMetricsData.map((_, index) => (
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
        {federatedConfusionMatrix && (
          <ConfusionMatrixDisplay
            matrix={federatedConfusionMatrix}
            title="Confusion Matrix"
          />
        )}
      </div>
    </div>
  );
};

export default FederatedTab;
