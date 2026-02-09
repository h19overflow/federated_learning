import React from "react";
import { HelpCircle, TrendingUp } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { MetricCard } from "../MetricCard";
import ConfusionMatrixDisplay from "../ConfusionMatrixDisplay";
import { chartColors } from "../chartConfig";

interface ServerEvaluationChartData {
  round: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auroc?: number;
}

interface MetricData {
  name: string;
  value: number;
}

interface ServerEvaluationTabProps {
  serverEvaluationLatestMetrics: MetricData[] | null;
  serverEvaluationChartData: ServerEvaluationChartData[];
  serverEvaluationConfusionMatrix: number[][] | null;
}

const ServerEvaluationTab: React.FC<ServerEvaluationTabProps> = ({
  serverEvaluationLatestMetrics,
  serverEvaluationChartData,
  serverEvaluationConfusionMatrix,
}) => {
  return (
    <div className="space-y-6">
      <div className="bg-[hsl(210_100%_97%)] rounded-xl p-4 border border-[hsl(210_60%_85%)] mb-6">
        <div className="flex gap-3">
          <HelpCircle className="h-5 w-5 text-[hsl(210_60%_45%)] flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">
              Server Evaluation - Global Model Performance
            </p>
            <p className="text-sm text-[hsl(210_50%_35%)]">
              The aggregated global model is evaluated on a held-out
              centralized test set after each training round. This provides an
              objective measure of the global model's generalization performance
              independent of client-side variations.
            </p>
          </div>
        </div>
      </div>

      {serverEvaluationLatestMetrics && (
        <>
          <div className="bg-[hsl(172_50%_96%)] rounded-xl p-4 border border-[hsl(172_40%_85%)] mb-4">
            <div className="flex gap-3">
              <TrendingUp className="h-5 w-5 text-[hsl(172_63%_35%)] flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-[hsl(172_63%_25%)] mb-1">
                  Latest Round - Global Model Performance
                </p>
                <p className="text-sm text-[hsl(172_43%_30%)]">
                  Performance metrics of the aggregated global model from the
                  most recent training round on the centralized test set.
                </p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {serverEvaluationLatestMetrics.map((metric, idx) => (
              <MetricCard
                key={metric.name}
                name={metric.name}
                value={metric.value}
                index={idx}
                total={serverEvaluationLatestMetrics.length}
              />
            ))}
          </div>
        </>
      )}

      <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">
          Metrics Over Rounds
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={serverEvaluationChartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 90%)"
              />
              <XAxis dataKey="round" tick={{ fill: "hsl(215, 15%, 45%)" }} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <Tooltip
                contentStyle={{ borderRadius: "12px" }}
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="accuracy"
                name="Accuracy"
                stroke={chartColors.accuracy}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="precision"
                name="Precision"
                stroke={chartColors.precision}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="recall"
                name="Recall"
                stroke={chartColors.recall}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="f1_score"
                name="F1 Score"
                stroke={chartColors.f1Score}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="auroc"
                name="AUC-ROC"
                stroke={chartColors.auc}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {serverEvaluationConfusionMatrix && (
        <>
          <ConfusionMatrixDisplay
            matrix={serverEvaluationConfusionMatrix}
            title="Confusion Matrix (Latest Round)"
          />
        </>
      )}
    </div>
  );
};

export default ServerEvaluationTab;
