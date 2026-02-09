import React from "react";
import { Users, TrendingUp, Activity } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { chartColors } from "../chartConfig";
import { truncateId } from "../../../utils/formatters";

interface ClientBestMetrics {

  best_val_accuracy: number | null;
  best_val_precision: number | null;
  best_val_recall: number | null;
  best_val_f1: number | null;
  best_val_auroc: number | null;
  lowest_val_loss: number | null;
}

interface TrainingHistoryData {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  valPrecision?: number;
  valRecall?: number;
  valF1?: number;
  valAuroc?: number;
}

interface ClientChartData {
  clientId: number;
  clientIdentifier: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auroc: number;
  loss: number;
}

interface ClientMetricsTabProps {
  clientMetricsChartData: ClientChartData[];
  clientTrainingHistories: Record<string, TrainingHistoryData[]>;
  aggregatedRoundMetrics: Array<{
    round: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auroc: number;
    loss: number;
  }>;
  numClients: number;
}

// Generate distinct colors for each client
const clientColors = [
  "hsl(172, 63%, 28%)", // Teal
  "hsl(210, 60%, 50%)", // Blue
  "hsl(35, 70%, 50%)", // Amber
  "hsl(280, 50%, 55%)", // Purple
  "hsl(150, 50%, 40%)", // Green
  "hsl(0, 65%, 55%)", // Red
  "hsl(190, 70%, 45%)", // Cyan
  "hsl(45, 80%, 50%)", // Yellow
];

const getClientColor = (index: number): string => {
  return clientColors[index % clientColors.length];
};

const ClientMetricsTab: React.FC<ClientMetricsTabProps> = ({
  clientMetricsChartData,
  clientTrainingHistories,
  aggregatedRoundMetrics,
  numClients,
}) => {
  if (!clientMetricsChartData || clientMetricsChartData.length === 0) {
    return (
      <div className="bg-[hsl(210_20%_97%)] rounded-xl p-8 text-center">
        <Users className="h-12 w-12 text-[hsl(210_30%_70%)] mx-auto mb-4" />
        <p className="text-[hsl(215_15%_45%)]">
          No per-client metrics available for this run.
        </p>
      </div>
    );
  }

  // Prepare data for the comparison bar chart
  const comparisonData = [
    {
      metric: "Accuracy",
      ...Object.fromEntries(
        clientMetricsChartData.map((c) => [
          truncateId(c.clientIdentifier),
          c.accuracy,
        ])
      ),
    },
    {
      metric: "Precision",
      ...Object.fromEntries(
        clientMetricsChartData.map((c) => [
          truncateId(c.clientIdentifier),
          c.precision,
        ])
      ),
    },
    {
      metric: "Recall",
      ...Object.fromEntries(
        clientMetricsChartData.map((c) => [truncateId(c.clientIdentifier), c.recall])
      ),
    },
    {
      metric: "F1 Score",
      ...Object.fromEntries(
        clientMetricsChartData.map((c) => [truncateId(c.clientIdentifier), c.f1])
      ),
    },
    {
      metric: "AUC-ROC",
      ...Object.fromEntries(
        clientMetricsChartData.map((c) => [truncateId(c.clientIdentifier), c.auroc])
      ),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-[hsl(210_100%_97%)] rounded-xl p-4 border border-[hsl(210_60%_85%)]">
        <div className="flex gap-3">
          <Users className="h-5 w-5 text-[hsl(210_60%_45%)] flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">
              Per-Client Training Metrics
            </p>
            <p className="text-sm text-[hsl(210_50%_35%)]">
              View detailed training progress for each client participating in
              federated learning. Each client trains on its local data partition
              before aggregating updates to the global model.
            </p>
          </div>
        </div>
      </div>

      {/* Client Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {clientMetricsChartData.map((client, index) => (
          <ClientCard
            key={client.clientId}
            client={client}
            colorIndex={index}
          />
        ))}
      </div>

      {/* Client Metrics Comparison Bar Chart */}
      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <TrendingUp className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          Client Performance Comparison
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData} layout="vertical">
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 90%)"
              />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <YAxis
                dataKey="metric"
                type="category"
                tick={{ fill: "hsl(215, 15%, 45%)" }}
                width={80}
              />
              <Tooltip
                contentStyle={{ borderRadius: "12px" }}
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
              />
              <Legend />
              {clientMetricsChartData.map((client, index) => (
                <Bar
                  key={client.clientIdentifier}
                  dataKey={truncateId(client.clientIdentifier)}
                  name={truncateId(client.clientIdentifier)}
                  fill={getClientColor(index)}
                  radius={[0, 4, 4, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-Client Training History Charts */}
      {Object.keys(clientTrainingHistories).length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
              <Activity className="h-5 w-5 text-[hsl(172_63%_35%)]" />
            </div>
            Training Progress by Client
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(clientTrainingHistories).map(
              ([clientId, history], index) => (
                <ClientTrainingChart
                  key={clientId}
                  clientIdentifier={clientId}
                  trainingHistory={history}
                  colorIndex={index}
                />
              )
            )}
          </div>
        </div>
      )}

      {/* Aggregated Round Metrics */}
      {aggregatedRoundMetrics.length > 0 && (
        <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
              <TrendingUp className="h-5 w-5 text-[hsl(172_63%_35%)]" />
            </div>
            Aggregated Global Model Metrics (Per Round)
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={aggregatedRoundMetrics}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(210, 15%, 90%)"
                />
                <XAxis
                  dataKey="round"
                  tick={{ fill: "hsl(215, 15%, 45%)" }}
                  label={{
                    value: "Round",
                    position: "insideBottom",
                    offset: -5,
                  }}
                />
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
                  dataKey="f1"
                  name="F1 Score"
                  stroke={chartColors.f1Score}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

// Client Summary Card Component
interface ClientCardProps {
  client: ClientChartData;
  colorIndex: number;
}

const ClientCard: React.FC<ClientCardProps> = ({ client, colorIndex }) => {
  const color = getClientColor(colorIndex);

  return (
    <div
      className="bg-white rounded-2xl p-5 border border-[hsl(210_15%_88%)] shadow-sm hover:shadow-md transition-shadow"
      style={{ borderLeftColor: color, borderLeftWidth: "4px" }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
          style={{ backgroundColor: color }}
        >
          {truncateId(client.clientIdentifier).replace("client_", "C")}
        </div>
        <div>
          <h4 className="font-semibold text-[hsl(172_43%_15%)] capitalize">
            {truncateId(client.clientIdentifier).replace("_", " ")}
          </h4>
          <p className="text-xs text-[hsl(215_15%_50%)]">Local Training</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricBadge label="Accuracy" value={client.accuracy} />
        <MetricBadge label="Precision" value={client.precision} />
        <MetricBadge label="Recall" value={client.recall} />
        <MetricBadge label="F1 Score" value={client.f1} />
      </div>
    </div>
  );
};

// Metric Badge Component
interface MetricBadgeProps {
  label: string;
  value: number;
}

const MetricBadge: React.FC<MetricBadgeProps> = ({ label, value }) => (
  <div className="bg-[hsl(168_25%_97%)] rounded-lg p-2">
    <p className="text-[10px] text-[hsl(215_15%_50%)] uppercase tracking-wider mb-0.5">
      {label}
    </p>
    <p className="text-sm font-semibold text-[hsl(172_63%_25%)]">
      {(value * 100).toFixed(1)}%
    </p>
  </div>
);

// Client Training Chart Component
interface ClientTrainingChartProps {
  clientIdentifier: string;
  trainingHistory: TrainingHistoryData[];
  colorIndex: number;
}

const ClientTrainingChart: React.FC<ClientTrainingChartProps> = ({
  clientIdentifier,
  trainingHistory,
  colorIndex,
}) => {
  const color = getClientColor(colorIndex);

  if (!trainingHistory || trainingHistory.length === 0) {
    return null;
  }

  return (
    <div
      className="bg-white rounded-2xl p-5 border border-[hsl(210_15%_88%)] shadow-sm"
      style={{ borderTopColor: color, borderTopWidth: "3px" }}
    >
      <h4 className="font-semibold text-[hsl(172_43%_15%)] capitalize mb-4 flex items-center gap-2">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
        {truncateId(clientIdentifier).replace("_", " ")} - Validation Metrics
      </h4>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trainingHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 92%)" />
            <XAxis
              dataKey="epoch"
              tick={{ fill: "hsl(215, 15%, 50%)", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "hsl(215, 15%, 50%)", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ borderRadius: "8px", fontSize: "12px" }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Line
              type="monotone"
              dataKey="valAcc"
              name="Val Acc"
              stroke={chartColors.valAcc}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="valF1"
              name="Val F1"
              stroke={chartColors.valF1}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="valRecall"
              name="Val Recall"
              stroke={chartColors.valRecall}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ClientMetricsTab;
