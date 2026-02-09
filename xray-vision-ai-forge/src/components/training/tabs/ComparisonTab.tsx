import React from "react";
import {
  BarChart,
  LineChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  ArrowLeftRight,
  BarChart2,
  BarChart3,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { chartColors } from "../chartConfig";

interface ComparisonMetric {
  name: string;
  centralized: number;
  federated: number;
  difference: string;
}

interface HistoryDataPoint {
  epoch: number;
  valAcc?: number;
  trainAcc?: number;
  valLoss?: number;
  trainLoss?: number;
  valPrecision?: number;
  valRecall?: number;
  valF1?: number;
  valAuroc?: number;
}

interface ComparisonTabProps {
  comparisonMetricsData: ComparisonMetric[];
  centralizedHistoryData: HistoryDataPoint[];
  federatedHistoryData: HistoryDataPoint[];
}

const ComparisonTab: React.FC<ComparisonTabProps> = ({
  comparisonMetricsData,
  centralizedHistoryData,
  federatedHistoryData,
}) => {
  return (
    <div className="space-y-6">
      {/* Metrics Comparison Table */}
      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <BarChart2 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          <span>Best Validation Metrics Comparison</span>
          <span className="ml-auto text-xs font-normal text-[hsl(215_15%_50%)]">Head-to-head performance</span>
        </h3>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent border-b-2 border-[hsl(168_20%_88%)]">
                <TableHead className="text-[hsl(215_15%_40%)] font-semibold">
                  Metric
                </TableHead>
                <TableHead className="text-[hsl(215_15%_40%)] font-semibold text-center">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-[hsl(172_63%_28%)]" />
                    Centralized
                  </div>
                </TableHead>
                <TableHead className="text-[hsl(215_15%_40%)] font-semibold text-center">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-[hsl(210_60%_50%)]" />
                    Federated
                  </div>
                </TableHead>
                <TableHead className="text-[hsl(215_15%_40%)] font-semibold text-center">
                  Difference
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {comparisonMetricsData.map((metric, idx) => {
                const diff = parseFloat(metric.difference);
                const isPositive = diff > 0;
                return (
                  <TableRow
                    key={metric.name}
                    className="hover:bg-[hsl(168_25%_97%)] border-b border-[hsl(210_15%_92%)] transition-colors"
                    style={{
                      animation: "slideInUp 0.4s ease-out forwards",
                      animationDelay: `${idx * 0.05}s`,
                      opacity: 0,
                    }}
                  >
                    <TableCell className="font-semibold text-[hsl(172_43%_20%)]">
                      {metric.name}
                    </TableCell>
                    <TableCell className="font-mono text-[hsl(172_63%_28%)] text-center font-medium">
                      {(metric.centralized * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="font-mono text-[hsl(210_60%_50%)] text-center font-medium">
                      {(metric.federated * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-center">
                      <span
                        className={`inline-flex items-center gap-1 px-3 py-1 rounded-lg font-mono font-semibold text-sm ${
                          isPositive
                            ? "bg-[hsl(152_60%_95%)] text-[hsl(152_60%_35%)]"
                            : "bg-[hsl(0_60%_95%)] text-[hsl(0_72%_45%)]"
                        }`}
                      >
                        {isPositive ? "↑" : "↓"}
                        {isPositive ? "+" : ""}
                        {(diff * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Comparison Bar Chart */}
      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <BarChart3 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          <span>Side-by-Side Performance</span>
          <span className="ml-auto text-xs font-normal text-[hsl(215_15%_50%)]">Visual comparison</span>
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={comparisonMetricsData}
              margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 90%)"
              />
              <XAxis
                dataKey="name"
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(tick) => `${tick * 100}%`}
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "white",
                  border: "1px solid hsl(168, 20%, 90%)",
                  borderRadius: "12px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                }}
                formatter={(value: number) =>
                  `${(value * 100).toFixed(1)}%`
                }
              />
              <Legend />
              <Bar
                name="Centralized"
                dataKey="centralized"
                fill={chartColors.centralized}
                radius={[4, 4, 0, 0]}
              />
              <Bar
                name="Federated"
                dataKey="federated"
                fill={chartColors.federated}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Training Progress Comparison */}
      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <ArrowLeftRight className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          <span>Training Progress Comparison</span>
          <span className="ml-auto text-xs font-normal text-[hsl(215_15%_50%)]">Epoch-by-epoch analysis</span>
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 90%)"
              />
              <XAxis
                dataKey="epoch"
                allowDuplicatedCategory={false}
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <YAxis
                domain={[0.5, 1]}
                tickFormatter={(tick) =>
                  `${(tick * 100).toFixed(0)}%`
                }
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "white",
                  border: "1px solid hsl(168, 20%, 90%)",
                  borderRadius: "12px",
                }}
                formatter={(value: number) =>
                  `${(value * 100).toFixed(1)}%`
                }
              />
              <Legend />
              <Line
                data={centralizedHistoryData}
                type="monotone"
                dataKey="valAcc"
                name="Centralized"
                stroke={chartColors.centralized}
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                data={federatedHistoryData}
                type="monotone"
                dataKey="valAcc"
                name="Federated"
                stroke={chartColors.federated}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default ComparisonTab;
