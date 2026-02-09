import React from "react";
import { Activity, FileText } from "lucide-react";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { chartColors } from "../chartConfig";

interface TrainingHistoryData {
  epoch: number;
  trainLoss?: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  valPrecision?: number;
  valRecall?: number;
  valF1?: number;
  valAuroc?: number;
}

interface ChartsTabProps {
  trainingHistoryData: TrainingHistoryData[];
}

const ChartsTab: React.FC<ChartsTabProps> = ({ trainingHistoryData }) => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
          <h3 className="text-md font-semibold text-[hsl(172_43%_15%)] mb-5 flex items-center gap-2">
            <div className="w-1 h-5 rounded-full bg-gradient-to-b from-[hsl(172_63%_28%)] to-[hsl(172_63%_35%)]" />
            Training & Validation Loss
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistoryData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(210, 15%, 90%)"
                />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "hsl(215, 15%, 45%)" }}
                />
                <YAxis tick={{ fill: "hsl(215, 15%, 45%)" }} />
                <Tooltip contentStyle={{ borderRadius: "12px" }} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  name="Train Loss"
                  stroke={chartColors.trainLoss}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  name="Val Loss"
                  stroke={chartColors.valLoss}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
          <h3 className="text-md font-semibold text-[hsl(172_43%_15%)] mb-5 flex items-center gap-2">
            <div className="w-1 h-5 rounded-full bg-gradient-to-b from-[hsl(168_40%_45%)] to-[hsl(168_40%_50%)]" />
            Training & Validation Accuracy
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistoryData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(210, 15%, 90%)"
                />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "hsl(215, 15%, 45%)" }}
                />
                <YAxis
                  domain={[0.5, 1]}
                  tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`}
                  tick={{ fill: "hsl(215, 15%, 45%)" }}
                />
                <Tooltip
                  contentStyle={{ borderRadius: "12px" }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="trainAcc"
                  name="Train Acc"
                  stroke={chartColors.trainAcc}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="valAcc"
                  name="Val Acc"
                  stroke={chartColors.valAcc}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <Activity className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          All Validation Metrics Over Time
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trainingHistoryData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 90%)"
              />
              <XAxis dataKey="epoch" tick={{ fill: "hsl(215, 15%, 45%)" }} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`}
                tick={{ fill: "hsl(215, 15%, 45%)" }}
              />
              <Tooltip
                contentStyle={{ borderRadius: "12px" }}
                formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="valAcc"
                name="Accuracy"
                stroke={chartColors.valAcc}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="valPrecision"
                name="Precision"
                stroke={chartColors.valPrecision}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="valRecall"
                name="Recall"
                stroke={chartColors.valRecall}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="valF1"
                name="F1 Score"
                stroke={chartColors.valF1}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="valAuroc"
                name="AUC-ROC"
                stroke={chartColors.valAuroc}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_20%_97%)] rounded-2xl p-6 border border-[hsl(168_20%_90%)] shadow-sm">
        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(210_40%_94%)]">
            <FileText className="h-5 w-5 text-[hsl(172_63%_35%)]" />
          </div>
          Validation Metrics History
        </h3>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Epoch
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val Accuracy
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val AUC-ROC
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val F1 Score
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val Loss
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val Precision
                </TableHead>
                <TableHead className="text-[hsl(215_15%_45%)] font-semibold">
                  Val Recall
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trainingHistoryData.map((row) => (
                <TableRow
                  key={row.epoch}
                  className="hover:bg-[hsl(168_25%_97%)]"
                >
                  <TableCell className="font-medium text-[hsl(172_43%_20%)]">
                    {row.epoch}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(172_63%_28%)]">
                    {row.valAcc ? (row.valAcc * 100).toFixed(2) + "%" : "N/A"}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(210_60%_50%)]">
                    {row.valAuroc
                      ? (row.valAuroc * 100).toFixed(2) + "%"
                      : "N/A"}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(172_50%_40%)]">
                    {row.valF1 ? (row.valF1 * 100).toFixed(2) + "%" : "N/A"}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(0_72%_40%)]">
                    {row.valLoss ? row.valLoss.toFixed(4) : "N/A"}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(168_40%_45%)]">
                    {row.valPrecision
                      ? (row.valPrecision * 100).toFixed(2) + "%"
                      : "N/A"}
                  </TableCell>
                  <TableCell className="font-mono text-[hsl(35_65%_55%)]">
                    {row.valRecall
                      ? (row.valRecall * 100).toFixed(2) + "%"
                      : "N/A"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
};

export default ChartsTab;
