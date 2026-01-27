/**
 * Performance comparison chart component for analytics visualization.
 *
 * Displays a bar chart comparing average precision, recall, and F1 scores
 * between centralized and federated training modes.
 */

import React from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { BarChart3 } from "lucide-react";
import { ModeStatistics } from "./types";
import { cardHoverVariants, itemVariants } from "./animationVariants";

/**
 * Props for the PerformanceChart component.
 */
interface PerformanceChartProps {
  /** Statistics for centralized training mode */
  centralizedData: ModeStatistics;
  /** Statistics for federated training mode */
  federatedData: ModeStatistics;
}

/**
 * Chart data point structure for recharts.
 */
interface ChartDataPoint {
  /** Metric name displayed on X-axis */
  metric: string;
  /** Centralized mode value (percentage 0-100) */
  Centralized: number;
  /** Federated mode value (percentage 0-100) */
  Federated: number;
}

/**
 * Computes chart data from mode statistics.
 *
 * Converts 0-1 scale metrics to percentage values (0-100) with one decimal place.
 *
 * @param centralizedData - Statistics for centralized mode
 * @param federatedData - Statistics for federated mode
 * @returns Array of chart data points for Precision, Recall, and F1 Score
 */
const computeChartData = (
  centralizedData: ModeStatistics,
  federatedData: ModeStatistics
): ChartDataPoint[] => {
  return [
    {
      metric: "Precision",
      Centralized: centralizedData.avg_precision
        ? Number((centralizedData.avg_precision * 100).toFixed(1))
        : 0,
      Federated: federatedData.avg_precision
        ? Number((federatedData.avg_precision * 100).toFixed(1))
        : 0,
    },
    {
      metric: "Recall",
      Centralized: centralizedData.avg_recall
        ? Number((centralizedData.avg_recall * 100).toFixed(1))
        : 0,
      Federated: federatedData.avg_recall
        ? Number((federatedData.avg_recall * 100).toFixed(1))
        : 0,
    },
    {
      metric: "F1 Score",
      Centralized: centralizedData.avg_f1
        ? Number((centralizedData.avg_f1 * 100).toFixed(1))
        : 0,
      Federated: federatedData.avg_f1
        ? Number((federatedData.avg_f1 * 100).toFixed(1))
        : 0,
    },
  ];
};

/**
 * Performance comparison chart component.
 *
 * Displays an animated bar chart comparing precision, recall, and F1 scores
 * between centralized and federated training modes. Uses motion animations
 * for hover effects and card interactions.
 *
 * @param props - Component props containing centralized and federated statistics
 * @returns Rendered performance comparison chart card
 */
export const PerformanceChart: React.FC<PerformanceChartProps> = ({
  centralizedData,
  federatedData,
}) => {
  const chartData = computeChartData(centralizedData, federatedData);

  return (
    <motion.div variants={itemVariants} initial="rest" whileHover="hover">
      <motion.div variants={cardHoverVariants}>
        <Card className="p-6 transition-shadow hover:shadow-xl">
          <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Performance Comparison
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(210, 15%, 92%)"
              />
              <XAxis dataKey="metric" stroke="hsl(172, 43%, 15%)" />
              <YAxis
                stroke="hsl(172, 43%, 15%)"
                domain={[0, 100]}
                label={{
                  value: "Percentage (%)",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "white",
                  border: "1px solid hsl(210, 15%, 92%)",
                  borderRadius: "8px",
                  boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                }}
              />
              <Legend />
              <Bar
                dataKey="Centralized"
                fill="hsl(172, 63%, 28%)"
                radius={[8, 8, 0, 0]}
                animationDuration={800}
              />
              <Bar
                dataKey="Federated"
                fill="hsl(210, 60%, 50%)"
                radius={[8, 8, 0, 0]}
                animationDuration={800}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </motion.div>
    </motion.div>
  );
};
