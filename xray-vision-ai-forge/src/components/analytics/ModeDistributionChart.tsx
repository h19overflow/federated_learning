import React from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { PieChartIcon } from "lucide-react";
import { ModeStatistics } from "./types";
import { itemVariants } from "./animationVariants";

/**
 * Props for the ModeDistributionChart component.
 * Requires statistics data for both centralized and federated training modes.
 */
interface ModeDistributionChartProps {
  /** Statistics for centralized training mode including count and metrics */
  centralizedData: ModeStatistics;
  /** Statistics for federated training mode including count and metrics */
  federatedData: ModeStatistics;
}

/**
 * Data structure for pie chart segments.
 * Represents a single slice of the training mode distribution pie.
 */
interface PieChartData {
  /** Display name for the training mode */
  name: string;
  /** Number of runs for this training mode */
  value: number;
  /** HSL color value for the pie segment */
  color: string;
}

/**
 * Displays a pie chart visualization of training mode distribution.
 *
 * Shows the proportion of centralized vs federated training runs
 * with color-coded segments and percentage labels. Includes an
 * animated legend showing exact counts for each mode.
 *
 * Features:
 * - Animated pie chart with hover effects
 * - Percentage labels on each segment
 * - Color-coded legend with run counts
 * - Responsive container sizing
 *
 * @param props - Component props containing centralized and federated data
 * @returns Rendered pie chart component with legend
 */
export function ModeDistributionChart({
  centralizedData,
  federatedData,
}: ModeDistributionChartProps): React.ReactElement {
  const chartData: PieChartData[] = [
    {
      name: "Centralized",
      value: centralizedData.count,
      color: "hsl(172, 63%, 28%)",
    },
    {
      name: "Federated",
      value: federatedData.count,
      color: "hsl(210, 60%, 50%)",
    },
  ];

  return (
    <motion.div variants={itemVariants} initial="rest" whileHover="hover">
      <Card className="p-6 transition-shadow hover:shadow-xl">
        <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
          <PieChartIcon className="w-5 h-5" />
          Training Mode Distribution
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) =>
                `${name}: ${(percent * 100).toFixed(0)}%`
              }
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
              animationBegin={0}
              animationDuration={800}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "white",
                border: "1px solid hsl(210, 15%, 92%)",
                borderRadius: "8px",
                boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-4 flex justify-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[hsl(172,63%,28%)]"></div>
            <span className="text-sm text-[hsl(215,15%,50%)]">
              Centralized ({centralizedData.count})
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[hsl(210,60%,50%)]"></div>
            <span className="text-sm text-[hsl(215,15%,50%)]">
              Federated ({federatedData.count})
            </span>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
