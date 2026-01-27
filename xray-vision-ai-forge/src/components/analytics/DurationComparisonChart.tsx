/**
 * Duration Comparison Chart Component
 *
 * Displays a bar chart comparing average training duration between
 * centralized and federated training modes. Shows duration in minutes
 * with formatted labels and interactive tooltips.
 */

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { ModeStatistics } from "./types";
import { formatDuration } from "@/utils/formatters";
import { itemVariants } from "./animationVariants";

/**
 * Props for DurationComparisonChart component.
 */
interface DurationComparisonChartProps {
  /** Statistics for centralized training mode */
  centralizedData: ModeStatistics;
  /** Statistics for federated training mode */
  federatedData: ModeStatistics;
}

/**
 * Renders a bar chart comparing average training duration between modes.
 *
 * Features:
 * - Side-by-side bar comparison
 * - Color-coded bars (teal for centralized, blue for federated)
 * - Interactive tooltips with formatted duration
 * - Rounded bar corners for modern aesthetic
 * - Summary text showing exact durations
 * - Smooth animations on mount and hover
 *
 * @param props - Component props containing centralized and federated statistics
 * @returns JSX element containing the animated duration comparison chart
 */
export function DurationComparisonChart({
  centralizedData,
  federatedData,
}: DurationComparisonChartProps) {
  const chartData = [
    {
      mode: "Centralized",
      duration: centralizedData.avg_duration_minutes
        ? Number(centralizedData.avg_duration_minutes.toFixed(1))
        : 0,
    },
    {
      mode: "Federated",
      duration: federatedData.avg_duration_minutes
        ? Number(federatedData.avg_duration_minutes.toFixed(1))
        : 0,
    },
  ];

  const barColors = [
    { mode: "Centralized", color: "hsl(172, 63%, 28%)" },
    { mode: "Federated", color: "hsl(210, 60%, 50%)" },
  ];

  return (
    <motion.div variants={itemVariants} initial="rest" whileHover="hover">
      <Card className="p-6 transition-shadow hover:shadow-xl">
        <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
          <Clock className="w-5 h-5" />
          Average Duration Comparison
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(210, 15%, 92%)"
            />
            <XAxis dataKey="mode" stroke="hsl(172, 43%, 15%)" />
            <YAxis
              stroke="hsl(172, 43%, 15%)"
              label={{
                value: "Minutes",
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
              formatter={(value: number) => [`${value} min`, "Duration"]}
            />
            <Bar
              dataKey="duration"
              radius={[8, 8, 0, 0]}
              animationDuration={800}
            >
              {barColors.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 text-center">
          <p className="text-sm text-[hsl(215,15%,50%)]">
            Centralized:{" "}
            {formatDuration(centralizedData.avg_duration_minutes)} |
            Federated: {formatDuration(federatedData.avg_duration_minutes)}
          </p>
        </div>
      </Card>
    </motion.div>
  );
}
