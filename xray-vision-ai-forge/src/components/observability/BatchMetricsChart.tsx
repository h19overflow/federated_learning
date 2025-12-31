/**
 * Real-time batch-level metrics visualization.
 * Shows accuracy and F1 on a single chart.
 */

import React, { memo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Activity } from 'lucide-react';
import type { BatchMetricsDataPoint } from '@/types/api';

interface BatchMetricsChartProps {
  data: BatchMetricsDataPoint[];
  height?: number;
}

const chartColors = {
  accuracy: 'hsl(172, 63%, 35%)',  // Teal
  f1: 'hsl(210, 60%, 50%)',        // Blue
};

const BatchMetricsChart = memo(function BatchMetricsChart({
  data,
  height = 220,
}: BatchMetricsChartProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex flex-col items-center justify-center bg-[hsl(168_25%_97%)] rounded-xl border border-dashed border-[hsl(168_20%_85%)]"
        style={{ height }}
      >
        <Activity className="h-8 w-8 text-[hsl(168_30%_70%)] mb-2 animate-pulse" />
        <p className="text-sm text-[hsl(215_15%_55%)]">Waiting for metrics...</p>
      </div>
    );
  }

  // Transform data for chart - convert to percentages
  const chartData = data.map((d) => ({
    step: d.step,
    accuracy: d.accuracy !== null ? d.accuracy * 100 : null,
    f1: d.f1 !== null ? d.f1 * 100 : null,
  }));

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
          <XAxis
            dataKey="step"
            tick={{ fill: 'hsl(215, 15%, 50%)', fontSize: 11 }}
            tickLine={{ stroke: 'hsl(210, 15%, 85%)' }}
            label={{ value: 'Step', position: 'bottom', offset: -5, fontSize: 11, fill: 'hsl(215, 15%, 50%)' }}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: 'hsl(215, 15%, 50%)', fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
            tickLine={{ stroke: 'hsl(210, 15%, 85%)' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid hsl(168, 20%, 88%)',
              borderRadius: '12px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              fontSize: '12px',
            }}
            formatter={(value: number) => `${value?.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Accuracy"
            stroke={chartColors.accuracy}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: chartColors.accuracy }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="f1"
            name="F1"
            stroke={chartColors.f1}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: chartColors.f1 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

export default BatchMetricsChart;
