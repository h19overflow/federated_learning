import React, { useState, startTransition } from "react";
import { motion } from "framer-motion";
import {
  Loader2,
  AlertCircle,
  BarChart3,
  Users,
  CheckCircle2,
  Activity,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

import { FilterControls } from "./FilterControls";
import { StatsCards } from "./StatsCards";
import { ModeDistributionChart } from "./ModeDistributionChart";
import { DurationComparisonChart } from "./DurationComparisonChart";
import { ComparisonTable } from "./ComparisonTable";
import { PerformanceChart } from "./PerformanceChart";
import { TopRunsTable } from "./TopRunsTable";

import { useAnalyticsData } from "./hooks/useAnalyticsData";
import { useSortedRuns } from "./hooks/useSortedRuns";

import type { SortMetric, SortDirection } from "./types";
import { containerVariants } from "./animationVariants";

const AnalyticsTab = () => {
  const [trainingMode, setTrainingMode] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [days, setDays] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortMetric>("best_f1");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const { data, isLoading, error } = useAnalyticsData(
    trainingMode,
    status,
    days
  );

  const sortedTopRuns = useSortedRuns(
    data?.top_runs,
    sortBy,
    sortDirection
  );

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-center min-h-[400px]"
      >
        <div className="text-center">
          <motion.div
            animate={{
              rotate: 360,
              scale: [1, 1.1, 1],
            }}
            transition={{
              rotate: { duration: 1, repeat: Infinity, ease: "linear" },
              scale: { duration: 0.8, repeat: Infinity, ease: "easeInOut" },
            }}
          >
            <Loader2 className="w-12 h-12 mx-auto mb-4 text-[hsl(172,63%,28%)]" />
          </motion.div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-[hsl(215,15%,50%)]"
          >
            Loading analytics...
          </motion.p>
        </div>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center justify-center min-h-[400px]"
      >
        <div className="text-center">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
          <p className="text-red-600">Failed to load analytics</p>
          <p className="text-sm text-[hsl(215,15%,50%)] mt-2">
            {(error as Error).message}
          </p>
        </div>
      </motion.div>
    );
  }

  if (!data) return null;

  const handleSort = (metric: SortMetric) => {
    if (sortBy === metric) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortBy(metric);
      setSortDirection("desc");
    }
  };

  const getSortIcon = (metric: SortMetric) => {
    if (sortBy !== metric) {
      return <ArrowUpDown className="w-4 h-4 ml-1 inline opacity-30" />;
    }
    return sortDirection === "desc" ? (
      <ChevronDown className="w-4 h-4 ml-1 inline" />
    ) : (
      <ChevronUp className="w-4 h-4 ml-1 inline" />
    );
  };

  const statsCards = [
    {
      icon: BarChart3,
      label: "Total Runs",
      value: data.total_runs,
      color: "hsl(172,63%,28%)",
    },
    {
      icon: Activity,
      label: "Centralized",
      value: data.centralized?.count || 0,
      color: "hsl(172,63%,28%)",
    },
    {
      icon: Users,
      label: "Federated",
      value: data.federated?.count || 0,
      color: "hsl(210,60%,50%)",
    },
    {
      icon: CheckCircle2,
      label: "Success Rate",
      value: `${(data.success_rate * 100).toFixed(1)}%`,
      color: "hsl(168,40%,45%)",
    },
  ];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8 p-6"
    >
      <FilterControls
        trainingMode={trainingMode}
        status={status}
        days={days}
        setTrainingMode={(value) =>
          startTransition(() => setTrainingMode(value))
        }
        setStatus={(value) => startTransition(() => setStatus(value))}
        setDays={(value) => startTransition(() => setDays(value))}
      />

      <StatsCards statsCards={statsCards} />

      {data.centralized && data.federated && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ModeDistributionChart
              centralizedData={data.centralized}
              federatedData={data.federated}
            />

            <DurationComparisonChart
              centralizedData={data.centralized}
              federatedData={data.federated}
            />
          </div>

          <ComparisonTable
            centralizedData={data.centralized}
            federatedData={data.federated}
          />

          <PerformanceChart
            centralizedData={data.centralized}
            federatedData={data.federated}
          />
        </>
      )}

      <TopRunsTable
        sortedTopRuns={sortedTopRuns}
        sortBy={sortBy}
        sortDirection={sortDirection}
        handleSort={handleSort}
        getSortIcon={getSortIcon}
      />
    </motion.div>
  );
};

export default AnalyticsTab;
