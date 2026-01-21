import React, { useState, startTransition, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  BarChart3,
  Users,
  CheckCircle2,
  Trophy,
  Activity,
  ArrowUp,
  Loader2,
  AlertCircle,
  Clock,
  PieChartIcon,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface ModeStatistics {
  count: number;
  avg_accuracy: number;
  avg_precision: number;
  avg_recall: number;
  avg_f1: number;
  avg_duration_minutes: number;
}

interface TopRun {
  run_id: number;
  training_mode: string;
  best_accuracy: number;
  best_precision: number;
  best_recall: number;
  best_f1: number;
  duration_minutes: number;
  start_time: string;
  status: string;
}

interface AnalyticsData {
  total_runs: number;
  success_rate: number;
  centralized: ModeStatistics;
  federated: ModeStatistics;
  top_runs: TopRun[];
}

type SortMetric =
  | "best_accuracy"
  | "best_precision"
  | "best_recall"
  | "best_f1";
type SortDirection = "asc" | "desc";

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    },
  },
};

const cardHoverVariants = {
  rest: { scale: 1, y: 0 },
  hover: {
    scale: 1.02,
    y: -4,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 25,
    },
  },
};

const tableRowVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: (i: number) => ({
    opacity: 1,
    x: 0,
    transition: {
      delay: i * 0.05,
      type: "spring",
      stiffness: 100,
      damping: 15,
    },
  }),
};

const AnalyticsTab = () => {
  const [trainingMode, setTrainingMode] = useState<string>("all");
  const [status, setStatus] = useState<string>("all");
  const [days, setDays] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortMetric>("best_f1");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const { data, isLoading, error } = useQuery<AnalyticsData>({
    queryKey: ["analytics", trainingMode, status, days],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (status !== "all") params.append("status", status);
      if (trainingMode !== "all") params.append("training_mode", trainingMode);
      if (days !== "all") params.append("days", days);

      const response = await fetch(
        `http://localhost:8001/api/runs/analytics/summary?${params}`,
      );
      if (!response.ok) throw new Error("Failed to fetch analytics");
      return response.json();
    },
    staleTime: 30000,
    gcTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });

  // Memoized sorted runs
  const sortedTopRuns = useMemo(() => {
    if (!data?.top_runs) return [];

    const sorted = [...data.top_runs].sort((a, b) => {
      const aValue = a[sortBy] ?? 0;
      const bValue = b[sortBy] ?? 0;

      if (sortDirection === "desc") {
        return bValue - aValue;
      } else {
        return aValue - bValue;
      }
    });

    return sorted.slice(0, 10);
  }, [data?.top_runs, sortBy, sortDirection]);

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

  const centralizedData = data.centralized || {
    count: 0,
    avg_accuracy: 0,
    avg_precision: 0,
    avg_recall: 0,
    avg_f1: 0,
    avg_duration_minutes: 0,
  };

  const federatedData = data.federated || {
    count: 0,
    avg_accuracy: 0,
    avg_precision: 0,
    avg_recall: 0,
    avg_f1: 0,
    avg_duration_minutes: 0,
  };

  const formatPercentage = (value: number | null | undefined) => {
    if (value === null || value === undefined) return "N/A";
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDuration = (minutes: number | null | undefined) => {
    if (minutes === null || minutes === undefined) return "N/A";
    return `${minutes.toFixed(1)} min`;
  };

  const chartData = [
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

  const getWinner = (
    centValue: number,
    fedValue: number,
  ): "centralized" | "federated" | "tie" => {
    if (Math.abs(centValue - fedValue) < 0.001) return "tie";
    return centValue > fedValue ? "centralized" : "federated";
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return dateString;
    }
  };

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
      value: centralizedData.count,
      color: "hsl(172,63%,28%)",
    },
    {
      icon: Users,
      label: "Federated",
      value: federatedData.count,
      color: "hsl(210,60%,50%)",
    },
    {
      icon: CheckCircle2,
      label: "Success Rate",
      value: formatPercentage(data.success_rate),
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
      {/* Filters Section */}
      <motion.div variants={itemVariants} className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
            Training Mode
          </label>
          <Select
            value={trainingMode}
            onValueChange={(value) =>
              startTransition(() => setTrainingMode(value))
            }
          >
            <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Modes</SelectItem>
              <SelectItem value="centralized">Centralized</SelectItem>
              <SelectItem value="federated">Federated</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
            Status
          </label>
          <Select
            value={status}
            onValueChange={(value) => startTransition(() => setStatus(value))}
          >
            <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
              <SelectItem value="in_progress">In Progress</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">
            Date Range
          </label>
          <Select
            value={days}
            onValueChange={(value) => startTransition(() => setDays(value))}
          >
            <SelectTrigger className="w-full transition-all hover:border-[hsl(172,63%,28%)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="7">Last 7 Days</SelectItem>
              <SelectItem value="30">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </motion.div>

      {/* Quick Stats Cards */}
      <motion.div
        variants={containerVariants}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <AnimatePresence mode="wait">
          {statsCards.map((card, index) => (
            <motion.div
              key={card.label}
              custom={index}
              variants={itemVariants}
              initial="rest"
              whileHover="hover"
            >
              <motion.div variants={cardHoverVariants}>
                <Card className="p-6 border-[hsl(210,15%,92%)] transition-shadow hover:shadow-xl">
                  <div className="flex items-center gap-4">
                    <motion.div
                      className="p-3 rounded-xl bg-[hsl(168,20%,95%)]"
                      whileHover={{ rotate: 5, scale: 1.1 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      <card.icon
                        className="w-6 h-6"
                        style={{ color: card.color }}
                      />
                    </motion.div>
                    <div>
                      <p className="text-sm text-[hsl(215,15%,50%)]">
                        {card.label}
                      </p>
                      <motion.p
                        className="text-3xl font-bold text-[hsl(172,43%,15%)]"
                        initial={{ scale: 0.5, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{
                          delay: index * 0.1 + 0.2,
                          type: "spring",
                        }}
                      >
                        {card.value}
                      </motion.p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* Pie Chart and Duration Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div variants={itemVariants} initial="rest" whileHover="hover">
          <Card className="p-6 transition-shadow hover:shadow-xl">
            <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
              <PieChartIcon className="w-5 h-5" />
              Training Mode Distribution
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
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
                  ]}
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
                  {[
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
                  ].map((entry, index) => (
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

        <motion.div variants={itemVariants} initial="rest" whileHover="hover">
          <Card className="p-6 transition-shadow hover:shadow-xl">
            <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Average Duration Comparison
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={[
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
                ]}
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
                  {[
                    { mode: "Centralized", color: "hsl(172, 63%, 28%)" },
                    { mode: "Federated", color: "hsl(210, 60%, 50%)" },
                  ].map((entry, index) => (
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
      </div>

      {/* Comparison Table */}
      <motion.div variants={itemVariants} initial="rest" whileHover="hover">
        <motion.div variants={cardHoverVariants}>
          <Card className="p-6 transition-shadow hover:shadow-xl">
            <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Centralized vs Federated Comparison
            </h2>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold">Metric</TableHead>
                  <TableHead className="font-semibold text-center">
                    Centralized
                  </TableHead>
                  <TableHead className="font-semibold text-center">
                    Federated
                  </TableHead>
                  <TableHead className="font-semibold text-center">
                    Winner
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  {
                    label: "Precision",
                    cent: centralizedData.avg_precision,
                    fed: federatedData.avg_precision,
                  },
                  {
                    label: "Recall",
                    cent: centralizedData.avg_recall,
                    fed: federatedData.avg_recall,
                  },
                  {
                    label: "F1 Score",
                    cent: centralizedData.avg_f1,
                    fed: federatedData.avg_f1,
                  },
                  {
                    label: "Avg Duration",
                    cent: centralizedData.avg_duration_minutes,
                    fed: federatedData.avg_duration_minutes,
                    isDuration: true,
                  },
                ].map((row, i) => (
                  <motion.tr
                    key={row.label}
                    custom={i}
                    variants={tableRowVariants}
                    initial="hidden"
                    animate="visible"
                    className="hover:bg-[hsl(168,20%,98%)] transition-colors"
                  >
                    <TableCell className="font-medium">{row.label}</TableCell>
                    <TableCell className="text-center">
                      {row.isDuration
                        ? formatDuration(row.cent)
                        : formatPercentage(row.cent)}
                    </TableCell>
                    <TableCell className="text-center">
                      {row.isDuration
                        ? formatDuration(row.fed)
                        : formatPercentage(row.fed)}
                    </TableCell>
                    <TableCell className="text-center">
                      {(() => {
                        const winner = row.isDuration
                          ? getWinner(row.fed, row.cent)
                          : getWinner(row.cent, row.fed);
                        if (winner === "tie") return null;
                        return (
                          <motion.div
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: i * 0.1 + 0.3 }}
                          >
                            <Badge
                              className={
                                winner === "centralized"
                                  ? "bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]"
                                  : "bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]"
                              }
                            >
                              {winner === "centralized" ? "Cent." : "Fed."}{" "}
                              <ArrowUp className="w-3 h-3 ml-1 inline" />
                            </Badge>
                          </motion.div>
                        );
                      })()}
                    </TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </Card>
        </motion.div>
      </motion.div>

      {/* Performance Chart */}
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

      {/* Top Runs Table */}
      <motion.div variants={itemVariants} initial="rest" whileHover="hover">
        <motion.div variants={cardHoverVariants}>
          <Card className="p-6 transition-shadow hover:shadow-xl">
            <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
              <Trophy className="w-5 h-5" />
              Top Performing Runs
            </h2>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold w-12">#</TableHead>
                  <TableHead className="font-semibold">Run ID</TableHead>
                  <TableHead className="font-semibold">Mode</TableHead>
                  <TableHead
                    className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                    onClick={() => handleSort("best_accuracy")}
                  >
                    Accuracy {getSortIcon("best_accuracy")}
                  </TableHead>
                  <TableHead
                    className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                    onClick={() => handleSort("best_precision")}
                  >
                    Precision {getSortIcon("best_precision")}
                  </TableHead>
                  <TableHead
                    className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                    onClick={() => handleSort("best_recall")}
                  >
                    Recall {getSortIcon("best_recall")}
                  </TableHead>
                  <TableHead
                    className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                    onClick={() => handleSort("best_f1")}
                  >
                    F1 Score {getSortIcon("best_f1")}
                  </TableHead>
                  <TableHead className="font-semibold text-right">
                    Duration
                  </TableHead>
                  <TableHead className="font-semibold">Status</TableHead>
                  <TableHead className="font-semibold">Date</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <AnimatePresence mode="wait">
                  {sortedTopRuns.map((run, index) => (
                    <motion.tr
                      key={run.run_id}
                      custom={index}
                      variants={tableRowVariants}
                      initial="hidden"
                      animate="visible"
                      exit={{ opacity: 0, x: -20 }}
                      className="hover:bg-[hsl(168,20%,98%)] transition-colors"
                    >
                      <TableCell className="font-medium text-[hsl(215,15%,50%)]">
                        {index + 1}
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {run.run_id}
                      </TableCell>
                      <TableCell>
                        <motion.div
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ delay: index * 0.05 + 0.2 }}
                        >
                          <Badge
                            className={
                              run.training_mode === "centralized"
                                ? "bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]"
                                : "bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]"
                            }
                          >
                            {run.training_mode === "centralized"
                              ? "Centralized"
                              : "Federated"}
                          </Badge>
                        </motion.div>
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatPercentage(run.best_accuracy)}
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatPercentage(run.best_precision)}
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatPercentage(run.best_recall)}
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatPercentage(run.best_f1)}
                      </TableCell>
                      <TableCell className="text-right text-[hsl(215,15%,50%)]">
                        {formatDuration(run.duration_minutes)}
                      </TableCell>
                      <TableCell>
                        <motion.div
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ delay: index * 0.05 + 0.3 }}
                        >
                          <Badge
                            className={
                              run.status === "completed"
                                ? "bg-[hsl(168,40%,45%)] hover:bg-[hsl(168,40%,40%)]"
                                : run.status === "failed"
                                  ? "bg-red-500 hover:bg-red-600"
                                  : "bg-yellow-500 hover:bg-yellow-600"
                            }
                          >
                            {run.status === "completed"
                              ? "Completed"
                              : run.status === "failed"
                                ? "Failed"
                                : "In Progress"}
                          </Badge>
                        </motion.div>
                      </TableCell>
                      <TableCell className="text-[hsl(215,15%,50%)]">
                        {formatDate(run.start_time)}
                      </TableCell>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </TableBody>
            </Table>
          </Card>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default AnalyticsTab;
