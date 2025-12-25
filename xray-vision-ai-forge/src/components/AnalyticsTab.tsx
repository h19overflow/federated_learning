import React, { useState, startTransition, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, BarChart3, Users, CheckCircle2, Trophy, Activity, ArrowUp, Loader2, AlertCircle, Clock, PieChartIcon, ArrowUpDown, ChevronDown, ChevronUp } from 'lucide-react';

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

type SortMetric = 'best_accuracy' | 'best_precision' | 'best_recall' | 'best_f1';
type SortDirection = 'asc' | 'desc';

const AnalyticsTab = () => {
  const [trainingMode, setTrainingMode] = useState<string>('all');
  const [status, setStatus] = useState<string>('completed');
  const [days, setDays] = useState<string>('all');
  const [sortBy, setSortBy] = useState<SortMetric>('best_f1');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const { data, isLoading, error } = useQuery<AnalyticsData>({
    queryKey: ['analytics', trainingMode, status, days],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (status !== 'all') params.append('status', status);
      if (trainingMode !== 'all') params.append('training_mode', trainingMode);
      if (days !== 'all') params.append('days', days);

      const response = await fetch(`http://localhost:8001/api/runs/analytics/summary?${params}`);
      if (!response.ok) throw new Error('Failed to fetch analytics');
      return response.json();
    },
    staleTime: 30000, // 30 seconds - prevents unnecessary refetches
    gcTime: 5 * 60 * 1000, // 5 minutes - keeps data in cache longer
    refetchOnWindowFocus: false, // Prevents refetch when tab regains focus
  });

  // Memoized sorted runs - MUST be before early returns (Rules of Hooks)
  const sortedTopRuns = useMemo(() => {
    if (!data?.top_runs) return [];

    const sorted = [...data.top_runs].sort((a, b) => {
      const aValue = a[sortBy] ?? 0;
      const bValue = b[sortBy] ?? 0;

      if (sortDirection === 'desc') {
        return bValue - aValue;
      } else {
        return aValue - bValue;
      }
    });

    return sorted.slice(0, 10);
  }, [data?.top_runs, sortBy, sortDirection]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-[hsl(172,63%,28%)]" />
          <p className="text-[hsl(215,15%,50%)]">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
          <p className="text-red-600">Failed to load analytics</p>
          <p className="text-sm text-[hsl(215,15%,50%)] mt-2">{(error as Error).message}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  // Ensure we have data structures even if empty
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
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDuration = (minutes: number | null | undefined) => {
    if (minutes === null || minutes === undefined) return 'N/A';
    return `${minutes.toFixed(1)} min`;
  };

  const chartData = [
    {
      metric: 'Precision',
      Centralized: centralizedData.avg_precision ? Number((centralizedData.avg_precision * 100).toFixed(1)) : 0,
      Federated: federatedData.avg_precision ? Number((federatedData.avg_precision * 100).toFixed(1)) : 0,
    },
    {
      metric: 'Recall',
      Centralized: centralizedData.avg_recall ? Number((centralizedData.avg_recall * 100).toFixed(1)) : 0,
      Federated: federatedData.avg_recall ? Number((federatedData.avg_recall * 100).toFixed(1)) : 0,
    },
    {
      metric: 'F1 Score',
      Centralized: centralizedData.avg_f1 ? Number((centralizedData.avg_f1 * 100).toFixed(1)) : 0,
      Federated: federatedData.avg_f1 ? Number((federatedData.avg_f1 * 100).toFixed(1)) : 0,
    },
  ];

  const getWinner = (centValue: number, fedValue: number): 'centralized' | 'federated' | 'tie' => {
    if (Math.abs(centValue - fedValue) < 0.001) return 'tie';
    return centValue > fedValue ? 'centralized' : 'federated';
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return dateString;
    }
  };

  const handleSort = (metric: SortMetric) => {
    if (sortBy === metric) {
      // Toggle direction if clicking same column
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // New column, default to descending (best first)
      setSortBy(metric);
      setSortDirection('desc');
    }
  };

  const getSortIcon = (metric: SortMetric) => {
    if (sortBy !== metric) {
      return <ArrowUpDown className="w-4 h-4 ml-1 inline opacity-30" />;
    }
    return sortDirection === 'desc'
      ? <ChevronDown className="w-4 h-4 ml-1 inline" />
      : <ChevronUp className="w-4 h-4 ml-1 inline" />;
  };

  return (
    <div className="space-y-8 p-6">
      {/* Filters Section */}
      <div className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">Training Mode</label>
          <Select value={trainingMode} onValueChange={(value) => startTransition(() => setTrainingMode(value))}>
            <SelectTrigger className="w-full">
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
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">Status</label>
          <Select value={status} onValueChange={(value) => startTransition(() => setStatus(value))}>
            <SelectTrigger className="w-full">
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
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">Date Range</label>
          <Select value={days} onValueChange={(value) => startTransition(() => setDays(value))}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="7">Last 7 Days</SelectItem>
              <SelectItem value="30">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Quick Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6 border-[hsl(210,15%,92%)] hover:shadow-lg transition-shadow">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-[hsl(168,20%,95%)]">
              <BarChart3 className="w-6 h-6 text-[hsl(172,63%,28%)]" />
            </div>
            <div>
              <p className="text-sm text-[hsl(215,15%,50%)]">Total Runs</p>
              <p className="text-3xl font-bold text-[hsl(172,43%,15%)]">{data.total_runs}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 border-[hsl(210,15%,92%)] hover:shadow-lg transition-shadow">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-[hsl(168,20%,95%)]">
              <Activity className="w-6 h-6 text-[hsl(172,63%,28%)]" />
            </div>
            <div>
              <p className="text-sm text-[hsl(215,15%,50%)]">Centralized</p>
              <p className="text-3xl font-bold text-[hsl(172,43%,15%)]">{centralizedData.count}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 border-[hsl(210,15%,92%)] hover:shadow-lg transition-shadow">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-[hsl(168,20%,95%)]">
              <Users className="w-6 h-6 text-[hsl(210,60%,50%)]" />
            </div>
            <div>
              <p className="text-sm text-[hsl(215,15%,50%)]">Federated</p>
              <p className="text-3xl font-bold text-[hsl(172,43%,15%)]">{federatedData.count}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 border-[hsl(210,15%,92%)] hover:shadow-lg transition-shadow">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-[hsl(168,20%,95%)]">
              <CheckCircle2 className="w-6 h-6 text-[hsl(168,40%,45%)]" />
            </div>
            <div>
              <p className="text-sm text-[hsl(215,15%,50%)]">Success Rate</p>
              <p className="text-3xl font-bold text-[hsl(172,43%,15%)]">{formatPercentage(data.success_rate)}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* New Row: Pie Chart and Duration Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart Widget */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
            <PieChartIcon className="w-5 h-5" />
            Training Mode Distribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Centralized', value: centralizedData.count, color: 'hsl(172, 63%, 28%)' },
                  { name: 'Federated', value: federatedData.count, color: 'hsl(210, 60%, 50%)' }
                ]}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {[
                  { name: 'Centralized', value: centralizedData.count, color: 'hsl(172, 63%, 28%)' },
                  { name: 'Federated', value: federatedData.count, color: 'hsl(210, 60%, 50%)' }
                ].map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid hsl(210, 15%, 92%)',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 flex justify-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-[hsl(172,63%,28%)]"></div>
              <span className="text-sm text-[hsl(215,15%,50%)]">Centralized ({centralizedData.count})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-[hsl(210,60%,50%)]"></div>
              <span className="text-sm text-[hsl(215,15%,50%)]">Federated ({federatedData.count})</span>
            </div>
          </div>
        </Card>

        {/* Average Duration Visualization */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
            <Clock className="w-5 h-5" />
            Average Duration Comparison
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                {
                  mode: 'Centralized',
                  duration: centralizedData.avg_duration_minutes ? Number(centralizedData.avg_duration_minutes.toFixed(1)) : 0,
                },
                {
                  mode: 'Federated',
                  duration: federatedData.avg_duration_minutes ? Number(federatedData.avg_duration_minutes.toFixed(1)) : 0,
                }
              ]}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 92%)" />
              <XAxis dataKey="mode" stroke="hsl(172, 43%, 15%)" />
              <YAxis stroke="hsl(172, 43%, 15%)" label={{ value: 'Minutes', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid hsl(210, 15%, 92%)',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                }}
                formatter={(value: number) => [`${value} min`, 'Duration']}
              />
              <Bar dataKey="duration" radius={[8, 8, 0, 0]}>
                {[
                  { mode: 'Centralized', color: 'hsl(172, 63%, 28%)' },
                  { mode: 'Federated', color: 'hsl(210, 60%, 50%)' }
                ].map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 text-center">
            <p className="text-sm text-[hsl(215,15%,50%)]">
              Centralized: {formatDuration(centralizedData.avg_duration_minutes)} |
              Federated: {formatDuration(federatedData.avg_duration_minutes)}
            </p>
          </div>
        </Card>
      </div>

      {/* Comparison Table */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Centralized vs Federated Comparison
        </h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="font-semibold">Metric</TableHead>
              <TableHead className="font-semibold text-center">Centralized</TableHead>
              <TableHead className="font-semibold text-center">Federated</TableHead>
              <TableHead className="font-semibold text-center">Winner</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow className="hover:bg-[hsl(168,20%,98%)]">
              <TableCell className="font-medium">Precision</TableCell>
              <TableCell className="text-center">{formatPercentage(centralizedData.avg_precision)}</TableCell>
              <TableCell className="text-center">{formatPercentage(federatedData.avg_precision)}</TableCell>
              <TableCell className="text-center">
                {getWinner(centralizedData.avg_precision, federatedData.avg_precision) === 'centralized' && (
                  <Badge className="bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]">
                    Cent. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
                {getWinner(centralizedData.avg_precision, federatedData.avg_precision) === 'federated' && (
                  <Badge className="bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]">
                    Fed. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
              </TableCell>
            </TableRow>
            <TableRow className="hover:bg-[hsl(168,20%,98%)]">
              <TableCell className="font-medium">Recall</TableCell>
              <TableCell className="text-center">{formatPercentage(centralizedData.avg_recall)}</TableCell>
              <TableCell className="text-center">{formatPercentage(federatedData.avg_recall)}</TableCell>
              <TableCell className="text-center">
                {getWinner(centralizedData.avg_recall, federatedData.avg_recall) === 'centralized' && (
                  <Badge className="bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]">
                    Cent. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
                {getWinner(centralizedData.avg_recall, federatedData.avg_recall) === 'federated' && (
                  <Badge className="bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]">
                    Fed. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
              </TableCell>
            </TableRow>
            <TableRow className="hover:bg-[hsl(168,20%,98%)]">
              <TableCell className="font-medium">F1 Score</TableCell>
              <TableCell className="text-center">{formatPercentage(centralizedData.avg_f1)}</TableCell>
              <TableCell className="text-center">{formatPercentage(federatedData.avg_f1)}</TableCell>
              <TableCell className="text-center">
                {getWinner(centralizedData.avg_f1, federatedData.avg_f1) === 'centralized' && (
                  <Badge className="bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]">
                    Cent. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
                {getWinner(centralizedData.avg_f1, federatedData.avg_f1) === 'federated' && (
                  <Badge className="bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]">
                    Fed. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
              </TableCell>
            </TableRow>
            <TableRow className="hover:bg-[hsl(168,20%,98%)]">
              <TableCell className="font-medium">Avg Duration</TableCell>
              <TableCell className="text-center">{formatDuration(centralizedData.avg_duration_minutes)}</TableCell>
              <TableCell className="text-center">{formatDuration(federatedData.avg_duration_minutes)}</TableCell>
              <TableCell className="text-center">
                {getWinner(federatedData.avg_duration_minutes, centralizedData.avg_duration_minutes) === 'federated' && (
                  <Badge className="bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]">
                    Fed. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
                {getWinner(federatedData.avg_duration_minutes, centralizedData.avg_duration_minutes) === 'centralized' && (
                  <Badge className="bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]">
                    Cent. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Card>

      {/* Performance Chart */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold text-[hsl(172,43%,15%)] mb-6 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Performance Comparison
        </h2>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 92%)" />
            <XAxis dataKey="metric" stroke="hsl(172, 43%, 15%)" />
            <YAxis stroke="hsl(172, 43%, 15%)" domain={[0, 100]} label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid hsl(210, 15%, 92%)',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              }}
            />
            <Legend />
            <Bar dataKey="Centralized" fill="hsl(172, 63%, 28%)" radius={[8, 8, 0, 0]} />
            <Bar dataKey="Federated" fill="hsl(210, 60%, 50%)" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top Runs Table */}
      <Card className="p-6">
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
                onClick={() => handleSort('best_accuracy')}
              >
                Accuracy {getSortIcon('best_accuracy')}
              </TableHead>
              <TableHead
                className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                onClick={() => handleSort('best_precision')}
              >
                Precision {getSortIcon('best_precision')}
              </TableHead>
              <TableHead
                className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                onClick={() => handleSort('best_recall')}
              >
                Recall {getSortIcon('best_recall')}
              </TableHead>
              <TableHead
                className="font-semibold text-right cursor-pointer hover:text-[hsl(172,63%,28%)] transition-colors select-none"
                onClick={() => handleSort('best_f1')}
              >
                F1 Score {getSortIcon('best_f1')}
              </TableHead>
              <TableHead className="font-semibold text-right">Duration</TableHead>
              <TableHead className="font-semibold">Status</TableHead>
              <TableHead className="font-semibold">Date</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedTopRuns.map((run, index) => (
              <TableRow key={run.run_id} className="hover:bg-[hsl(168,20%,98%)]">
                <TableCell className="font-medium text-[hsl(215,15%,50%)]">{index + 1}</TableCell>
                <TableCell className="font-mono text-sm">{run.run_id}</TableCell>
                <TableCell>
                  <Badge className={run.training_mode === 'centralized'
                    ? 'bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]'
                    : 'bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]'
                  }>
                    {run.training_mode === 'centralized' ? 'Centralized' : 'Federated'}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-medium">{formatPercentage(run.best_accuracy)}</TableCell>
                <TableCell className="text-right font-medium">{formatPercentage(run.best_precision)}</TableCell>
                <TableCell className="text-right font-medium">{formatPercentage(run.best_recall)}</TableCell>
                <TableCell className="text-right font-medium">{formatPercentage(run.best_f1)}</TableCell>
                <TableCell className="text-right text-[hsl(215,15%,50%)]">{formatDuration(run.duration_minutes)}</TableCell>
                <TableCell>
                  <Badge className={
                    run.status === 'completed'
                      ? 'bg-[hsl(168,40%,45%)] hover:bg-[hsl(168,40%,40%)]'
                      : run.status === 'failed'
                      ? 'bg-red-500 hover:bg-red-600'
                      : 'bg-yellow-500 hover:bg-yellow-600'
                  }>
                    {run.status === 'completed' ? 'Completed' : run.status === 'failed' ? 'Failed' : 'In Progress'}
                  </Badge>
                </TableCell>
                <TableCell className="text-[hsl(215,15%,50%)]">{formatDate(run.start_time)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default AnalyticsTab;
