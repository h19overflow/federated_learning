import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { TrendingUp, BarChart3, Users, CheckCircle2, Trophy, Activity, ArrowUp, Loader2, AlertCircle } from 'lucide-react';

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

const AnalyticsTab = () => {
  const [trainingMode, setTrainingMode] = useState<string>('all');
  const [status, setStatus] = useState<string>('completed');
  const [days, setDays] = useState<string>('all');

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
  });

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
      metric: 'Accuracy',
      Centralized: centralizedData.avg_accuracy ? Number((centralizedData.avg_accuracy * 100).toFixed(1)) : 0,
      Federated: federatedData.avg_accuracy ? Number((federatedData.avg_accuracy * 100).toFixed(1)) : 0,
    },
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

  return (
    <div className="space-y-8 p-6">
      {/* Filters Section */}
      <div className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium text-[hsl(172,43%,15%)] mb-2 block">Training Mode</label>
          <Select value={trainingMode} onValueChange={setTrainingMode}>
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
          <Select value={status} onValueChange={setStatus}>
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
          <Select value={days} onValueChange={setDays}>
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
              <TableCell className="font-medium">Accuracy</TableCell>
              <TableCell className="text-center">{formatPercentage(centralizedData.avg_accuracy)}</TableCell>
              <TableCell className="text-center">{formatPercentage(federatedData.avg_accuracy)}</TableCell>
              <TableCell className="text-center">
                {getWinner(centralizedData.avg_accuracy, federatedData.avg_accuracy) === 'centralized' && (
                  <Badge className="bg-[hsl(172,63%,28%)] hover:bg-[hsl(172,63%,22%)]">
                    Cent. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
                {getWinner(centralizedData.avg_accuracy, federatedData.avg_accuracy) === 'federated' && (
                  <Badge className="bg-[hsl(210,60%,50%)] hover:bg-[hsl(210,60%,45%)]">
                    Fed. <ArrowUp className="w-3 h-3 ml-1 inline" />
                  </Badge>
                )}
              </TableCell>
            </TableRow>
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
              <TableHead className="font-semibold text-right">Accuracy</TableHead>
              <TableHead className="font-semibold text-right">Duration</TableHead>
              <TableHead className="font-semibold">Date</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.top_runs.slice(0, 10).map((run, index) => (
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
                <TableCell className="text-right font-semibold">{formatPercentage(run.best_accuracy)}</TableCell>
                <TableCell className="text-right text-[hsl(215,15%,50%)]">{formatDuration(run.duration_minutes)}</TableCell>
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
