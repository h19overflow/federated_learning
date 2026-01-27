import { useQuery, UseQueryResult } from "@tanstack/react-query";

/**
 * Mode-specific statistics for training runs.
 */
interface ModeStatistics {
  count: number;
  avg_accuracy: number;
  avg_precision: number;
  avg_recall: number;
  avg_f1: number;
  avg_duration_minutes: number;
}

/**
 * Individual training run performance data.
 */
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

/**
 * Complete analytics data structure returned from the API.
 */
export interface AnalyticsData {
  total_runs: number;
  success_rate: number;
  centralized: ModeStatistics;
  federated: ModeStatistics;
  top_runs: TopRun[];
}

/**
 * Parameters for filtering analytics data.
 */
interface UseAnalyticsDataParams {
  trainingMode: string;
  status: string;
  days: string;
}

/**
 * Fetches analytics data from the backend API.
 *
 * @param params - Query parameters for filtering analytics
 * @returns Promise resolving to AnalyticsData
 */
async function fetchAnalyticsData(
  params: UseAnalyticsDataParams,
): Promise<AnalyticsData> {
  const queryParams = new URLSearchParams();

  if (params.status !== "all") {
    queryParams.append("status", params.status);
  }
  if (params.trainingMode !== "all") {
    queryParams.append("training_mode", params.trainingMode);
  }
  if (params.days !== "all") {
    queryParams.append("days", params.days);
  }

  const response = await fetch(
    `http://localhost:8001/api/runs/analytics/summary?${queryParams}`,
  );

  if (!response.ok) {
    throw new Error("Failed to fetch analytics");
  }

  return response.json();
}

/**
 * Custom hook for fetching and managing analytics data.
 *
 * Handles data fetching with React Query, including caching,
 * automatic refetching, and error handling.
 *
 * @param trainingMode - Filter by training mode ("all", "centralized", "federated")
 * @param status - Filter by run status ("all", "completed", "failed", "in_progress")
 * @param days - Filter by date range ("all", "7", "30")
 * @returns Query result with data, loading state, and error
 *
 * @example
 * ```tsx
 * const { data, isLoading, error } = useAnalyticsData("all", "completed", "7");
 *
 * if (isLoading) return <LoadingSpinner />;
 * if (error) return <ErrorMessage error={error} />;
 * if (!data) return null;
 *
 * return <AnalyticsDisplay data={data} />;
 * ```
 */
export function useAnalyticsData(
  trainingMode: string,
  status: string,
  days: string,
): UseQueryResult<AnalyticsData, Error> {
  return useQuery<AnalyticsData, Error>({
    queryKey: ["analytics", trainingMode, status, days],
    queryFn: () => fetchAnalyticsData({ trainingMode, status, days }),
    staleTime: 30000,
    gcTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}
