import { useMemo } from "react";

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
 * Metrics available for sorting training runs.
 */
export type SortMetric =
  | "best_accuracy"
  | "best_precision"
  | "best_recall"
  | "best_f1";

/**
 * Sort direction for ordering runs.
 */
export type SortDirection = "asc" | "desc";

/**
 * Parameters for sorting training runs.
 */
interface UseSortedRunsParams {
  data: TopRun[] | undefined;
  sortBy: SortMetric;
  sortDirection: SortDirection;
}

/**
 * Sorts training runs by a specified metric.
 *
 * @param runs - Array of training runs to sort
 * @param sortBy - Metric to sort by
 * @param sortDirection - Sort order (ascending or descending)
 * @returns Sorted array of runs, limited to top 10
 */
function sortRuns(
  runs: TopRun[],
  sortBy: SortMetric,
  sortDirection: SortDirection,
): TopRun[] {
  const sorted = [...runs].sort((a, b) => {
    const aValue = a[sortBy] ?? 0;
    const bValue = b[sortBy] ?? 0;

    if (sortDirection === "desc") {
      return bValue - aValue;
    } else {
      return aValue - bValue;
    }
  });

  return sorted.slice(0, 10);
}

/**
 * Custom hook for sorting and limiting training runs.
 *
 * Memoizes the sorting operation to prevent unnecessary recalculations.
 * Returns only the top 10 runs based on the selected metric and direction.
 *
 * @param data - Array of training runs from analytics data
 * @param sortBy - Performance metric to sort by
 * @param sortDirection - Direction to sort ("asc" or "desc")
 * @returns Memoized array of top 10 sorted runs
 *
 * @example
 * ```tsx
 * const { data } = useAnalyticsData("all", "completed", "7");
 * const sortedRuns = useSortedRuns(data?.top_runs, "best_f1", "desc");
 *
 * return (
 *   <Table>
 *     {sortedRuns.map(run => (
 *       <TableRow key={run.run_id}>
 *         <TableCell>{run.run_id}</TableCell>
 *         <TableCell>{run.best_f1}</TableCell>
 *       </TableRow>
 *     ))}
 *   </Table>
 * );
 * ```
 */
export function useSortedRuns(
  data: TopRun[] | undefined,
  sortBy: SortMetric,
  sortDirection: SortDirection,
): TopRun[] {
  return useMemo(() => {
    if (!data) {
      return [];
    }

    return sortRuns(data, sortBy, sortDirection);
  }, [data, sortBy, sortDirection]);
}
