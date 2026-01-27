/**
 * Analytics hooks for data fetching and sorting.
 *
 * This module provides custom React hooks for managing analytics data
 * and sorting training runs by performance metrics.
 */

export { useAnalyticsData } from "./useAnalyticsData";
export type { AnalyticsData } from "./useAnalyticsData";

export { useSortedRuns } from "./useSortedRuns";
export type { SortMetric, SortDirection } from "./useSortedRuns";
