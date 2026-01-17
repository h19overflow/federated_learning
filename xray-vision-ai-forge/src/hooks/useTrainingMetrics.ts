/**
 * Hook for managing real-time training observability metrics.
 * Subscribes to batch_metrics WebSocket events.
 * Implements throttling (500ms) and data windowing (max 200 points).
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type { TrainingProgressWebSocket } from '@/services/websocket';
import type { BatchMetricsData, BatchMetricsDataPoint } from '@/types/api';

const MAX_DATA_POINTS = 200;
const THROTTLE_MS = 500; // 2 Hz updates

interface UseTrainingMetricsReturn {
  batchMetrics: BatchMetricsDataPoint[];
  currentLoss: number | null;
  currentAccuracy: number | null;
  currentF1: number | null;
  isReceiving: boolean;
  reset: () => void;
}

export function useTrainingMetrics(
  ws: TrainingProgressWebSocket | null
): UseTrainingMetricsReturn {
  const [batchMetrics, setBatchMetrics] = useState<BatchMetricsDataPoint[]>([]);
  const [isReceiving, setIsReceiving] = useState(false);

  const batchBufferRef = useRef<BatchMetricsDataPoint[]>([]);
  const lastFlushRef = useRef<number>(0);
  const flushTimerRef = useRef<NodeJS.Timeout | null>(null);

  const flushBuffers = useCallback(() => {
    const now = Date.now();
    if (now - lastFlushRef.current < THROTTLE_MS) return;
    lastFlushRef.current = now;

    if (batchBufferRef.current.length > 0) {
      setBatchMetrics((prev) => {
        const combined = [...prev, ...batchBufferRef.current];
        return combined.slice(-MAX_DATA_POINTS);
      });
      batchBufferRef.current = [];
    }
  }, []);

  const scheduleFlush = useCallback(() => {
    if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
    flushTimerRef.current = setTimeout(flushBuffers, THROTTLE_MS);
  }, [flushBuffers]);

  useEffect(() => {
    if (!ws) return;

    const handleBatchMetrics = (data: BatchMetricsData) => {
      setIsReceiving(true);
      batchBufferRef.current.push({
        step: data.step,
        loss: data.loss,
        accuracy: data.accuracy,
        f1: data.f1,
        timestamp: data.timestamp,
      });
      scheduleFlush();
    };

    const unsubBatch = ws.on('batch_metrics', handleBatchMetrics);

    return () => {
      unsubBatch();
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
    };
  }, [ws, scheduleFlush]);

  const currentLoss = useMemo(() => {
    if (batchMetrics.length === 0) return null;
    return batchMetrics[batchMetrics.length - 1].loss;
  }, [batchMetrics]);

  const currentAccuracy = useMemo(() => {
    if (batchMetrics.length === 0) return null;
    return batchMetrics[batchMetrics.length - 1].accuracy;
  }, [batchMetrics]);

  const currentF1 = useMemo(() => {
    if (batchMetrics.length === 0) return null;
    return batchMetrics[batchMetrics.length - 1].f1;
  }, [batchMetrics]);

  const reset = useCallback(() => {
    setBatchMetrics([]);
    setIsReceiving(false);
    batchBufferRef.current = [];
  }, []);

  return {
    batchMetrics,
    currentLoss,
    currentAccuracy,
    currentF1,
    isReceiving,
    reset,
  };
}
