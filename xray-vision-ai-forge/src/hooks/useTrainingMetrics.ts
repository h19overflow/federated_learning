/**
 * Hook for managing real-time training observability metrics.
 * Subscribes to batch_metrics and epoch_end WebSocket events.
 * Subscribes to batch_metrics and epoch_end WebSocket events.
 * Implements throttling (500ms) and data windowing (max 200 points).
 */

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import type { TrainingProgressWebSocket } from "@/services/websocket";
import type { BatchMetricsData, BatchMetricsDataPoint, EpochEndData } from "@/types/api";
import type { BatchMetricsData, BatchMetricsDataPoint, EpochEndData } from "@/types/api";

const MAX_DATA_POINTS = 200;
const THROTTLE_MS = 500; // 2 Hz updates

interface ConfusionMatrixValues {
  truePositives: number | null;
  trueNegatives: number | null;
  falsePositives: number | null;
  falseNegatives: number | null;
}

interface UseTrainingMetricsReturn {
  batchMetrics: BatchMetricsDataPoint[];
  currentLoss: number | null;
  currentAccuracy: number | null;
  currentF1: number | null;
  isReceiving: boolean;
  confusionMatrix: ConfusionMatrixValues;
  reset: () => void;
}

export function useTrainingMetrics(
  ws: TrainingProgressWebSocket | null,
): UseTrainingMetricsReturn {
  const [batchMetrics, setBatchMetrics] = useState<BatchMetricsDataPoint[]>([]);
  const [isReceiving, setIsReceiving] = useState(false);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixData | null>(null);

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

    const handleEpochEnd = (data: EpochEndData) => {
      setIsReceiving(true);
      console.log("[useTrainingMetrics] Received epoch_end event:", {
        phase: data.phase,
        hasCmTp: "val_cm_tp" in data,
        hasCmTn: "val_cm_tn" in data,
        hasCmFp: "val_cm_fp" in data,
        hasCmFn: "val_cm_fn" in data,
        val_cm_tp: data.val_cm_tp,
        val_cm_tn: data.val_cm_tn,
        val_cm_fp: data.val_cm_fp,
        val_cm_fn: data.val_cm_fn,
      });

      // Extract confusion matrix values when phase is validation
      if (data.phase === "val" && (data.val_cm_tp !== undefined || data.val_cm_tn !== undefined)) {
        setConfusionMatrix({
          tp: data.val_cm_tp ?? null,
          tn: data.val_cm_tn ?? null,
          fp: data.val_cm_fp ?? null,
          fn: data.val_cm_fn ?? null,
          epoch: data.epoch,
        });
      }
    };

    const unsubBatch = ws.on("batch_metrics", handleBatchMetrics);
    const unsubEpoch = ws.on("epoch_end", handleEpochEnd);

    return () => {
      unsubBatch();
      unsubEpoch();
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
    setConfusionMatrix(null);
    batchBufferRef.current = [];
  }, []);

  return {
    batchMetrics,
    currentLoss,
    currentAccuracy,
    currentF1,
    isReceiving,
    confusionMatrix,
    reset,
  };
}
