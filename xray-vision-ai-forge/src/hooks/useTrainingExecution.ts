import { useState, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import api from '@/services/api';
import { createTrainingProgressWebSocket } from '@/services/websocket';
import { mapToBackendConfig, generateExperimentName } from '@/utils/configMapper';
import type {
  EpochStartData,
  EpochEndData,
  StatusData,
  ErrorData,
  TrainingStartData,
  TrainingEndData,
  TrainingModeData,
  RoundMetricsData,
  RoundStartData,
  RoundEndData,
  ClientTrainingStartData,
  ClientProgressData,
  ClientCompleteData,
} from '@/types/api';
import { ExperimentConfiguration } from '@/components/ExperimentConfig';

export type StatusMessage = {
  id: number;
  type: 'info' | 'progress' | 'success' | 'error';
  message: string;
  progress?: number;
  timestamp?: string;
  sortKey?: string;
  group?: string;
};

export const useTrainingExecution = (
  config: ExperimentConfiguration,
  datasetFile: File | null,
  onComplete: (runId: number) => void
) => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessages, setStatusMessages] = useState<StatusMessage[]>([]);
  const [overallStatus, setOverallStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle');
  const [experimentId, setExperimentId] = useState<string>('');
  const [runId, setRunId] = useState<number | null>(null);
  const [totalEpochs, setTotalEpochs] = useState<number>(0);

  // Federated learning state
  const [isFederatedTraining, setIsFederatedTraining] = useState(false);
  const [federatedRounds, setFederatedRounds] = useState<Array<{
    round: number;
    metrics: Record<string, number>;
  }>>([]);
  const [federatedContext, setFederatedContext] = useState({
    numRounds: 0,
    numClients: 0,
  });

  const messageIdRef = useRef(0);
  const wsRef = useRef<ReturnType<typeof createTrainingProgressWebSocket> | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const messageBufferRef = useRef<StatusMessage[]>([]);
  const bufferTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll log container
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [statusMessages]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
      if (bufferTimerRef.current) {
        clearTimeout(bufferTimerRef.current);
      }
    };
  }, []);

  const flushMessageBuffer = () => {
    if (messageBufferRef.current.length === 0) return;

    // Sort messages by: round → client_id → local_epoch → timestamp
    const sorted = [...messageBufferRef.current].sort((a, b) => {
      if (a.sortKey && b.sortKey) {
        return a.sortKey.localeCompare(b.sortKey);
      }
      return a.id - b.id;
    });

    setStatusMessages(prev => [...prev, ...sorted]);
    messageBufferRef.current = [];
  };

  const addStatusMessage = (
    type: StatusMessage['type'],
    message: string,
    progress?: number,
    metadata?: {
      timestamp?: string;
      round?: number;
      clientId?: string;
      localEpoch?: number;
      immediate?: boolean;
    }
  ) => {
    const msg: StatusMessage = {
      id: ++messageIdRef.current,
      type,
      message,
      progress,
      timestamp: metadata?.timestamp || new Date().toISOString(),
    };

    // Create sortKey for proper ordering: round_clientId_localEpoch_timestamp
    if (metadata) {
      const roundPart = metadata.round !== undefined ? metadata.round.toString().padStart(5, '0') : '00000';
      const clientPart = metadata.clientId || 'z';
      const epochPart = metadata.localEpoch !== undefined ? metadata.localEpoch.toString().padStart(3, '0') : '999';
      msg.sortKey = `${roundPart}_${clientPart}_${epochPart}_${msg.timestamp}`;
      msg.group = `round_${metadata.round}_client_${metadata.clientId}`;
    }

    // Immediate messages (like initial setup) bypass buffering
    if (metadata?.immediate) {
      setStatusMessages(prev => [...prev, msg]);
      return;
    }

    // Buffer the message
    messageBufferRef.current.push(msg);

    // Set timer to flush buffer after 500ms of no new messages
    if (bufferTimerRef.current) {
      clearTimeout(bufferTimerRef.current);
    }
    bufferTimerRef.current = setTimeout(flushMessageBuffer, 500);
  };

  const setupWebSocket = (expId: string) => {
    const ws = createTrainingProgressWebSocket(expId);
    wsRef.current = ws;

    ws.on('connected', () => {
      addStatusMessage('info', 'Connected to training progress stream', undefined, { immediate: true });
    });

    ws.on('training_mode', (data: TrainingModeData) => {
      setIsFederatedTraining(data.is_federated);
      if (data.is_federated) {
        setFederatedContext({
          numRounds: data.num_rounds,
          numClients: data.num_clients,
        });
        addStatusMessage(
          'info',
          `🔗 Federated Learning: ${data.num_clients} clients × ${data.num_rounds} rounds`,
          undefined,
          { immediate: true }
        );
      }
    });

    ws.on('round_metrics', (data: RoundMetricsData) => {
      setFederatedRounds(prev => [...prev, {
        round: data.round,
        metrics: data.metrics as Record<string, number>,
      }]);

      const acc = data.metrics.accuracy ? (data.metrics.accuracy * 100).toFixed(1) : 'N/A';
      const loss = data.metrics.loss ? data.metrics.loss.toFixed(4) : 'N/A';
      addStatusMessage(
        'success',
        `✅ Round ${data.round}/${data.total_rounds}: Accuracy = ${acc}%, Loss = ${loss}`,
        (data.round / data.total_rounds) * 100,
        { immediate: true }
      );
    });

    ws.on('training_start', (data: TrainingStartData) => {
      setRunId(data.run_id);
      setTotalEpochs(data.max_epochs);
      addStatusMessage('success', `Training started! Run ID: ${data.run_id}`, undefined, { immediate: true });
      addStatusMessage('info', `Training mode: ${data.training_mode}`, undefined, { immediate: true });
    });

    ws.on('client_training_start', (data: ClientTrainingStartData) => {
      console.log('[TrainingExecution] client_training_start handler called:', data);
      if (data.run_id) setRunId(data.run_id);

      const round = data.round || 1;
      const clientId = data.client_id?.toString() || '0';

      addStatusMessage(
        'info',
        `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🚀 Client ${clientId} - Round ${round}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
        undefined,
        {
          round,
          clientId,
          timestamp: data.timestamp,
          immediate: false
        }
      );
      addStatusMessage(
        'info',
        `   Local epochs: ${data.local_epochs} | Training samples: ${data.num_samples}`,
        undefined,
        {
          round,
          clientId,
          timestamp: data.timestamp,
          immediate: false
        }
      );
    });

    ws.on('round_start', (data: RoundStartData) => {
      console.log('[TrainingExecution] round_start handler called:', data);
      if (data.run_id) setRunId(data.run_id);

      const round = (data.round || 0);

      if (data.total_rounds > 0) {
        const roundProgress = (round / data.total_rounds) * 100;
        setProgress(roundProgress);
      }
    });

    ws.on('client_progress', (data: ClientProgressData) => {
      console.log('[TrainingExecution] client_progress handler called:', data);
      const metrics = data.metrics;
      const metricsStr = `loss: ${metrics.train_loss?.toFixed(4) || 'N/A'}, lr: ${metrics.learning_rate || 'N/A'}`;
      const progressInfo = metrics.epoch_progress ? ` ${metrics.epoch_progress}` : '';
      const percentInfo = metrics.overall_progress_percent ? ` - ${metrics.overall_progress_percent.toFixed(1)}%` : '';

      const round = data.round || 1;
      const clientId = data.client_id?.toString() || '0';
      const localEpoch = data.local_epoch || 0;

      addStatusMessage(
        'progress',
        `   ├─ Epoch ${localEpoch + 1}${progressInfo}: ${metricsStr}${percentInfo}`,
        metrics.overall_progress_percent,
        {
          round,
          clientId,
          localEpoch,
          timestamp: data.timestamp
        }
      );
    });

    ws.on('client_complete', (data: ClientCompleteData) => {
      console.log('[TrainingExecution] client_complete handler called:', data);

      const clientId = data.client_id?.toString() || '0';

      addStatusMessage(
        'success',
        `   └─ ✅ Completed - Best accuracy: ${data.best_val_accuracy?.toFixed(4) || 'N/A'}`,
        undefined,
        {
          round: data.best_round || 0,
          clientId,
          localEpoch: 999, // Put completion at end
          immediate: false
        }
      );
    });

    ws.on('round_end', (data: RoundEndData) => {
      console.log('[TrainingExecution] round_end handler called:', data);
      const fitMetrics = Object.entries(data.fit_metrics || {})
        .filter(([k, v]) => typeof v === 'number')
        .map(([k, v]) => `${k}: ${(v as number).toFixed(4)}`)
        .join(', ');

      const evalMetrics = Object.entries(data.eval_metrics || {})
        .filter(([k, v]) => typeof v === 'number')
        .map(([k, v]) => `${k}: ${(v as number).toFixed(4)}`)
        .join(', ');

      if (fitMetrics) {
        addStatusMessage('progress', `✅ Round ${data.round} [train] - ${fitMetrics}`);
      }
      if (evalMetrics) {
        addStatusMessage('progress', `✅ Round ${data.round} [val] - ${evalMetrics}`);
      }
    });

    // ws.on('federated_round_end', (data: any) => {
    //   console.log('[TrainingExecution] federated_round_end handler called:', data);
    //   const round = data.round || 0;
    //   const roundIndex = data.round_index !== undefined ? data.round_index : round - 1;
    //   const totalRounds = data.total_rounds || 0;

    //   // Update progress based on federated round completion
    //   if (totalRounds > 0) {
    //     const roundProgress = (round / totalRounds) * 100;
    //     setProgress(roundProgress);
    //   }

    //   addStatusMessage(
    //     'success',
    //     `\n✅ Federated Round ${round}/${totalRounds} completed (Round Index: ${roundIndex})`,
    //     undefined,
    //     {
    //       round,
    //       timestamp: data.timestamp,
    //       immediate: false
    //     }
    //   );
    // });

    ws.on('epoch_start', (data: EpochStartData) => {
      addStatusMessage('info', `Starting epoch ${data.epoch}/${data.total_epochs}...`);
      const epochProgress = ((data.epoch - 1) / data.total_epochs) * 100;
      setProgress(Math.min(epochProgress, 95));
    });

    ws.on('epoch_end', (data: EpochEndData) => {
      const metrics = data.metrics;
      const metricsStr = Object.entries(metrics)
        .filter(([_, v]) => typeof v === 'number')
        .map(([k, v]) => `${k}: ${(v as number).toFixed(4)}`)
        .join(', ');

      addStatusMessage(
        'progress',
        `Epoch ${data.epoch} [${data.phase}] - ${metricsStr}`,
        metrics.accuracy ? metrics.accuracy * 100 : undefined
      );

      if (totalEpochs > 0) {
        const epochProgress = (data.epoch / totalEpochs) * 100;
        setProgress(Math.min(epochProgress, 99));
      }
    });

    ws.on('status', (data: StatusData) => {
      if (data.message) {
        addStatusMessage('info', data.message);
      }

      if (data.status === 'completed') {
        setProgress(100);
        setIsRunning(false);
        setOverallStatus('completed');
        toast.success('Training completed successfully!');
      } else if (data.status === 'failed') {
        setIsRunning(false);
        setOverallStatus('error');
        toast.error('Training failed!');
      } else if (data.status === 'running') {
        setOverallStatus('running');
      }
    });

    ws.on('training_end', (data: TrainingEndData) => {
      // Flush any remaining buffered messages before showing completion
      flushMessageBuffer();

      if (data.status === 'completed') {
        addStatusMessage('success', `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, undefined, { immediate: true });
        addStatusMessage('success', `✅ Training completed! Run ID: ${data.run_id}`, undefined, { immediate: true });
        addStatusMessage('info', `Best epoch: ${data.best_epoch}, Best recall: ${data.best_val_recall?.toFixed(4) || 'N/A'}`, undefined, { immediate: true });
        addStatusMessage('info', `Total epochs: ${data.total_epochs}, Duration: ${data.training_duration || 'N/A'}`, undefined, { immediate: true });
        addStatusMessage('success', `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`, undefined, { immediate: true });
        setProgress(100);
        setIsRunning(false);
        setOverallStatus('completed');
        toast.success('Training completed successfully!');

        // Trigger navigation to results view
        if (data.run_id) {
          onComplete(data.run_id);
        }
      } else {
        addStatusMessage('error', `Training failed for run ${data.run_id}`, undefined, { immediate: true });
        setIsRunning(false);
        setOverallStatus('error');
        toast.error('Training failed!');
      }

      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    });

    // ws.on('early_stopping', (data: TrainingEndData) => {
    //   console.log('[TrainingExecution] early_stopping handler called:', data);
    //   addStatusMessage('info', `Early stopping triggered at epoch ${data.best_epoch}`, undefined, { immediate: true });
    //   addStatusMessage('info', `Best metric value: ${data.best_val_recall?.toFixed(4) || 'N/A'}`, undefined, { immediate: true });
    //   addStatusMessage('info', `Metric name: val_recall`, undefined, { immediate: true });
    //   addStatusMessage('info', `Reason: ${data.reason}`, undefined, { immediate: true });
    // });

    ws.on('error', (data: ErrorData) => {
      console.error('[TrainingExecution] WebSocket error:', data);
      flushMessageBuffer(); // Flush before showing error
      setIsRunning(false);
      setOverallStatus('error');
      addStatusMessage('error', `Error: ${data.error}`, undefined, { immediate: true });
      toast.error(`Training error: ${data.error}`);

      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    });

    ws.on('disconnected', () => {
      flushMessageBuffer(); // Flush on disconnect
      addStatusMessage('info', 'Disconnected from training progress stream', undefined, { immediate: true });
      if (overallStatus === 'running') {
        setIsRunning(false);
        setOverallStatus('error');
        addStatusMessage('error', 'Connection lost during training', undefined, { immediate: true });
      }
    });

    ws.connect();
  };

  const startTraining = async () => {
    if (!datasetFile) {
      toast.error('Please upload a dataset file first');
      return;
    }

    try {
      setIsRunning(true);
      setOverallStatus('running');
      setProgress(0);
      setStatusMessages([]);
      setRunId(null);
      messageIdRef.current = 0;

      // Reset federated state
      setIsFederatedTraining(false);
      setFederatedRounds([]);
      setFederatedContext({ numRounds: 0, numClients: 0 });

      addStatusMessage('info', 'Preparing to start training...', undefined, { immediate: true });

      const expName = generateExperimentName(config.trainingMode);
      setExperimentId(expName);
      addStatusMessage('info', `Experiment: ${expName}`, undefined, { immediate: true });

      addStatusMessage('info', 'Uploading dataset and starting training...', undefined, { immediate: true });

      const backendConfig = mapToBackendConfig(config);

      await api.configuration.setConfiguration(backendConfig);
      addStatusMessage('info', 'Configuration set successfully', undefined, { immediate: true });

      setupWebSocket(expName);

      let response;
      if (config.trainingMode === 'centralized') {
        response = await api.experiments.startCentralizedTraining(datasetFile, expName);
      } else if (config.trainingMode === 'federated') {
        response = await api.experiments.startFederatedTraining(datasetFile, expName);
      } else {
        addStatusMessage('info', 'Running both centralized and federated training...', undefined, { immediate: true });
        await api.experiments.startCentralizedTraining(datasetFile, `${expName}_centralized`);
        response = await api.experiments.startFederatedTraining(datasetFile, `${expName}_federated`);
      }

      addStatusMessage('success', response.message || 'Training started successfully', undefined, { immediate: true });
      addStatusMessage('info', 'Waiting for progress updates...', undefined, { immediate: true });
    } catch (error: any) {
      console.error('Failed to start training:', error);
      flushMessageBuffer(); // Flush any pending messages before showing error
      setIsRunning(false);
      setOverallStatus('error');
      addStatusMessage('error', `Failed to start training: ${error.message || 'Unknown error'}`, undefined, { immediate: true });
      toast.error('Failed to start training. Please try again.');
    }
  };

  const handleCompleteStep = () => {
    if (runId === null) {
      toast.error('No run ID available. Cannot proceed to results.');
      return;
    }
    onComplete(runId);
  };

  return {
    isRunning,
    progress,
    statusMessages,
    overallStatus,
    experimentId,
    runId,
    logContainerRef,
    startTraining,
    handleCompleteStep,
    // Federated learning state
    isFederatedTraining,
    federatedRounds,
    federatedContext,
  };
};
