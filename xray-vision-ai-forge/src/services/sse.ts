/**
 * SSE Service for real-time training progress updates.
 *
 * Manages Server-Sent Events connections to the backend for live training streaming.
 * Provides event-based message handling and type-safe listeners.
 *
 * Dependencies:
 * - Environment variables for API URL
 * - Type definitions from @/types/api
 */

import {
  WebSocketMessage,
  WebSocketMessageType,
  EpochStartData,
  EpochEndData,
  RoundStartData,
  RoundEndData,
  LocalEpochData,
  StatusData,
  ErrorData,
  TrainingStartData,
  TrainingEndData,
  TrainingModeData,
  RoundMetricsData,
  ClientTrainingStartData,
  ClientProgressData,
  ClientCompleteData,
  EarlyStoppingData,
  BatchMetricsData,
  GradientStatsData,
  LRUpdateData,
} from '@/types/api';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = 'http://localhost:8001';
const MAX_RECONNECT_ATTEMPTS = 10;

// ============================================================================
// Event Listener Types
// ============================================================================

type EventListener<T = unknown> = (data: T) => void;

interface EventListeners {
  connected: EventListener<{ experiment_id: string }>[];
  training_start: EventListener<TrainingStartData>[];
  training_end: EventListener<TrainingEndData>[];
  training_mode: EventListener<TrainingModeData>[];
  round_metrics: EventListener<RoundMetricsData>[];
  epoch_start: EventListener<EpochStartData>[];
  epoch_end: EventListener<EpochEndData>[];
  round_start: EventListener<RoundStartData>[];
  round_end: EventListener<RoundEndData>[];
  local_epoch: EventListener<LocalEpochData>[];
  client_training_start: EventListener<ClientTrainingStartData>[];
  client_progress: EventListener<ClientProgressData>[];
  client_complete: EventListener<ClientCompleteData>[];
  status: EventListener<StatusData>[];
  error: EventListener<ErrorData>[];
  early_stopping: EventListener<EarlyStoppingData>[];
  disconnected: EventListener<Record<string, never>>[];
  reconnecting: EventListener<{ attempt: number }>[];
  reconnected: EventListener<Record<string, never>>[];
  // Training observability events
  batch_metrics: EventListener<BatchMetricsData>[];
  gradient_stats: EventListener<GradientStatsData>[];
  lr_update: EventListener<LRUpdateData>[];
}

// ============================================================================
// SSE Manager Class
// ============================================================================

export class TrainingProgressSSE {
  private eventSource: EventSource | null = null;
  private experimentId: string;
  private listeners: EventListeners = {
    connected: [],
    training_start: [],
    training_end: [],
    training_mode: [],
    round_metrics: [],
    epoch_start: [],
    epoch_end: [],
    round_start: [],
    round_end: [],
    local_epoch: [],
    client_training_start: [],
    client_progress: [],
    client_complete: [],
    status: [],
    error: [],
    early_stopping: [],
    disconnected: [],
    reconnecting: [],
    reconnected: [],
    batch_metrics: [],
    gradient_stats: [],
    lr_update: [],
  };
  private reconnectAttempts = 0;
  private shouldReconnect = true;
  private isConnected = false;

  constructor(experimentId: string) {
    this.experimentId = experimentId;
  }

  /**
   * Connect to SSE endpoint
   */
  connect(): void {
    if (this.eventSource && this.isConnected) {
      console.warn('[SSE] Already connected');
      return;
    }

    const url = `${API_BASE_URL}/api/training/stream/${this.experimentId}`;
    console.log(`🔌 [SSE] Connecting to: ${url}`);

    try {
      this.eventSource = new EventSource(url);

      // Register all event types
      const messageTypes: (keyof EventListeners)[] = [
        'connected',
        'training_start',
        'training_end',
        'training_mode',
        'round_metrics',
        'epoch_start',
        'epoch_end',
        'round_start',
        'round_end',
        'local_epoch',
        'client_training_start',
        'client_progress',
        'client_complete',
        'status',
        'error',
        'early_stopping',
        'batch_metrics',
        'gradient_stats',
        'lr_update',
      ];

      messageTypes.forEach((type) => {
        this.eventSource!.addEventListener(type, (event: MessageEvent) => {
          this.handleMessage(type, event.data);
        });
      });

      this.eventSource.onopen = this.handleOpen.bind(this);
      this.eventSource.onerror = this.handleError.bind(this);
    } catch (error) {
      console.error('❌ [SSE] Failed to create EventSource:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from SSE endpoint
   */
  disconnect(): void {
    this.shouldReconnect = false;

    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }

    this.isConnected = false;
  }

  /**
   * Check if SSE is connected
   */
  isConnectedState(): boolean {
    return this.isConnected && this.eventSource?.readyState === EventSource.OPEN;
  }

  /**
   * Handle SSE connection open
   */
  private handleOpen(): void {
    console.log(`✅ [SSE] Connected to experiment: ${this.experimentId}`);
    this.isConnected = true;
    this.reconnectAttempts = 0;

    if (this.reconnectAttempts > 0) {
      this.emit('reconnected', {});
    }
  }

  /**
   * Handle incoming SSE messages
   */
  private handleMessage(eventType: string, data: string): void {
    try {
      const parsedData = JSON.parse(data);
      console.log(`📨 [SSE] Message received [${eventType}]:`, parsedData);

      // Emit to appropriate listeners
      this.emit(eventType as keyof EventListeners, parsedData);
    } catch (error) {
      console.error('❌ [SSE] Failed to parse message:', error, 'Raw data:', data);
    }
  }

  /**
   * Handle SSE errors (connection lost, etc.)
   */
  private handleError(event: Event): void {
    console.error('❌ [SSE] Error:', event);

    // EventSource auto-reconnects for network errors, but we track attempts
    if (this.eventSource?.readyState === EventSource.CLOSED) {
      console.log('⚠️  [SSE] Connection closed');
      this.isConnected = false;
      this.emit('disconnected', {});

      if (this.shouldReconnect) {
        this.scheduleReconnect();
      }
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error('[SSE] Max reconnection attempts reached. Giving up.');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000); // Exponential backoff, max 30s

    console.log(
      `[SSE] Scheduling reconnect attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms`
    );

    this.emit('reconnecting', { attempt: this.reconnectAttempts });

    setTimeout(() => {
      console.log('[SSE] Attempting to reconnect...');
      this.connect();
    }, delay);
  }

  /**
   * Emit event to listeners
   */
  private emit<K extends keyof EventListeners>(event: K, data: unknown): void {
    const eventListeners = this.listeners[event];
    console.log(`[SSE] Emitting event '${event}' to ${eventListeners?.length || 0} listeners`);

    if (eventListeners && eventListeners.length > 0) {
      eventListeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    } else {
      console.warn(`[SSE] No listeners registered for event '${event}'`);
    }
  }

  /**
   * Add event listener
   */
  on<K extends keyof EventListeners>(
    event: K,
    listener: EventListeners[K][number]
  ): () => void {
    this.listeners[event].push(listener);

    // Return unsubscribe function
    return () => {
      this.off(event, listener);
    };
  }

  /**
   * Remove event listener
   */
  off<K extends keyof EventListeners>(
    event: K,
    listener: EventListeners[K][number]
  ): void {
    const index = this.listeners[event].indexOf(listener);
    if (index > -1) {
      this.listeners[event].splice(index, 1);
    }
  }

  /**
   * Remove all listeners for an event or all events
   */
  removeAllListeners(event?: keyof EventListeners): void {
    if (event) {
      this.listeners[event] = [] as EventListeners[typeof event];
    } else {
      // Reset all listeners
      Object.keys(this.listeners).forEach((key) => {
        const eventKey = key as keyof EventListeners;
        this.listeners[eventKey] = [] as EventListeners[typeof eventKey];
      });
    }
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Create an SSE connection for an experiment
 */
export function createTrainingProgressSSE(experimentId: string): TrainingProgressSSE {
  return new TrainingProgressSSE(experimentId);
}

/**
 * Get SSE base URL
 */
export function getSSEBaseUrl(): string {
  return API_BASE_URL;
}

export default TrainingProgressSSE;
