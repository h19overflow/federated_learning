/**
 * WebSocket Service for real-time training progress updates.
 *
 * Manages WebSocket connections to the backend for live training progress streaming.
 * Provides auto-reconnect, event-based message handling, and type-safe listeners.
 *
 * Dependencies:
 * - Environment variables for WebSocket URL
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
} from "@/types/api";
import { getEnv } from "@/utils/env";

// ============================================================================
// Configuration
// ============================================================================

const env = getEnv();
const WS_BASE_URL = env.VITE_WS_BASE_URL;
const RECONNECT_DELAY = 3000; // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 10;
const PING_INTERVAL = 30000; // 30 seconds

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
  pong: EventListener<Record<string, never>>[];
  disconnected: EventListener<Record<string, never>>[];
  reconnecting: EventListener<{ attempt: number }>[];
  reconnected: EventListener<Record<string, never>>[];
  // Training observability events
  batch_metrics: EventListener<BatchMetricsData>[];
}

// ============================================================================
// WebSocket Manager Class
// ============================================================================

export class TrainingProgressWebSocket {
  private ws: WebSocket | null = null;
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
    pong: [],
    disconnected: [],
    reconnecting: [],
    reconnected: [],
    // Training observability events
    batch_metrics: [],
  };
  private reconnectAttempts = 0;
  private reconnectTimeout: number | null = null;
  private pingInterval: number | null = null;
  private shouldReconnect = true;
  private isConnected = false;

  constructor(experimentId: string) {
    this.experimentId = experimentId;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws && this.isConnected) {
      console.warn("WebSocket already connected");
      return;
    }

    // Connect to simple WebSocket server (no experiment path needed)
    // Backend sends all metrics to ws://localhost:8765
    const url = WS_BASE_URL;
    console.log(
      `🔌 [WebSocket] Attempting connection to: ${url} for experiment: ${this.experimentId}`,
    );

    try {
      this.ws = new WebSocket(url);
      console.log(
        `🔌 [WebSocket] WebSocket object created, waiting for onopen...`,
      );

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      console.error(
        "❌ [WebSocket] Failed to create WebSocket connection:",
        error,
      );
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;
    this.stopPing();

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnected = false;
  }

  /**
   * Check if WebSocket is connected
   */
  isConnectedState(): boolean {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    console.log(
      `✅ [WebSocket] Connected! Ready for metrics. Experiment: ${this.experimentId}`,
    );
    this.isConnected = true;
    this.reconnectAttempts = 0;
    this.startPing();

    // Emit connected event
    this.emit("connected", { experiment_id: this.experimentId });

    if (this.reconnectAttempts > 0) {
      this.emit("reconnected", {});
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      console.log(
        `📨 [WebSocket] Message received [${message.type}]:`,
        message.data,
      );

      // Emit to appropriate listeners
      this.emit(message.type, message.data);
    } catch (error) {
      console.error(
        "❌ [WebSocket] Failed to parse message:",
        error,
        "Raw data:",
        event.data,
      );
    }
  }

  /**
   * Handle WebSocket error
   */
  private handleError(event: Event): void {
    console.error("❌ [WebSocket] Error:", event);
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    console.log(
      `⚠️  [WebSocket] Connection closed: ${event.code} - ${event.reason}`,
    );
    this.isConnected = false;
    this.stopPing();
    this.emit("disconnected", {});

    if (this.shouldReconnect) {
      this.scheduleReconnect();
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error("Max reconnection attempts reached. Giving up.");
      return;
    }

    this.reconnectAttempts++;
    console.log(
      `Scheduling reconnect attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${RECONNECT_DELAY}ms`,
    );

    this.emit("reconnecting", { attempt: this.reconnectAttempts });

    this.reconnectTimeout = window.setTimeout(() => {
      console.log("Attempting to reconnect...");
      this.connect();
    }, RECONNECT_DELAY);
  }

  /**
   * Start periodic ping to keep connection alive
   */
  private startPing(): void {
    this.stopPing();

    this.pingInterval = window.setInterval(() => {
      if (this.isConnectedState()) {
        this.send({ type: "ping" });
      }
    }, PING_INTERVAL);
  }

  /**
   * Stop periodic ping
   */
  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Send message to server
   */
  private send(message: { type: string }): void {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn("Cannot send message: WebSocket not connected");
    }
  }

  /**
   * Emit event to listeners
   */
  private emit<K extends keyof EventListeners>(event: K, data: unknown): void {
    const eventListeners = this.listeners[event];
    console.log(
      `[WebSocket] Emitting event '${event}' to ${eventListeners?.length || 0} listeners`,
    );
    if (eventListeners && eventListeners.length > 0) {
      eventListeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    } else {
      console.warn(`[WebSocket] No listeners registered for event '${event}'`);
    }
  }

  /**
   * Add event listener
   */
  on<K extends keyof EventListeners>(
    event: K,
    listener: EventListeners[K][number],
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
    listener: EventListeners[K][number],
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
      this.listeners[event] = [];
    } else {
      // Reset all listeners
      Object.keys(this.listeners).forEach((key) => {
        this.listeners[key as keyof EventListeners] = [];
      });
    }
  }
}

// ============================================================================
// Convenience Hooks for React Components
// ============================================================================

/**
 * Create a WebSocket connection for an experiment
 */
export function createTrainingProgressWebSocket(
  experimentId: string,
): TrainingProgressWebSocket {
  return new TrainingProgressWebSocket(experimentId);
}

/**
 * Get WebSocket base URL
 */
export function getWebSocketBaseUrl(): string {
  return WS_BASE_URL;
}

export default TrainingProgressWebSocket;
