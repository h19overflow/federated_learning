import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TrainingProgressSSE, createTrainingProgressSSE } from '../sse';

// Mock EventSource
class MockEventSource {
  url: string;
  readyState: number = 0; // CONNECTING
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  listeners: Map<string, ((event: MessageEvent) => void)[]> = new Map();

  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void): void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type)!.push(listener);
  }

  close(): void {
    this.readyState = MockEventSource.CLOSED;
  }

  // Simulate opening the connection
  simulateOpen(): void {
    this.readyState = MockEventSource.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }

  // Simulate receiving a message
  simulateMessage(eventType: string, data: any): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      const event = new MessageEvent('message', {
        data: JSON.stringify(data),
      });
      listeners.forEach((listener) => listener(event));
    }
  }

  // Simulate an error
  simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
}

// Replace global EventSource with mock
const originalEventSource = global.EventSource;

describe('TrainingProgressSSE', () => {
  let mockEventSource: MockEventSource;

  beforeEach(() => {
    // Mock EventSource globally
    global.EventSource = MockEventSource as any;
  });

  afterEach(() => {
    // Restore original EventSource
    global.EventSource = originalEventSource;
    vi.clearAllMocks();
  });

  it('should create SSE connection with correct URL', () => {
    const experimentId = 'test-exp-123';
    const sse = new TrainingProgressSSE(experimentId);
    sse.connect();

    // Access the private eventSource through type assertion for testing
    const eventSource = (sse as any).eventSource as MockEventSource;
    expect(eventSource.url).toBe(`http://localhost:8001/api/training/stream/${experimentId}`);
  });

  it('should register event listeners for all event types', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const eventSource = (sse as any).eventSource as MockEventSource;
    const eventTypes = [
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

    eventTypes.forEach((type) => {
      expect(eventSource.listeners.has(type)).toBe(true);
    });
  });

  it('should emit events to registered listeners', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const mockListener = vi.fn();
    sse.on('training_start', mockListener);

    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();

    const testData = {
      run_id: 42,
      max_epochs: 10,
      training_mode: 'centralized',
    };

    eventSource.simulateMessage('training_start', testData);

    expect(mockListener).toHaveBeenCalledWith(testData);
  });

  it('should support multiple listeners for the same event', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const listener1 = vi.fn();
    const listener2 = vi.fn();

    sse.on('epoch_end', listener1);
    sse.on('epoch_end', listener2);

    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();

    const testData = {
      epoch: 5,
      total_epochs: 10,
      phase: 'train',
      metrics: { accuracy: 0.95 },
    };

    eventSource.simulateMessage('epoch_end', testData);

    expect(listener1).toHaveBeenCalledWith(testData);
    expect(listener2).toHaveBeenCalledWith(testData);
  });

  it('should unsubscribe listeners using returned function', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const mockListener = vi.fn();
    const unsubscribe = sse.on('status', mockListener);

    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();

    // First message should be received
    eventSource.simulateMessage('status', { status: 'running', message: 'Test 1' });
    expect(mockListener).toHaveBeenCalledTimes(1);

    // Unsubscribe
    unsubscribe();

    // Second message should not be received
    eventSource.simulateMessage('status', { status: 'completed', message: 'Test 2' });
    expect(mockListener).toHaveBeenCalledTimes(1); // Still 1, not called again
  });

  it('should handle disconnection properly', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();

    expect(sse.isConnectedState()).toBe(true);

    sse.disconnect();

    expect(eventSource.readyState).toBe(MockEventSource.CLOSED);
    expect(sse.isConnectedState()).toBe(false);
  });

  it('should remove all listeners for a specific event', () => {
    const sse = new TrainingProgressSSE('test-exp');

    const listener1 = vi.fn();
    const listener2 = vi.fn();

    sse.on('error', listener1);
    sse.on('error', listener2);
    sse.on('status', vi.fn());

    sse.removeAllListeners('error');

    sse.connect();
    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();
    eventSource.simulateMessage('error', { error: 'Test error' });

    expect(listener1).not.toHaveBeenCalled();
    expect(listener2).not.toHaveBeenCalled();
  });

  it('should create SSE instance using factory function', () => {
    const experimentId = 'factory-test';
    const sse = createTrainingProgressSSE(experimentId);

    expect(sse).toBeInstanceOf(TrainingProgressSSE);
    sse.connect();

    const eventSource = (sse as any).eventSource as MockEventSource;
    expect(eventSource.url).toContain(experimentId);
  });

  it('should handle JSON parse errors gracefully', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const mockListener = vi.fn();
    sse.on('training_start', mockListener);

    const eventSource = (sse as any).eventSource as MockEventSource;
    eventSource.simulateOpen();

    // Simulate message with invalid JSON
    const listeners = eventSource.listeners.get('training_start');
    if (listeners) {
      const event = new MessageEvent('message', {
        data: 'invalid json {{{',
      });
      listeners.forEach((listener) => listener(event));
    }

    expect(consoleErrorSpy).toHaveBeenCalled();
    expect(mockListener).not.toHaveBeenCalled();

    consoleErrorSpy.mockRestore();
  });

  it('should not connect if already connected', () => {
    const sse = new TrainingProgressSSE('test-exp');
    sse.connect();

    const eventSource1 = (sse as any).eventSource;
    eventSource1.simulateOpen();

    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    // Try to connect again
    sse.connect();

    expect(consoleWarnSpy).toHaveBeenCalledWith('[SSE] Already connected');
    expect((sse as any).eventSource).toBe(eventSource1); // Should be same instance

    consoleWarnSpy.mockRestore();
  });
});
