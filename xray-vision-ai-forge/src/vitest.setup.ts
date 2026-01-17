import "@testing-library/jest-dom";
import { vi } from "vitest";

// Mock WebSocket
class MockWebSocket {
  url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: any) => void) | null = null;
  onerror: ((ev: any) => void) | null = null;
  readyState: number = 0; // CONNECTING

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      this.readyState = 1; // OPEN
      if (this.onopen) this.onopen();
    }, 0);
  }

  send(data: string) {
    // console.log("MockWebSocket send:", data);
  }

  close() {
    this.readyState = 3; // CLOSED
    if (this.onclose) this.onclose();
  }
}

vi.stubGlobal("WebSocket", MockWebSocket);

// Mock EventSource (for SSE)
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

  close() {
    this.readyState = MockEventSource.CLOSED;
  }
}

vi.stubGlobal("EventSource", MockEventSource);

// Mock ResizeObserver
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", MockResizeObserver);

// Mock URL.createObjectURL and URL.revokeObjectURL
// Preserve the URL constructor while adding mock methods
const OriginalURL = globalThis.URL;
class MockURL extends OriginalURL {
  static createObjectURL = vi.fn(() => "blob:mock-url");
  static revokeObjectURL = vi.fn();
}
vi.stubGlobal("URL", MockURL);

// Mock FileReader if needed (jsdom has basic support, but sometimes needs help)
if (typeof FileReader === "undefined") {
  class MockFileReader {
    onload: (() => void) | null = null;
    readAsArrayBuffer() {
      setTimeout(() => this.onload && this.onload(), 0);
    }
    readAsText() {
      setTimeout(() => this.onload && this.onload(), 0);
    }
    readAsDataURL() {
      setTimeout(() => this.onload && this.onload(), 0);
    }
  }
  vi.stubGlobal("FileReader", MockFileReader);
}
