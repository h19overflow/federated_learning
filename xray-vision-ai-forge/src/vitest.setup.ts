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

// Mock ResizeObserver
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal("ResizeObserver", MockResizeObserver);

// Mock URL.createObjectURL and URL.revokeObjectURL
vi.stubGlobal("URL", {
  ...URL,
  createObjectURL: vi.fn(() => "blob:mock-url"),
  revokeObjectURL: vi.fn(),
});

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
