import { RunSummary } from "@/types/runs";

export interface Message {
  role: "user" | "assistant";
  content: string;
  runContext?: RunContext;
}

export interface RunContext {
  runId: number;
  trainingMode: string;
  status: string;
  startTime: string;
  bestRecall?: number;
}

export interface ChatSession {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
}

export interface ChatSidebarProps {
  apiUrl?: string;
}

export { RunSummary };
