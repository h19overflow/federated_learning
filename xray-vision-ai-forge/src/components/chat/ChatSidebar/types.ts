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

export interface RunSummary {
  id: number;
  training_mode: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  best_val_recall: number;
  metrics_count: number;
  run_description: string | null;
  federated_info: any;
}

export interface ChatSidebarProps {
  apiUrl?: string;
}
