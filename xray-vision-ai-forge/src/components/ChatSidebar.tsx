import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Markdown } from "@/components/ui/markdown";
import { Textarea } from "@/components/ui/textarea";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  Trash2,
  Loader2,
  X,
  BarChart,
  Activity,
  GripVertical,
  BookOpen,
  Plus,
  History,
  Database,
} from "lucide-react";

import { cn } from "@/lib/utils";
import api from "@/services/api";

const MIN_WIDTH = 320;
const MAX_WIDTH = 800;
const DEFAULT_WIDTH = 384; // w-96

interface Message {
  role: "user" | "assistant";
  content: string;
  runContext?: RunContext;
}

interface RunContext {
  runId: number;
  trainingMode: string;
  status: string;
  startTime: string;
}

interface ChatSession {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
}

interface RunSummary {
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

interface ChatSidebarProps {
  apiUrl?: string;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  apiUrl = "http://127.0.0.1:8001",
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [isOpen, setIsOpen] = useState(true);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const runPickerRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);

  // Slash command state
  const [showRunPicker, setShowRunPicker] = useState(false);
  const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunContext | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [highlightedRunIndex, setHighlightedRunIndex] = useState(0);

  // Arxiv toggle state
  const [arxivEnabled, setArxivEnabled] = useState(false);
  const [arxivAvailable, setArxivAvailable] = useState(false);

  // Agent status for observability
  const [agentStatus, setAgentStatus] = useState<string | null>(null);

  // Generate session ID or fetch existing on mount
  useEffect(() => {
    const checkSession = async () => {
      const storedSessionId = localStorage.getItem("chat_session_id");
      // Basic UUID v4 regex check
      const isUUID =
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

      if (storedSessionId && isUUID.test(storedSessionId)) {
        setSessionId(storedSessionId);
        loadHistory(storedSessionId);
      } else {
        // We will create a session on the first message or when the user clicks "New Chat"
        // For now, just generate a temporary ID if none exists
        const newSessionId = generateSessionId();
        setSessionId(newSessionId);
        localStorage.setItem("chat_session_id", newSessionId);
      }
      fetchSessions();
    };
    checkSession();
  }, []);

  const fetchSessions = async () => {
    try {
      setIsLoadingSessions(true);
      const response = await fetch(`${apiUrl}/chat/sessions`);
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const createNewSession = async (initialQuery?: string) => {
    try {
      setIsLoading(true);
      const requestBody: { title?: string; initial_query?: string } = {};

      // If we have an initial query, use it to generate title
      if (initialQuery) {
        requestBody.initial_query = initialQuery;
      } else {
        requestBody.title = "New Chat";
      }

      const response = await fetch(`${apiUrl}/chat/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (response.ok) {
        const newSession = await response.json();
        setSessionId(newSession.id);
        localStorage.setItem("chat_session_id", newSession.id);
        // Only clear messages if this is a manual "New Chat" (no initial query)
        if (!initialQuery) {
          setMessages([]);
        }
        setShowHistory(false);
        fetchSessions();
        return newSession.id;
      }
    } catch (error) {
      console.error("Error creating new session:", error);
    } finally {
      setIsLoading(false);
    }
    return null;
  };

  const handleSwitchSession = (sid: string) => {
    setSessionId(sid);
    localStorage.setItem("chat_session_id", sid);
    loadHistory(sid);
    setShowHistory(false);
  };

  const handleDeleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await fetch(`${apiUrl}/chat/sessions/${sid}`, {
        method: "DELETE",
      });
      if (response.ok) {
        if (sid === sessionId) {
          // If we deleted the current session, create a new one
          createNewSession();
        } else {
          fetchSessions();
        }
      }
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  // Check arxiv availability on mount
  useEffect(() => {
    const checkArxivStatus = async () => {
      try {
        const res = await fetch(`${apiUrl}/chat/arxiv/status`);
        if (res.ok) {
          const data = await res.json();
          setArxivAvailable(data.available);
        } else {
          setArxivAvailable(false);
        }
      } catch {
        setArxivAvailable(false);
      }
    };
    checkArxivStatus();
  }, [apiUrl]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [messages]);

  // Resize handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      const newWidth = window.innerWidth - e.clientX;
      if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) {
        setWidth(newWidth);
      } else if (newWidth < MIN_WIDTH / 2) {
        // If dragged too far right, close the sidebar
        setIsOpen(false);
        setIsResizing(false);
      }
    },
    [isResizing]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  // Attach global mouse events for resize
  useEffect(() => {
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "ew-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  const generateSessionId = (): string => {
    return crypto.randomUUID();
  };

  const fetchAvailableRuns = async () => {
    try {
      setLoadingRuns(true);
      const response = await api.results.listRuns();
      setAvailableRuns(response.runs);
    } catch (error) {
      console.error("Error fetching runs:", error);
    } finally {
      setLoadingRuns(false);
    }
  };

  const loadHistory = async (sid: string) => {
    try {
      const response = await fetch(`${apiUrl}/chat/history/${sid}`);
      if (response.ok) {
        const data = await response.json();
        const loadedMessages: Message[] = [];
        data.history.forEach((item: { user: string; assistant: string }) => {
          loadedMessages.push({ role: "user", content: item.user });
          loadedMessages.push({ role: "assistant", content: item.assistant });
        });
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const handleSelectRun = (run: RunSummary) => {
    const runContext: RunContext = {
      runId: run.id,
      trainingMode: run.training_mode,
      status: run.status,
      startTime: run.start_time || "Unknown",
    };
    setSelectedRun(runContext);
    setShowRunPicker(false);

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: `Now discussing Run #${run.id} (${run.training_mode} training). You can ask me about its metrics, performance, or training details.`,
        runContext,
      },
    ]);

    setInput("");
    inputRef.current?.focus();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInput(value);

    // Auto-resize textarea
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(
        inputRef.current.scrollHeight,
        200
      )}px`;
    }

    if (value === "/") {
      setShowRunPicker(true);
      setHighlightedRunIndex(0);
      if (availableRuns.length === 0) {
        fetchAvailableRuns();
      }
    } else if (value.length === 0 || !value.startsWith("/")) {
      setShowRunPicker(false);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    setShowRunPicker(false);

    const newMessage: Message = {
      role: "user",
      content: userMessage,
      runContext: selectedRun || undefined,
    };

    // Add user message and empty assistant placeholder
    setMessages((prev) => [
      ...prev,
      newMessage,
      { role: "assistant", content: "", runContext: selectedRun || undefined },
    ]);
    setIsLoading(true);
    setAgentStatus(null); // Reset agent status

    try {
      // If this is the first message (empty history), create a session with title generation
      let currentSessionId = sessionId;
      if (messages.length === 0) {
        const newSessionId = await createNewSession(userMessage);
        if (newSessionId) {
          currentSessionId = newSessionId;
        }
      }

      const requestBody: any = {
        query: userMessage,
        session_id: currentSessionId,
        arxiv_enabled: arxivEnabled,
      };

      if (selectedRun) {
        requestBody.run_id = selectedRun.runId;
        requestBody.training_mode = selectedRun.trainingMode;
      }

      // Use streaming endpoint
      const response = await fetch(`${apiUrl}/chat/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        // Keep incomplete line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "token") {
                // Append token to the last message
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastMsg = updated[updated.length - 1];
                  if (lastMsg.role === "assistant") {
                    lastMsg.content += data.content;
                  }
                  return updated;
                });
              } else if (data.type === "session") {
                // Update session ID if different
                if (data.session_id && data.session_id !== sessionId) {
                  setSessionId(data.session_id);
                  localStorage.setItem("chat_session_id", data.session_id);
                }
              } else if (data.type === "error") {
                console.error("Stream error:", data.message);
                setAgentStatus(null);
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastMsg = updated[updated.length - 1];
                  if (lastMsg.role === "assistant") {
                    lastMsg.content =
                      "Sorry, I encountered an error while processing your query. Please try again.";
                  }
                  return updated;
                });
              } else if (data.type === "status") {
                // Update the agent status indicator
                setAgentStatus(data.content);
              } else if (data.type === "tool_call") {
                // Update status with the tool being used
                const toolName = data.tool?.replace(/_/g, " ") || "a tool";
                setAgentStatus(`Using ${toolName}...`);
              } else if (data.type === "done") {
                // Clear agent status when done
                setAgentStatus(null);
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg.role === "assistant") {
          lastMsg.content =
            "Sorry, I encountered an error while processing your query. Please try again.";
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
      setAgentStatus(null);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  };

  const handleClearChat = async () => {
    // If we have messages, we just start a new session
    if (messages.length > 0) {
      createNewSession();
    }
  };

  const formatRunTime = (timeStr: string | null) => {
    if (!timeStr) return "Unknown";
    try {
      const date = new Date(timeStr);
      return date.toLocaleString();
    } catch {
      return timeStr;
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !showRunPicker) {
      e.preventDefault();
      handleSendMessage();
      return;
    }

    if (showRunPicker && availableRuns.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlightedRunIndex((prev) => {
          const newIndex = prev < availableRuns.length - 1 ? prev + 1 : prev;
          setTimeout(() => {
            const runItem = runPickerRef.current?.querySelector(
              `[data-run-index="${newIndex}"]`
            );
            runItem?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 0);
          return newIndex;
        });
        return;
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlightedRunIndex((prev) => {
          const newIndex = prev > 0 ? prev - 1 : prev;
          setTimeout(() => {
            const runItem = runPickerRef.current?.querySelector(
              `[data-run-index="${newIndex}"]`
            );
            runItem?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 0);
          return newIndex;
        });
        return;
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleSelectRun(availableRuns[highlightedRunIndex]);
        return;
      } else if (e.key === "Escape") {
        e.preventDefault();
        setShowRunPicker(false);
        setInput("");
        return;
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Floating Action Button - Apple Style */}
      {!isOpen && (
        <Button
          onClick={() => {
            setWidth(DEFAULT_WIDTH);
            setIsOpen(true);
          }}
          className="fixed bottom-6 right-6 rounded-2xl h-14 w-14 shadow-xl shadow-[hsl(172_63%_22%)]/25 hover:shadow-2xl hover:shadow-[hsl(172_63%_22%)]/35 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white transition-all duration-300 hover:scale-105"
          size="icon"
        >
          <MessageSquare className="h-6 w-6" />
        </Button>
      )}

      {/* Sidebar Container */}
      <div
        ref={sidebarRef}
        className={cn(
          "flex flex-col border-l border-[hsl(210_15%_92%)] bg-white overflow-hidden relative",
          "h-full",
          isOpen ? "opacity-100" : "w-0 opacity-0 pointer-events-none",
          !isResizing && "transition-all duration-300 ease-out"
        )}
        style={{ width: isOpen ? width : 0 }}
      >
        {/* Resize Handle */}
        <div
          onMouseDown={handleMouseDown}
          className={cn(
            "absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize z-50 group",
            "hover:bg-[hsl(172_63%_35%)] transition-colors",
            isResizing && "bg-[hsl(172_63%_35%)]"
          )}
        >
          <div
            className={cn(
              "absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-8 rounded-md flex items-center justify-center",
              "opacity-0 group-hover:opacity-100 transition-opacity",
              "bg-[hsl(172_63%_22%)] shadow-md",
              isResizing && "opacity-100"
            )}
          >
            <GripVertical className="h-4 w-4 text-white" />
          </div>
        </div>

        {/* Header - Clean Apple Style */}
        <div className="bg-[hsl(168_25%_98%)] border-b border-[hsl(210_15%_92%)] p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center shadow-md shadow-[hsl(172_63%_22%)]/20">
                <MessageSquare className="h-5 w-5 text-white" />
              </div>
              <div className="flex flex-col">
                <h2 className="font-semibold text-[hsl(172_43%_15%)] text-lg tracking-tight">
                  Assistant
                </h2>
                <p className="text-xs text-[hsl(215_15%_50%)]">
                  {showHistory
                    ? "Conversation History"
                    : "AI-powered insights"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  const newState = !showHistory;
                  setShowHistory(newState);
                  if (newState) fetchSessions();
                }}
                title="History"
                className={cn(
                  "h-9 w-9 rounded-xl transition-all",
                  showHistory
                    ? "text-[hsl(172_63%_22%)] bg-[hsl(172_40%_94%)]"
                    : "text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)]"
                )}
              >
                <History className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => createNewSession()}
                title="New chat"
                className="h-9 w-9 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
              >
                <Plus className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                title="Close chat"
                className="h-9 w-9 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
        {/* Main Content Area */}
        <AnimatePresence mode="wait">
          {showHistory ? (
            <motion.div
              key="history"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col overflow-hidden"
            >
              <ScrollArea className="flex-1 p-4">
                {isLoadingSessions ? (
                  <div className="flex flex-col items-center justify-center p-12 gap-3">
                    <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
                    <p className="text-sm text-[hsl(215_15%_50%)]">
                      Loading history...
                    </p>
                  </div>
                ) : sessions.length === 0 ? (
                  <div className="text-center p-12">
                    <div className="w-12 h-12 rounded-2xl bg-[hsl(210_15%_95%)] flex items-center justify-center mx-auto mb-4">
                      <History className="h-6 w-6 text-[hsl(215_15%_50%)]" />
                    </div>
                    <p className="text-sm font-medium text-[hsl(172_43%_15%)]">
                      No history yet
                    </p>
                    <p className="text-xs text-[hsl(215_15%_55%)] mt-1">
                      Your conversations will appear here
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {sessions.map((session) => (
                      <div
                        key={session.id}
                        onClick={() => handleSwitchSession(session.id)}
                        className={cn(
                          "group p-4 rounded-2xl cursor-pointer transition-all border-2",
                          sessionId === session.id
                            ? "bg-[hsl(172_40%_96%)] border-[hsl(172_63%_35%)]"
                            : "bg-[hsl(210_15%_98%)] border-transparent hover:border-[hsl(210_15%_88%)] hover:bg-white"
                        )}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-[hsl(172_43%_15%)] truncate">
                              {session.title || "Untitled Conversation"}
                            </p>
                            <div className="flex items-center gap-2 mt-1">
                              <Database className="h-3 w-3 text-[hsl(215_15%_60%)]" />
                              <p className="text-[10px] text-[hsl(215_15%_55%)]">
                                {new Date(
                                  session.created_at
                                ).toLocaleDateString()}{" "}
                                at{" "}
                                {new Date(
                                  session.created_at
                                ).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit",
                                })}
                              </p>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClickCapture={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              handleDeleteSession(session.id, e);
                            }}
                            className="opacity-0 group-hover:opacity-100 h-8 w-8 rounded-lg text-red-500 hover:bg-red-50 hover:text-red-600 transition-all"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </motion.div>
          ) : (
            <motion.div
              key="chat"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col overflow-hidden"
            >
              {/* Selected Run Context Badge */}
              {selectedRun && (
                <div className="px-4 pt-4 pb-2">
                  <div className="bg-[hsl(172_40%_95%)] border border-[hsl(172_30%_88%)] rounded-2xl p-4 flex items-start gap-3">
                    <div className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center flex-shrink-0">
                      <BarChart className="h-5 w-5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <p className="text-sm font-semibold text-[hsl(172_43%_20%)]">
                          Run #{selectedRun.runId}
                        </p>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedRun(null)}
                          className="h-6 w-6 p-0 rounded-lg hover:bg-[hsl(172_30%_88%)] text-[hsl(215_15%_45%)]"
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                      <p className="text-xs text-[hsl(215_15%_50%)]">
                        {selectedRun.trainingMode} training
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* arXiv Research Tool Info Card */}
              {arxivAvailable && (
                <div className="px-4 pt-3 pb-2">
                  <div className="w-full bg-[hsl(168_25%_98%)] rounded-2xl p-4 border border-[hsl(168_20%_92%)] text-left">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-5 h-5 rounded-md bg-[hsl(172_63%_22%)] flex items-center justify-center">
                        <BookOpen className="h-3 w-3 text-white" />
                      </div>
                      <p className="text-xs font-semibold text-[hsl(172_43%_20%)]">
                        arXiv Research Tool
                      </p>
                    </div>
                    <p className="text-xs text-[hsl(215_15%_50%)] leading-relaxed">
                      Click the <BookOpen className="inline h-3 w-3 mx-0.5" />{" "}
                      button to enable academic paper search. When active, your
                      queries will be augmented with relevant research from
                      arXiv.
                    </p>
                  </div>
                </div>
              )}

              {/* Welcome State */}
              {messages.length === 0 && (
                <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
                  <div className="mb-6 w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
                    <MessageSquare className="w-8 h-8 text-[hsl(172_63%_28%)]" />
                  </div>
                  <h3 className="font-semibold text-[hsl(172_43%_15%)] text-xl mb-2">
                    Welcome
                  </h3>
                  <p className="text-sm text-[hsl(215_15%_50%)] leading-relaxed mb-6 max-w-xs text-balance">
                    Ask me anything about federated learning, pneumonia
                    detection, or your training runs.
                  </p>
                </div>
              )}

              {/* Messages Area */}
              <ScrollArea
                ref={scrollAreaRef}
                className={cn("flex-1", messages.length > 0 && "px-4 pt-4")}
              >
                <div className="space-y-4 pb-4">
                  <AnimatePresence initial={false}>
                    {messages.map((message, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{ duration: 0.2, delay: 0.05 }}
                        className={cn(
                          "flex",
                          message.role === "user"
                            ? "justify-end"
                            : "justify-start"
                        )}
                      >
                        <div
                          className={cn(
                            "max-w-[85%] px-4 py-3",
                            message.role === "user"
                              ? "bg-[hsl(172_63%_22%)] text-white rounded-2xl rounded-br-md shadow-md shadow-[hsl(172_63%_22%)]/15"
                              : "bg-[hsl(168_25%_96%)] text-[hsl(172_43%_15%)] rounded-2xl rounded-bl-md border border-[hsl(168_20%_92%)]"
                          )}
                        >
                          {message.role === "assistant" ? (
                            <Markdown
                              content={message.content}
                              className="text-[hsl(172_43%_15%)]"
                            />
                          ) : (
                            <p className="text-sm shadow-none">
                              {message.content}
                            </p>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>

                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-[hsl(168_25%_96%)] border border-[hsl(168_20%_92%)] rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-3">
                        <div className="flex gap-1.5">
                          <div className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce" />
                          <div
                            className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce"
                            style={{ animationDelay: "150ms" }}
                          />
                          <div
                            className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce"
                            style={{ animationDelay: "300ms" }}
                          />
                        </div>
                        <p className="text-sm text-[hsl(215_15%_50%)]">
                          {agentStatus || "Thinking..."}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Section */}
        <div className="border-t border-[hsl(210_15%_92%)] bg-white">
          {/* Run Picker Dropdown */}
          {showRunPicker && (
            <div
              ref={runPickerRef}
              className="border-b border-[hsl(210_15%_92%)] bg-[hsl(168_25%_98%)] max-h-72 overflow-y-auto"
            >
              <div className="p-4 border-b border-[hsl(210_15%_92%)] bg-white sticky top-0">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center">
                    <Activity className="h-4 w-4 text-[hsl(172_63%_28%)]" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-[hsl(172_43%_15%)]">
                      Select a Training Run
                    </p>
                    <p className="text-xs text-[hsl(215_15%_55%)]">
                      Arrow keys to navigate, Enter to select
                    </p>
                  </div>
                </div>
              </div>

              {loadingRuns ? (
                <div className="p-8 flex flex-col items-center justify-center gap-3">
                  <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
                  <p className="text-sm text-[hsl(215_15%_50%)]">
                    Loading runs...
                  </p>
                </div>
              ) : availableRuns.length === 0 ? (
                <div className="p-8 text-center">
                  <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-[hsl(210_15%_95%)] flex items-center justify-center">
                    <BarChart className="h-6 w-6 text-[hsl(215_15%_55%)]" />
                  </div>
                  <p className="text-sm font-medium text-[hsl(172_43%_20%)]">
                    No training runs found
                  </p>
                  <p className="text-xs text-[hsl(215_15%_55%)] mt-1">
                    Start a training run first
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-[hsl(210_15%_94%)]">
                  {availableRuns.map((run, index) => (
                    <button
                      key={run.id}
                      data-run-index={index}
                      onClick={() => handleSelectRun(run)}
                      className={cn(
                        "w-full p-4 transition-all duration-200 text-left flex items-center gap-3 group",
                        index === highlightedRunIndex
                          ? "bg-[hsl(172_40%_94%)] border-l-4 border-[hsl(172_63%_35%)]"
                          : "hover:bg-[hsl(168_25%_96%)] border-l-4 border-transparent"
                      )}
                    >
                      <div
                        className={cn(
                          "h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors",
                          run.training_mode === "federated"
                            ? "bg-[hsl(210_60%_92%)] text-[hsl(210_60%_40%)]"
                            : "bg-[hsl(152_50%_92%)] text-[hsl(152_60%_35%)]"
                        )}
                      >
                        <BarChart className="h-5 w-5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-[hsl(172_43%_15%)]">
                          Run #{run.id}
                        </p>
                        <p className="text-xs text-[hsl(215_15%_50%)] truncate mt-0.5">
                          {run.training_mode} • {formatRunTime(run.start_time)}
                        </p>
                        {run.best_val_recall > 0 && (
                          <p className="text-xs text-[hsl(152_60%_35%)] font-medium mt-1">
                            Best Recall:{" "}
                            {(run.best_val_recall * 100).toFixed(2)}%
                          </p>
                        )}
                      </div>
                      <div
                        className={cn(
                          "transition-all duration-200",
                          index === highlightedRunIndex
                            ? "opacity-100 scale-100"
                            : "opacity-0 scale-90 group-hover:opacity-100 group-hover:scale-100"
                        )}
                      >
                        <div className="w-8 h-8 rounded-lg bg-[hsl(172_63%_22%)] flex items-center justify-center">
                          <Send className="h-4 w-4 text-white" />
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Input Field */}
          <div className="p-4">
            <div className="flex gap-3">
              <Textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyPress}
                placeholder={
                  selectedRun
                    ? `Ask about Run #${selectedRun.runId}...`
                    : "Type / to select a run..."
                }
                disabled={isLoading}
                rows={1}
                className="flex-1 min-h-[44px] max-h-[200px] rounded-xl border-2 border-[hsl(210_15%_90%)] hover:border-[hsl(172_40%_80%)] focus:border-[hsl(172_63%_35%)] focus-visible:ring-0 transition-all duration-200 px-4 py-2.5 bg-[hsl(168_25%_99%)] placeholder:text-[hsl(215_15%_60%)] resize-none scrollbar-none"
              />
              <Button
                variant={arxivEnabled ? "default" : "outline"}
                size="icon"
                onClick={() => setArxivEnabled(!arxivEnabled)}
                disabled={!arxivAvailable}
                title={
                  arxivAvailable
                    ? arxivEnabled
                      ? "arXiv Research: ON - Searching academic papers for research-backed answers"
                      : "Enable arXiv Research - Search academic papers for research-backed answers"
                    : "arXiv unavailable - start backend first"
                }
                className={cn(
                  "h-11 w-11 rounded-xl transition-all duration-200",
                  arxivEnabled
                    ? "bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white border-0 shadow-md shadow-[hsl(172_63%_22%)]/20"
                    : "border-2 border-[hsl(210_15%_90%)] hover:border-[hsl(172_40%_80%)] hover:bg-[hsl(172_40%_94%)] text-[hsl(215_15%_45%)]",
                  !arxivAvailable && "opacity-50 cursor-not-allowed"
                )}
              >
                <BookOpen className="h-4 w-4" />
              </Button>
              <Button
                onClick={handleSendMessage}
                disabled={isLoading || !input.trim()}
                size="icon"
                className="h-11 w-11 rounded-xl bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white transition-all duration-200 hover:scale-105 disabled:opacity-40 disabled:hover:scale-100 shadow-md shadow-[hsl(172_63%_22%)]/20 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/30"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
