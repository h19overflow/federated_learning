import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  GripVertical,
  Plus,
  History,
  ChevronRight,
  BookOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import api from "@/services/api";

import { ChatHeader } from "./ChatHeader";
import { ChatHistory } from "./ChatHistory";
import { ChatMessages } from "./ChatMessages";
import { ChatInput } from "./ChatInput";
import { RunPicker } from "./RunPicker";
import { RunContextBadge } from "./RunContextBadge";
import { WelcomeState } from "./WelcomeState";
import {
  Message,
  RunContext,
  ChatSession,
  RunSummary,
  ChatSidebarProps,
} from "./types";

const MIN_WIDTH = 320;
const MAX_WIDTH = 800;
const DEFAULT_WIDTH = 384;

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  apiUrl = "http://127.0.0.1:8001",
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [isOpen, setIsOpen] = useState(true);
  const [isRailMode, setIsRailMode] = useState(false);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);

  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showRunPicker, setShowRunPicker] = useState(false);
  const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunContext | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [arxivEnabled, setArxivEnabled] = useState(false);
  const [arxivAvailable, setArxivAvailable] = useState(false);
  const [agentStatus, setAgentStatus] = useState<string | null>(null);

  const recentRuns = React.useMemo(() => {
    return [...availableRuns]
      .filter((r) => r.start_time)
      .sort(
        (a, b) =>
          new Date(b.start_time!).getTime() - new Date(a.start_time!).getTime()
      )
      .slice(0, 5);
  }, [availableRuns]);

  const federatedRuns = React.useMemo(() => {
    return availableRuns.filter((r) => r.training_mode === "federated");
  }, [availableRuns]);

  const centralizedRuns = React.useMemo(() => {
    return availableRuns.filter((r) => r.training_mode === "centralized");
  }, [availableRuns]);

  useEffect(() => {
    const checkSession = async () => {
      const storedSessionId = localStorage.getItem("chat_session_id");
      const isUUID =
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

      if (storedSessionId && isUUID.test(storedSessionId)) {
        setSessionId(storedSessionId);
        loadHistory(storedSessionId);
      } else {
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
          createNewSession();
        } else {
          fetchSessions();
        }
      }
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 5;
    const retryDelay = 2000;

    const checkArxivStatus = async () => {
      try {
        const res = await fetch(`${apiUrl}/chat/arxiv/status`);
        if (res.ok) {
          const data = await res.json();
          console.log("[ChatSidebar] Arxiv status:", data);
          setArxivAvailable(data.available);
          return data.available;
        } else {
          console.warn("[ChatSidebar] Arxiv status check failed:", res.status);
          setArxivAvailable(false);
          return false;
        }
      } catch (error) {
        console.warn("[ChatSidebar] Arxiv status check error:", error);
        setArxivAvailable(false);
        return false;
      }
    };

    const checkWithRetry = async () => {
      const available = await checkArxivStatus();
      if (!available && retryCount < maxRetries) {
        retryCount++;
        console.log(
          `[ChatSidebar] Arxiv not available, retry ${retryCount}/${maxRetries} in ${retryDelay}ms`
        );
        setTimeout(checkWithRetry, retryDelay);
      }
    };

    checkWithRetry();
  }, [apiUrl]);

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
        setIsOpen(false);
        setIsResizing(false);
      }
    },
    [isResizing]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

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
      bestRecall: run.best_val_recall,
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

    if (availableRuns.length === 0) {
      fetchAvailableRuns();
    }
  };

  const getSuggestedPrompts = () => {
    if (selectedRun) {
      return [
        "Show training metrics",
        "Compare with previous run",
        "Explain the model architecture",
      ];
    }
    if (arxivEnabled) {
      return [
        "Latest federated learning papers",
        "Privacy-preserving techniques",
        "Pneumonia detection methods",
      ];
    }
    return [
      "What is federated learning?",
      "How does the system work?",
      "Show me available training runs",
    ];
  };

  const handleSendMessage = async (overrideInput?: string) => {
    console.log("[ChatSidebar] handleSendMessage called");
    console.log("[ChatSidebar] overrideInput:", overrideInput);
    console.log("[ChatSidebar] current input:", input);

    const messageToSend = overrideInput || input;

    console.log("[ChatSidebar] messageToSend:", messageToSend);
    console.log("[ChatSidebar] isLoading:", isLoading);

    if (!messageToSend.trim() || isLoading) {
      console.log("[ChatSidebar] Aborting - empty message or loading");
      return;
    }

    const userMessage = messageToSend.trim();
    console.log("[ChatSidebar] Sending user message:", userMessage);

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

    setMessages((prev) => [...prev, newMessage]);
    setIsLoading(true);
    setAgentStatus(null);

    try {
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

        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "token") {
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastMsg = updated[updated.length - 1];
                  if (lastMsg && lastMsg.role === "assistant") {
                    lastMsg.content += data.content;
                  } else {
                    updated.push({
                      role: "assistant",
                      content: data.content,
                      runContext: selectedRun || undefined,
                    });
                  }
                  return updated;
                });
              } else if (data.type === "session") {
                if (data.session_id && data.session_id !== sessionId) {
                  setSessionId(data.session_id);
                  localStorage.setItem("chat_session_id", data.session_id);
                }
              } else if (data.type === "error") {
                console.error("Stream error:", data.message);
                setAgentStatus(null);
                setMessages((prev) => {
                  if (prev.length === 0) {
                    return prev;
                  }
                  const updated = [...prev];
                  const lastMsg = updated[updated.length - 1];
                  if (lastMsg && lastMsg.role === "assistant") {
                    lastMsg.content =
                      "Sorry, I encountered an error while processing your query. Please try again.";
                  }
                  return updated;
                });
              } else if (data.type === "status") {
                setAgentStatus(data.content);
              } else if (data.type === "tool_call") {
                if (data.tool === "arxiv" || data.tool?.includes("arxiv")) {
                  setAgentStatus("Searching arXiv papers...");
                } else {
                  const toolName = data.tool?.replace(/_/g, " ") || "a tool";
                  setAgentStatus(`Using ${toolName}...`);
                }
              } else if (data.type === "done") {
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
        if (prev.length === 0) {
          return prev;
        }
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg && lastMsg.role === "assistant") {
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

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleQuote = (text: string) => {
    setInput((prev) => `> ${text}\n\n${prev}`);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (showRunPicker) {
      if (e.key === "Escape") {
        e.preventDefault();
        setShowRunPicker(false);
        setInput("");
        inputRef.current?.focus();
      }
      return;
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      console.log("[ChatSidebar] Enter key pressed - calling handleSendMessage");
      handleSendMessage();
    }
  };

  return (
    <>
      {!isOpen && (
        <Button
          onClick={() => {
            setWidth(DEFAULT_WIDTH);
            setIsOpen(true);
            setIsRailMode(false);
          }}
          className="fixed bottom-6 right-6 rounded-2xl h-14 w-14 shadow-xl shadow-[hsl(172_63%_22%)]/25 hover:shadow-2xl hover:shadow-[hsl(172_63%_22%)]/35 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white transition-all duration-300 hover:scale-105"
          size="icon"
          aria-label="Open chat assistant"
        >
          <MessageSquare className="h-6 w-6" />
        </Button>
      )}

      <div
        ref={sidebarRef}
        className={cn(
          "flex flex-col border-l border-[hsl(210_15%_92%)] bg-white overflow-hidden relative",
          "h-full",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none",
          !isResizing && "transition-all duration-300 ease-out"
        )}
        style={{ width: isOpen ? (isRailMode ? 64 : width) : 0 }}
      >
        {!isRailMode && (
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
        )}

        {isRailMode ? (
          <div className="flex flex-col items-center py-6 gap-6 h-full bg-[hsl(168_25%_98%)]">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsRailMode(false)}
              className="h-10 w-10 rounded-xl bg-white shadow-sm text-[hsl(172_63%_22%)] border border-[hsl(210_15%_90%)] hover:bg-[hsl(172_40%_94%)] transition-all"
            >
              <ChevronRight className="h-5 w-5" />
            </Button>

            <div className="flex flex-col gap-4 mt-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  createNewSession();
                  setIsRailMode(false);
                }}
                title="New Chat"
                className="h-10 w-10 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
              >
                <Plus className="h-5 w-5" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setShowHistory(true);
                  fetchSessions();
                  setIsRailMode(false);
                }}
                title="History"
                className="h-10 w-10 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
              >
                <History className="h-5 w-5" />
              </Button>

              {selectedRun && (
                <div
                  className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] text-white flex items-center justify-center text-[10px] font-bold shadow-md cursor-pointer hover:scale-105 transition-all"
                  onClick={() => setIsRailMode(false)}
                  title={`Run #${selectedRun.runId}`}
                >
                  {selectedRun.runId}
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            <ChatHeader
              showHistory={showHistory}
              onToggleHistory={() => {
                const newState = !showHistory;
                setShowHistory(newState);
                if (newState) fetchSessions();
              }}
              onNewChat={() => createNewSession()}
              onCollapse={() => {
                if (isRailMode) {
                  setIsOpen(false);
                  setIsRailMode(false);
                } else {
                  setIsRailMode(true);
                }
              }}
              isRailMode={isRailMode}
            />

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
                  <ChatHistory
                    sessions={sessions}
                    isLoading={isLoadingSessions}
                    currentSessionId={sessionId}
                    onSelectSession={handleSwitchSession}
                    onDeleteSession={handleDeleteSession}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="chat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.2 }}
                  className="flex-1 flex flex-col overflow-hidden"
                  aria-busy={isLoading}
                >
                  {selectedRun && (
                    <RunContextBadge
                      selectedRun={selectedRun}
                      onClear={() => setSelectedRun(null)}
                    />
                  )}

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
                          Click the <BookOpen className="inline h-3 w-3 mx-0.5" /> button
                          to enable academic paper search. When active, your queries will be
                          augmented with relevant research from arXiv.
                        </p>
                      </div>
                    </div>
                  )}

                  {messages.length === 0 && (
                    <WelcomeState
                      suggestedPrompts={getSuggestedPrompts()}
                      onSelectPrompt={handleSendMessage}
                    />
                  )}

                  <ChatMessages
                    messages={messages}
                    isLoading={isLoading}
                    agentStatus={agentStatus}
                    copiedIndex={copiedIndex}
                    scrollAreaRef={scrollAreaRef}
                    onCopy={handleCopy}
                    onQuote={handleQuote}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <RunPicker
              show={showRunPicker}
              runs={availableRuns}
              recentRuns={recentRuns}
              federatedRuns={federatedRuns}
              centralizedRuns={centralizedRuns}
              isLoading={loadingRuns}
              onSelectRun={handleSelectRun}
              onClose={() => {
                setShowRunPicker(false);
                setInput("");
                inputRef.current?.focus();
              }}
            />

            <ChatInput
              input={input}
              setInput={setInput}
              onSend={handleSendMessage}
              isLoading={isLoading}
              arxivEnabled={arxivEnabled}
              setArxivEnabled={setArxivEnabled}
              arxivAvailable={arxivAvailable}
              selectedRun={selectedRun}
              showRunPicker={showRunPicker}
              setShowRunPicker={setShowRunPicker}
              inputRef={inputRef}
              onKeyDown={handleKeyPress}
            />
          </>
        )}
      </div>
    </>
  );
};
