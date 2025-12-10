import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageSquare, Send, Trash2, Loader2, X, BarChart, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import api from '@/services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  runContext?: RunContext;
}

interface RunContext {
  runId: number;
  trainingMode: string;
  status: string;
  startTime: string;
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
  apiUrl = 'http://127.0.0.1:8001'
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [isOpen, setIsOpen] = useState(true);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const runPickerRef = useRef<HTMLDivElement>(null);

  // Slash command state
  const [showRunPicker, setShowRunPicker] = useState(false);
  const [availableRuns, setAvailableRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunContext | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [highlightedRunIndex, setHighlightedRunIndex] = useState(0);

  // Generate session ID on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem('chat_session_id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      loadHistory(storedSessionId);
    } else {
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      localStorage.setItem('chat_session_id', newSessionId);
    }
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [messages]);

  const generateSessionId = (): string => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const fetchAvailableRuns = async () => {
    try {
      setLoadingRuns(true);
      const response = await api.results.listRuns();
      setAvailableRuns(response.runs);
    } catch (error) {
      console.error('Error fetching runs:', error);
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
          loadedMessages.push({ role: 'user', content: item.user });
          loadedMessages.push({ role: 'assistant', content: item.assistant });
        });
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const handleSelectRun = (run: RunSummary) => {
    const runContext: RunContext = {
      runId: run.id,
      trainingMode: run.training_mode,
      status: run.status,
      startTime: run.start_time || 'Unknown',
    };
    setSelectedRun(runContext);
    setShowRunPicker(false);

    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `Now discussing Run #${run.id} (${run.training_mode} training). You can ask me about its metrics, performance, or training details.`,
      runContext,
    }]);

    setInput('');
    inputRef.current?.focus();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInput(value);

    if (value === '/') {
      setShowRunPicker(true);
      setHighlightedRunIndex(0);
      if (availableRuns.length === 0) {
        fetchAvailableRuns();
      }
    } else if (value.length === 0 || !value.startsWith('/')) {
      setShowRunPicker(false);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setShowRunPicker(false);

    const newMessage: Message = {
      role: 'user',
      content: userMessage,
      runContext: selectedRun || undefined,
    };

    setMessages(prev => [...prev, newMessage]);
    setIsLoading(true);

    try {
      const requestBody: any = {
        query: userMessage,
        session_id: sessionId,
      };

      if (selectedRun) {
        requestBody.run_id = selectedRun.runId;
        requestBody.training_mode = selectedRun.trainingMode;
      }

      const response = await fetch(`${apiUrl}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: data.answer, runContext: selectedRun || undefined },
      ]);

      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem('chat_session_id', data.session_id);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error while processing your query. Please try again.',
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    }
  };

  const handleClearChat = async () => {
    try {
      await fetch(`${apiUrl}/chat/history/${sessionId}`, {
        method: 'DELETE',
      });

      setMessages([]);
      setSelectedRun(null);
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      localStorage.setItem('chat_session_id', newSessionId);

      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);
    } catch (error) {
      console.error('Error clearing chat:', error);
    }
  };

  const formatRunTime = (timeStr: string | null) => {
    if (!timeStr) return 'Unknown';
    try {
      const date = new Date(timeStr);
      return date.toLocaleString();
    } catch {
      return timeStr;
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (showRunPicker && availableRuns.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setHighlightedRunIndex((prev) => {
          const newIndex = prev < availableRuns.length - 1 ? prev + 1 : prev;
          setTimeout(() => {
            const runItem = runPickerRef.current?.querySelector(`[data-run-index="${newIndex}"]`);
            runItem?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }, 0);
          return newIndex;
        });
        return;
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setHighlightedRunIndex((prev) => {
          const newIndex = prev > 0 ? prev - 1 : prev;
          setTimeout(() => {
            const runItem = runPickerRef.current?.querySelector(`[data-run-index="${newIndex}"]`);
            runItem?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }, 0);
          return newIndex;
        });
        return;
      } else if (e.key === 'Enter') {
        e.preventDefault();
        handleSelectRun(availableRuns[highlightedRunIndex]);
        return;
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setShowRunPicker(false);
        setInput('');
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Floating Action Button - Apple Style */}
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 rounded-2xl h-14 w-14 shadow-xl shadow-[hsl(172_63%_22%)]/25 hover:shadow-2xl hover:shadow-[hsl(172_63%_22%)]/35 bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white transition-all duration-300 hover:scale-105"
          size="icon"
        >
          <MessageSquare className="h-6 w-6" />
        </Button>
      )}

      {/* Sidebar Container */}
      <div className={cn(
        "flex flex-col border-l border-[hsl(210_15%_92%)] bg-white overflow-hidden transition-all duration-300 ease-out",
        "h-full",
        isOpen ? "w-96 opacity-100" : "w-0 opacity-0 pointer-events-none"
      )}>

        {/* Header - Clean Apple Style */}
        <div className="bg-[hsl(168_25%_98%)] border-b border-[hsl(210_15%_92%)] p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center shadow-md shadow-[hsl(172_63%_22%)]/20">
                <MessageSquare className="h-5 w-5 text-white" />
              </div>
              <div className="flex flex-col">
                <h2 className="font-semibold text-[hsl(172_43%_15%)] text-lg tracking-tight">Assistant</h2>
                <p className="text-xs text-[hsl(215_15%_50%)]">AI-powered insights</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleClearChat}
                title="Clear chat"
                className="h-9 w-9 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
              >
                <Trash2 className="h-4 w-4" />
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

        {/* Selected Run Context Badge */}
        {selectedRun && (
          <div className="px-4 pt-4 pb-2">
            <div className="bg-[hsl(172_40%_95%)] border border-[hsl(172_30%_88%)] rounded-2xl p-4 flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center flex-shrink-0">
                <BarChart className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <p className="text-sm font-semibold text-[hsl(172_43%_20%)]">Run #{selectedRun.runId}</p>
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

        {/* Welcome State - Apple Style */}
        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
            <div className="mb-6 w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
              <svg className="w-8 h-8 text-[hsl(172_63%_28%)]" viewBox="0 0 32 32" fill="none">
                <path d="M16 4v24M4 16h24" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                <circle cx="16" cy="16" r="12" stroke="currentColor" strokeWidth="1.5" opacity="0.3" />
              </svg>
            </div>
            <h3 className="font-semibold text-[hsl(172_43%_15%)] text-xl mb-2">Welcome</h3>
            <p className="text-sm text-[hsl(215_15%_50%)] leading-relaxed mb-6 max-w-xs">
              Ask me anything about federated learning, pneumonia detection, or your training runs.
            </p>

            {/* Pro Tip Card */}
            <div className="w-full bg-[hsl(168_25%_98%)] rounded-2xl p-4 border border-[hsl(168_20%_92%)]">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-md bg-[hsl(172_63%_22%)] flex items-center justify-center">
                  <Activity className="h-3 w-3 text-white" />
                </div>
                <p className="text-xs font-semibold text-[hsl(172_43%_20%)]">Quick Tip</p>
              </div>
              <p className="text-xs text-[hsl(215_15%_50%)] text-left leading-relaxed">
                Type <kbd className="px-1.5 py-0.5 bg-white border border-[hsl(210_15%_88%)] rounded-md text-xs font-mono text-[hsl(172_63%_28%)]">/</kbd> to select a training run and get real-time insights about its metrics.
              </p>
            </div>
          </div>
        )}

        {/* Messages Area */}
        <ScrollArea ref={scrollAreaRef} className={cn("flex-1", messages.length > 0 && "px-4 pt-4")}>
          <div className="space-y-4 pb-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  'flex',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
                style={{
                  animation: 'fadeIn 0.3s ease-out forwards',
                  animationDelay: `${index * 0.05}s`,
                  opacity: 0
                }}
              >
                <div
                  className={cn(
                    'max-w-[85%] px-4 py-3 transition-all duration-200',
                    message.role === 'user'
                      ? 'bg-[hsl(172_63%_22%)] text-white rounded-2xl rounded-br-md shadow-md shadow-[hsl(172_63%_22%)]/15'
                      : 'bg-[hsl(168_25%_96%)] text-[hsl(172_43%_15%)] rounded-2xl rounded-bl-md border border-[hsl(168_20%_92%)]'
                  )}
                >
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                </div>
              </div>
            ))}

            {/* Loading State */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-[hsl(168_25%_96%)] border border-[hsl(168_20%_92%)] rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-3">
                  <div className="flex gap-1.5">
                    <div className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="h-2 w-2 bg-[hsl(172_63%_35%)] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <p className="text-sm text-[hsl(215_15%_50%)]">Thinking...</p>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Input Section */}
        <div className="border-t border-[hsl(210_15%_92%)] bg-white">
          {/* Run Picker Dropdown */}
          {showRunPicker && (
            <div ref={runPickerRef} className="border-b border-[hsl(210_15%_92%)] bg-[hsl(168_25%_98%)] max-h-72 overflow-y-auto">
              <div className="p-4 border-b border-[hsl(210_15%_92%)] bg-white sticky top-0">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-[hsl(172_40%_94%)] flex items-center justify-center">
                    <Activity className="h-4 w-4 text-[hsl(172_63%_28%)]" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-[hsl(172_43%_15%)]">Select a Training Run</p>
                    <p className="text-xs text-[hsl(215_15%_55%)]">
                      Arrow keys to navigate, Enter to select
                    </p>
                  </div>
                </div>
              </div>

              {loadingRuns ? (
                <div className="p-8 flex flex-col items-center justify-center gap-3">
                  <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
                  <p className="text-sm text-[hsl(215_15%_50%)]">Loading runs...</p>
                </div>
              ) : availableRuns.length === 0 ? (
                <div className="p-8 text-center">
                  <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-[hsl(210_15%_95%)] flex items-center justify-center">
                    <BarChart className="h-6 w-6 text-[hsl(215_15%_55%)]" />
                  </div>
                  <p className="text-sm font-medium text-[hsl(172_43%_20%)]">No training runs found</p>
                  <p className="text-xs text-[hsl(215_15%_55%)] mt-1">Start a training run first</p>
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
                      <div className={cn(
                        "h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors",
                        run.training_mode === 'federated'
                          ? "bg-[hsl(210_60%_92%)] text-[hsl(210_60%_40%)]"
                          : "bg-[hsl(152_50%_92%)] text-[hsl(152_60%_35%)]"
                      )}>
                        <BarChart className="h-5 w-5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-[hsl(172_43%_15%)]">Run #{run.id}</p>
                        <p className="text-xs text-[hsl(215_15%_50%)] truncate mt-0.5">
                          {run.training_mode} • {formatRunTime(run.start_time)}
                        </p>
                        {run.best_val_recall > 0 && (
                          <p className="text-xs text-[hsl(152_60%_35%)] font-medium mt-1">
                            Best Recall: {(run.best_val_recall * 100).toFixed(2)}%
                          </p>
                        )}
                      </div>
                      <div className={cn(
                        "transition-all duration-200",
                        index === highlightedRunIndex ? "opacity-100 scale-100" : "opacity-0 scale-90 group-hover:opacity-100 group-hover:scale-100"
                      )}>
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
              <Input
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyPress}
                placeholder={selectedRun ? `Ask about Run #${selectedRun.runId}...` : "Type / to select a run..."}
                disabled={isLoading}
                className="flex-1 rounded-xl border-2 border-[hsl(210_15%_90%)] hover:border-[hsl(172_40%_80%)] focus:border-[hsl(172_63%_35%)] focus:ring-0 transition-colors duration-200 px-4 py-2.5 bg-[hsl(168_25%_99%)] placeholder:text-[hsl(215_15%_60%)]"
              />
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
