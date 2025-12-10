import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { MessageSquare, Send, Trash2, Loader2, X, Sparkles, BarChart, Activity, Zap } from 'lucide-react';
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
      // Load history for existing session
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

  // Load available runs when slash command is detected
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
    
    // Add a system message to indicate run selection
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `📊 Now discussing Run #${run.id} (${run.training_mode} training). You can ask me about its metrics, performance, or training details.`,
      runContext,
    }]);
    
    setInput('');
    inputRef.current?.focus();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setInput(value);
    
    // Detect slash command
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
      
      // Include run context if a run is selected
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

      // Update session ID if it changed
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
      // Focus with a small delay to ensure state updates complete
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

      // Focus input after clearing
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
    // Handle arrow navigation in run picker
    if (showRunPicker && availableRuns.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setHighlightedRunIndex((prev) => {
          const newIndex = prev < availableRuns.length - 1 ? prev + 1 : prev;
          // Scroll highlighted item into view
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
          // Scroll highlighted item into view
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
    
    // Normal message sending
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-4 right-4 rounded-full h-14 w-14 shadow-xl hover:shadow-2xl bg-medical hover:bg-medical-dark text-white transition-all duration-300 hover:scale-110 animate-pulse"
          size="icon"
        >
          <MessageSquare className="h-6 w-6" />
        </Button>
      )}
      <div className={cn(
        "flex flex-col shadow-2xl border-l bg-gradient-to-b from-card via-card to-card text-card-foreground overflow-hidden transition-all duration-300 ease-in-out",
        "h-full",
        isOpen ? "w-96 opacity-100" : "w-0 opacity-0 pointer-events-none"
      )}>
      {/* Header with Gradient Background */}
      <div className="bg-gradient-to-r from-medical to-medical-dark text-white p-4 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <MessageSquare className="h-5 w-5" />
            </div>
            <div className="flex flex-col gap-0.5">
              <h2 className="font-bold text-lg">AI Assistant</h2>
              <p className="text-xs text-white/70">Powered by AI</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearChat}
              title="Clear chat"
              className="hover:bg-white/20 text-white transition-colors duration-200"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen(false)}
              title="Close chat"
              className="hover:bg-white/20 text-white transition-colors duration-200"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Selected Run Context Badge */}
      {selectedRun && (
        <div className="px-4 pt-4 pb-2">
          <div className="bg-medical/10 border border-medical/30 rounded-lg p-3 flex items-start gap-2">
            <BarChart className="h-5 w-5 text-medical flex-shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2 mb-1">
                <p className="text-sm font-semibold text-medical">Run #{selectedRun.runId}</p>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedRun(null)}
                  className="h-6 w-6 p-0 hover:bg-medical/20"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                {selectedRun.trainingMode}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Instructions / Welcome State */}
      {messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center px-4 text-center">
          <div className="mb-4 p-3 bg-medical/10 rounded-full">
            <Sparkles className="h-8 w-8 text-medical" />
          </div>
          <h3 className="font-semibold text-foreground mb-2">Welcome!</h3>
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
            Ask me anything about federated learning, pneumonia detection, or how to use this application.
          </p>
          <div className="bg-gradient-to-r from-medical/5 to-blue-500/5 rounded-lg p-3 border border-medical/20">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="h-4 w-4 text-medical" />
              <p className="text-xs font-semibold text-medical">Pro Tip</p>
            </div>
            <p className="text-xs text-muted-foreground text-left">
              Type <kbd className="px-1.5 py-0.5 bg-gray-200 rounded text-xs font-mono">/</kbd> to select a training run and get real-time insights about its metrics, performance, and more!
            </p>
          </div>
        </div>
      )}

      {/* Messages */}
      <ScrollArea ref={scrollAreaRef} className={cn("flex-1", messages.length > 0 && "p-4")}>
        <div className="space-y-3">
          {messages.map((message, index) => (
            <div
              key={index}
              className={cn(
                'flex animate-in fade-in slide-in-from-bottom-2 duration-300',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[80%] rounded-2xl px-4 py-2.5 shadow-sm transition-all duration-200 hover:shadow-md',
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-medical to-medical-dark text-white rounded-br-sm'
                    : 'bg-gray-100 dark:bg-gray-800 text-foreground rounded-bl-sm'
                )}
              >
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start animate-in fade-in duration-300">
              <div className="bg-gray-100 dark:bg-gray-800 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="h-2 w-2 bg-medical rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="h-2 w-2 bg-medical rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="h-2 w-2 bg-medical rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <p className="text-sm text-muted-foreground ml-1">Thinking...</p>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Section */}
      <div className="border-t bg-white dark:bg-card">
        {/* Run Picker Dropdown */}
        {showRunPicker && (
          <div ref={runPickerRef} className="border-b bg-gray-50 max-h-64 overflow-y-auto">
            <div className="p-3 border-b bg-white">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-medical" />
                <p className="text-sm font-semibold text-foreground">Select a Training Run</p>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Use arrow keys to navigate, Enter to select
              </p>
            </div>
            {loadingRuns ? (
              <div className="p-6 flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-medical" />
              </div>
            ) : availableRuns.length === 0 ? (
              <div className="p-6 text-center">
                <p className="text-sm text-muted-foreground">No training runs found</p>
                <p className="text-xs text-muted-foreground mt-1">Start a training run first</p>
              </div>
            ) : (
              <div className="divide-y">
                {availableRuns.map((run, index) => (
                  <button
                    key={run.id}
                    data-run-index={index}
                    onClick={() => handleSelectRun(run)}
                    className={cn(
                      "w-full p-3 transition-colors text-left flex items-center gap-3 group",
                      index === highlightedRunIndex
                        ? "bg-medical/10 border-l-4 border-medical"
                        : "hover:bg-medical/5"
                    )}
                  >
                    <div className={cn(
                      "h-10 w-10 rounded-lg flex items-center justify-center flex-shrink-0",
                      run.training_mode === 'federated' 
                        ? "bg-blue-100 text-blue-600" 
                        : "bg-green-100 text-green-600"
                    )}>
                      <BarChart className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="text-sm font-semibold text-foreground">Run #{run.id}</p>
                      </div>
                      <p className="text-xs text-muted-foreground truncate">
                        {run.training_mode} • {formatRunTime(run.start_time)}
                      </p>
                      {run.best_val_recall > 0 && (
                        <p className="text-xs text-medical font-medium mt-1">
                          Best Recall: {(run.best_val_recall * 100).toFixed(2)}%
                        </p>
                      )}
                    </div>
                    <div className={cn(
                      "transition-opacity",
                      index === highlightedRunIndex ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                    )}>
                      <Send className="h-4 w-4 text-medical" />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        
        {/* Input Field */}
        <div className="p-4">
          <div className="flex gap-2">
            <Input
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyPress}
              placeholder={selectedRun ? `Ask about Run #${selectedRun.runId}...` : "Type / to select a run or ask anything..."}
              disabled={isLoading}
              className="flex-1 rounded-full border-2 border-gray-200 hover:border-medical focus:border-medical focus:ring-0 transition-colors duration-200 px-4 py-2"
            />
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
              size="icon"
              className="rounded-full bg-medical hover:bg-medical-dark text-white transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:hover:scale-100 shadow-md hover:shadow-lg"
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
