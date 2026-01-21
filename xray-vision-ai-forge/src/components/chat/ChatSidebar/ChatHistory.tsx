import React from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, History, Trash2, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import { ChatSession } from "./types";

interface ChatHistoryProps {
  sessions: ChatSession[];
  isLoading: boolean;
  currentSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string, e: React.MouseEvent) => void;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({
  sessions,
  isLoading,
  currentSessionId,
  onSelectSession,
  onDeleteSession,
}) => {
  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 gap-3">
        <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
        <p className="text-sm text-[hsl(215_15%_50%)]">Loading history...</p>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
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
    );
  }

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-3">
        {sessions.map((session) => (
          <div
            key={session.id}
            onClick={() => onSelectSession(session.id)}
            className={cn(
              "group p-4 rounded-2xl cursor-pointer transition-all border-2",
              currentSessionId === session.id
                ? "bg-[hsl(172_40%_96%)] border-[hsl(172_63%_35%)]"
                : "bg-[hsl(210_15%_98%)] border-transparent hover:border-[hsl(210_15%_88%)] hover:bg-white",
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
                    {new Date(session.created_at).toLocaleDateString()} at{" "}
                    {new Date(session.created_at).toLocaleTimeString([], {
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
                  onDeleteSession(session.id, e);
                }}
                className="opacity-0 group-hover:opacity-100 h-8 w-8 rounded-lg text-red-500 hover:bg-red-50 hover:text-red-600 transition-all"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
};
