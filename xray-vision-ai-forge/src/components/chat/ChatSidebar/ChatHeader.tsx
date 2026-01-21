import React from "react";
import { Button } from "@/components/ui/button";
import { MessageSquare, History, Plus, PanelLeftClose } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatHeaderProps {
  showHistory: boolean;
  onToggleHistory: () => void;
  onNewChat: () => void;
  onCollapse: () => void;
  isRailMode: boolean;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  showHistory,
  onToggleHistory,
  onNewChat,
  onCollapse,
  isRailMode,
}) => {
  return (
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
              {showHistory ? "Conversation History" : "AI-powered insights"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleHistory}
            title="History"
            aria-label="View chat history"
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
            onClick={onNewChat}
            title="New chat"
            aria-label="Start new chat"
            className="h-9 w-9 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
          >
            <Plus className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onCollapse}
            title={isRailMode ? "Close chat" : "Collapse to rail"}
            aria-label={isRailMode ? "Close chat sidebar" : "Collapse to rail"}
            className="h-9 w-9 rounded-xl text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(172_40%_94%)] transition-all"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
