import React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/ui/markdown";
import { motion, AnimatePresence } from "framer-motion";
import { CitationRenderer, parseCitations } from "../CitationRenderer";
import { Copy, MessageSquareQuote, RotateCcw, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { Message } from "./types";

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
  agentStatus: string | null;
  copiedIndex: number | null;
  scrollAreaRef: React.RefObject<HTMLDivElement>;
  onCopy: (text: string, index: number) => void;
  onQuote: (text: string) => void;
}

export const ChatMessages: React.FC<ChatMessagesProps> = ({
  messages,
  isLoading,
  agentStatus,
  copiedIndex,
  scrollAreaRef,
  onCopy,
  onQuote,
}) => {
  return (
    <ScrollArea
      ref={scrollAreaRef}
      role="log"
      aria-label="Chat messages"
      className={cn("flex-1", messages.length > 0 && "px-4 pt-4")}
    >
      <div className="space-y-4 pb-4" aria-live="polite">
        <AnimatePresence initial={false}>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.2, delay: 0.05 }}
              className={cn(
                "flex",
                message.role === "user" ? "justify-end" : "justify-start",
              )}
            >
              <div
                className={cn(
                  "max-w-[85%] px-4 py-3 relative group",
                  message.role === "user"
                    ? "bg-[hsl(172_63%_22%)] text-white rounded-2xl rounded-br-md shadow-md shadow-[hsl(172_63%_22%)]/15"
                    : "bg-[hsl(168_25%_96%)] text-[hsl(172_43%_15%)] rounded-2xl rounded-bl-md border border-[hsl(168_20%_92%)]",
                )}
              >
                {/* Message Toolbar */}
                {message.role === "user" && (
                  <div className="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 backdrop-blur-sm p-0.5 rounded-lg border border-[hsl(210_15%_90%)] shadow-sm z-10">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onCopy(message.content, index)}
                      className="h-6 w-6 p-1 rounded-md hover:bg-[hsl(172_40%_94%)] text-[hsl(215_15%_45%)]"
                      title="Copy message"
                      aria-label="Copy message"
                    >
                      {copiedIndex === index ? (
                        <Check className="h-3.5 w-3.5 text-green-500" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onQuote(message.content)}
                      className="h-6 w-6 p-1 rounded-md hover:bg-[hsl(172_40%_94%)] text-[hsl(215_15%_45%)]"
                      title="Quote message"
                      aria-label="Quote message"
                    >
                      <MessageSquareQuote className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                )}
                {message.role === "assistant" && (
                  <div className="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 backdrop-blur-sm p-0.5 rounded-lg border border-[hsl(210_15%_90%)] shadow-sm z-10">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => onQuote(message.content)}
                      className="h-6 w-6 p-1 rounded-md hover:bg-[hsl(172_40%_94%)] text-[hsl(215_15%_45%)]"
                      title="Quote message"
                      aria-label="Quote message"
                    >
                      <MessageSquareQuote className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                )}

                {message.role === "assistant" ? (
                  (() => {
                    const { cleanedContent, citations } = parseCitations(
                      message.content,
                    );
                    return (
                      <>
                        <Markdown
                          content={cleanedContent}
                          citations={citations}
                          className="text-[hsl(172_43%_15%)]"
                        />
                        <CitationRenderer citations={citations} />
                      </>
                    );
                  })()
                ) : (
                  <p className="text-sm shadow-none whitespace-pre-wrap">
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
  );
};
