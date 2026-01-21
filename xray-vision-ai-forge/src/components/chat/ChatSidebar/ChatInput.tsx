import React from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { BookOpen, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { RunContext } from "./types";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  arxivEnabled: boolean;
  setArxivEnabled: (enabled: boolean) => void;
  arxivAvailable: boolean;
  selectedRun: RunContext | null;
  showRunPicker: boolean;
  setShowRunPicker: (show: boolean) => void;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onKeyDown: (e: React.KeyboardEvent) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  onSend,
  isLoading,
  arxivEnabled,
  setArxivEnabled,
  arxivAvailable,
  selectedRun,
  showRunPicker,
  setShowRunPicker,
  inputRef,
  onKeyDown,
}) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    console.log("[ChatInput] Input changed:", value);
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
      console.log("[ChatInput] Showing run picker");
      setShowRunPicker(true);
    } else if (value.length === 0 || !value.startsWith("/")) {
      setShowRunPicker(false);
    }
  };

  const handleSendClick = () => {
    console.log("[ChatInput] Send button clicked");
    console.log("[ChatInput] Input value:", input);
    console.log("[ChatInput] Is loading:", isLoading);
    console.log("[ChatInput] Calling onSend()");
    onSend();
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("[ChatInput] Form submitted");
    console.log("[ChatInput] Input value:", input);
    console.log("[ChatInput] Is loading:", isLoading);
    console.log("[ChatInput] Calling onSend()");
    onSend();
  };

  const canSend = !isLoading && input.trim().length > 0;

  console.log("[ChatInput] Render - canSend:", canSend, "input:", input, "isLoading:", isLoading);

  return (
    <div className="border-t border-[hsl(210_15%_92%)] bg-white">
      <div className="p-4">
        <form onSubmit={handleFormSubmit} className="flex gap-3 relative z-10">
          <Textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={onKeyDown}
            placeholder={
              arxivEnabled
                ? "Search with arXiv research..."
                : selectedRun
                ? `Ask about Run #${selectedRun.runId}...`
                : "Type / to select a run..."
            }
            disabled={isLoading}
            rows={1}
            className={cn(
              "flex-1 min-h-[44px] max-h-[200px] rounded-xl border-2 border-[hsl(210_15%_90%)] hover:border-[hsl(172_40%_80%)] focus:border-[hsl(172_63%_35%)] focus-visible:ring-0 transition-all duration-200 px-4 py-2.5 bg-[hsl(168_25%_99%)] placeholder:text-[hsl(215_15%_60%)] resize-none scrollbar-none",
              arxivEnabled &&
                "ring-2 ring-[hsl(172_63%_35%)]/30 border-[hsl(172_63%_35%)]"
            )}
          />
          <Button
            type="button"
            variant={arxivEnabled ? "default" : "outline"}
            size="icon"
            onClick={() => {
              console.log("[ChatInput] ArXiv button clicked");
              setArxivEnabled(!arxivEnabled);
            }}
            disabled={!arxivAvailable}
            title={
              arxivAvailable
                ? arxivEnabled
                  ? "arXiv Research: ON - Searching academic papers for research-backed answers"
                  : "Enable arXiv Research - Search academic papers for research-backed answers"
                : "arXiv unavailable - start backend first"
            }
            aria-label={
              arxivEnabled ? "Disable arXiv research" : "Enable arXiv research"
            }
            className={cn(
              "h-11 w-11 rounded-xl transition-all duration-200 flex-shrink-0",
              arxivEnabled
                ? "bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white border-0 shadow-md shadow-[hsl(172_63%_22%)]/20"
                : "border-2 border-[hsl(210_15%_90%)] hover:border-[hsl(172_40%_80%)] hover:bg-[hsl(172_40%_94%)] text-[hsl(215_15%_45%)]",
              !arxivAvailable && "opacity-50 cursor-not-allowed"
            )}
          >
            <BookOpen className="h-4 w-4" />
          </Button>
          <Button
            type="submit"
            onClick={handleSendClick}
            disabled={!canSend}
            size="icon"
            aria-label="Send message"
            className="h-11 w-11 rounded-xl bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white transition-all duration-300 hover:scale-105 disabled:opacity-40 disabled:hover:scale-100 shadow-md shadow-[hsl(172_63%_22%)]/20 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/30 flex-shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
};
