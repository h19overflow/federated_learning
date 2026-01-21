import React from "react";
import { Sparkles } from "lucide-react";

interface WelcomeStateProps {
  suggestedPrompts: string[];
  onSelectPrompt: (prompt: string) => void;
}

export const WelcomeState: React.FC<WelcomeStateProps> = ({
  suggestedPrompts,
  onSelectPrompt,
}) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 text-center">
      <div className="mb-6 w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center">
        <Sparkles className="w-8 h-8 text-[hsl(172_63%_28%)]" />
      </div>
      <h3 className="font-semibold text-[hsl(172_43%_15%)] text-xl mb-2">
        How can I help you today?
      </h3>
      <p className="text-sm text-[hsl(215_15%_50%)] leading-relaxed mb-8 max-w-xs text-balance">
        Select a suggested inquiry or type your own question below to start our
        conversation.
      </p>

      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2 justify-center mb-4">
          <div className="h-px w-8 bg-[hsl(172_30%_90%)]" />
          <p className="text-[10px] font-bold text-[hsl(172_63%_35%)] uppercase tracking-widest">
            Suggested Inquiries
          </p>
          <div className="h-px w-8 bg-[hsl(172_30%_90%)]" />
        </div>
        <div className="flex flex-wrap gap-2 justify-center">
          {suggestedPrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => onSelectPrompt(prompt)}
              className="px-4 py-2 text-xs font-medium text-[hsl(172_63%_22%)] bg-[hsl(172_40%_96%)] border border-[hsl(172_30%_90%)] rounded-full hover:bg-[hsl(172_63%_22%)] hover:text-white hover:border-[hsl(172_63%_22%)] hover:scale-105 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/20 transition-all duration-200 cursor-pointer active:scale-95"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
