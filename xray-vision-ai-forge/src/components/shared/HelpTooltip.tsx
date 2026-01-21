import React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HelpCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface HelpTooltipProps {
  content: React.ReactNode;
  title?: string;
  icon?: "help" | "info";
  side?: "top" | "right" | "bottom" | "left";
  className?: string;
  iconClassName?: string;
}

/**
 * Reusable help tooltip component for providing contextual information
 * Redesigned with Clinical Clarity theme - glass morphism style
 */
const HelpTooltip = ({
  content,
  title,
  icon = "help",
  side = "top",
  className,
  iconClassName,
}: HelpTooltipProps) => {
  const IconComponent = icon === "help" ? HelpCircle : Info;

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className={cn(
              "inline-flex items-center justify-center",
              "text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_35%)]",
              "transition-all duration-200",
              "focus:outline-none focus:ring-2 focus:ring-[hsl(172_63%_35%)]/30 focus:ring-offset-2",
              "rounded-full p-0.5",
              "hover:bg-[hsl(172_40%_95%)]",
              className,
            )}
            onClick={(e) => e.preventDefault()}
          >
            <IconComponent className={cn("h-4 w-4", iconClassName)} />
            <span className="sr-only">Help information</span>
          </button>
        </TooltipTrigger>
        <TooltipContent
          side={side}
          className={cn(
            "max-w-xs p-4 rounded-xl",
            "bg-white/95 backdrop-blur-md",
            "border border-[hsl(168_20%_90%)]",
            "shadow-lg shadow-[hsl(172_40%_85%)]/20",
          )}
          style={{ animation: "fadeIn 0.15s ease-out" }}
        >
          {title && (
            <div className="font-semibold mb-2 text-sm text-[hsl(172_43%_20%)]">
              {title}
            </div>
          )}
          <div className="text-sm text-[hsl(215_15%_40%)] leading-relaxed">
            {content}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default HelpTooltip;
