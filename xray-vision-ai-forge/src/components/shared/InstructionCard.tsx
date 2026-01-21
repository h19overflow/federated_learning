import React from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Info,
  AlertTriangle,
  CheckCircle2,
  Lightbulb,
  BookOpen,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface InstructionCardProps {
  title?: string;
  children: React.ReactNode;
  variant?: "default" | "info" | "warning" | "success" | "tip" | "guide";
  icon?: React.ReactNode;
  className?: string;
}

/**
 * Instruction card component for displaying helpful information and guidance
 * Redesigned with Clinical Clarity theme
 */
const InstructionCard = ({
  title,
  children,
  variant = "default",
  icon,
  className,
}: InstructionCardProps) => {
  // Choose icon based on variant
  const getIcon = () => {
    if (icon) return icon;

    switch (variant) {
      case "info":
        return <Info className="h-5 w-5" />;
      case "warning":
        return <AlertTriangle className="h-5 w-5" />;
      case "success":
        return <CheckCircle2 className="h-5 w-5" />;
      case "tip":
        return <Lightbulb className="h-5 w-5" />;
      case "guide":
        return <BookOpen className="h-5 w-5" />;
      default:
        return <Sparkles className="h-5 w-5" />;
    }
  };

  // Get colors based on variant - Clinical Clarity theme
  const getVariantStyles = () => {
    switch (variant) {
      case "info":
        // Trust blue / sage
        return "bg-[hsl(210_100%_97%)] border-[hsl(210_60%_85%)]";
      case "warning":
        // Amber / sage mix
        return "bg-[hsl(35_60%_96%)] border-[hsl(35_50%_80%)]";
      case "success":
        // Teal success (not green!)
        return "bg-[hsl(172_40%_96%)] border-[hsl(172_40%_80%)]";
      case "tip":
        // Mint background
        return "bg-[hsl(168_40%_95%)] border-[hsl(168_35%_80%)]";
      case "guide":
        // Sage with subtle gradient
        return "bg-gradient-to-br from-[hsl(168_25%_97%)] to-[hsl(172_30%_95%)] border-[hsl(172_30%_85%)]";
      default:
        return "bg-[hsl(165_20%_97%)] border-[hsl(210_15%_90%)]";
    }
  };

  const getIconContainerStyles = () => {
    switch (variant) {
      case "info":
        return "bg-[hsl(210_80%_92%)]";
      case "warning":
        return "bg-[hsl(35_50%_90%)]";
      case "success":
        return "bg-[hsl(172_50%_90%)]";
      case "tip":
        return "bg-[hsl(168_45%_90%)]";
      case "guide":
        return "bg-[hsl(172_40%_92%)]";
      default:
        return "bg-[hsl(168_20%_92%)]";
    }
  };

  const getIconColor = () => {
    switch (variant) {
      case "info":
        return "text-[hsl(210_80%_45%)]";
      case "warning":
        return "text-[hsl(35_70%_45%)]";
      case "success":
        return "text-[hsl(172_63%_28%)]";
      case "tip":
        return "text-[hsl(168_55%_35%)]";
      case "guide":
        return "text-[hsl(172_63%_30%)]";
      default:
        return "text-[hsl(215_15%_50%)]";
    }
  };

  const getTitleColor = () => {
    switch (variant) {
      case "info":
        return "text-[hsl(210_70%_30%)]";
      case "warning":
        return "text-[hsl(35_70%_30%)]";
      case "success":
        return "text-[hsl(172_43%_22%)]";
      case "tip":
        return "text-[hsl(168_45%_25%)]";
      case "guide":
        return "text-[hsl(172_43%_20%)]";
      default:
        return "text-[hsl(172_43%_20%)]";
    }
  };

  const getTextColor = () => {
    switch (variant) {
      case "info":
        return "text-[hsl(210_50%_35%)]";
      case "warning":
        return "text-[hsl(35_50%_30%)]";
      case "success":
        return "text-[hsl(172_35%_30%)]";
      case "tip":
        return "text-[hsl(168_35%_30%)]";
      case "guide":
        return "text-[hsl(172_30%_30%)]";
      default:
        return "text-[hsl(215_15%_40%)]";
    }
  };

  return (
    <Alert
      className={cn(
        getVariantStyles(),
        "border rounded-2xl p-5 shadow-sm",
        className,
      )}
      style={{ animation: "fadeIn 0.3s ease-out" }}
    >
      <div className="flex items-start gap-4">
        <div
          className={cn(
            "flex-shrink-0 p-2.5 rounded-xl",
            getIconContainerStyles(),
          )}
        >
          <div className={getIconColor()}>{getIcon()}</div>
        </div>
        <div className="flex-1 space-y-1.5 min-w-0">
          {title && (
            <div className={cn("font-semibold text-base", getTitleColor())}>
              {title}
            </div>
          )}
          <AlertDescription
            className={cn("text-sm leading-relaxed", getTextColor())}
          >
            {children}
          </AlertDescription>
        </div>
      </div>
    </Alert>
  );
};

export default InstructionCard;
