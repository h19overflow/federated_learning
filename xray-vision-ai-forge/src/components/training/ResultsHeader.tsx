import React from "react";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, FlaskConical, Users, Server } from "lucide-react";
import { ExperimentConfiguration } from "@/types/experiment";
import { cn } from "@/lib/utils";

interface ResultsHeaderProps {
  config: ExperimentConfiguration;
  runId: number;
  showComparison: boolean;
}

const ResultsHeader: React.FC<ResultsHeaderProps> = ({
  config,
  runId,
  showComparison,
}) => {
  const isFederated = config.trainingMode === "federated";
  const isComparison = config.trainingMode === "comparison";

  const getModeIcon = () => {
    if (showComparison || isComparison) return <FlaskConical className="h-4 w-4" />;
    if (isFederated) return <Users className="h-4 w-4" />;
    return <Server className="h-4 w-4" />;
  };

  const getModeLabel = () => {
    if (showComparison || isComparison) return "Comparison";
    if (isFederated) return "Federated";
    return "Centralized";
  };

  return (
    <div className="relative border-b border-[hsl(172_20%_90%)]">
      {/* Subtle gradient background matching Landing theme */}
      <div className="absolute inset-0 bg-gradient-to-b from-[hsl(168_25%_98%)] to-white" />

      {/* Content */}
      <div className="relative px-6 py-5 md:px-8 md:py-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          {/* Left: Icon and Title */}
          <div className="flex items-center gap-3">
            {/* Main Icon */}
            <div className="p-2.5 rounded-xl bg-[hsl(172_40%_95%)] border border-[hsl(172_30%_88%)] shadow-sm">
              <TrendingUp className="h-5 w-5 text-[hsl(172_63%_28%)]" />
            </div>

            {/* Title and Subtitle */}
            <div>
              <h2 className="text-lg md:text-xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                {showComparison ? "Comparison Results" : "Experiment Results"}
              </h2>
              
              {/* Mode and Run Info */}
              <div className="flex items-center gap-2 mt-0.5">
                <span className="flex items-center gap-1.5 text-sm text-[hsl(172_43%_35%)]">
                  {getModeIcon()}
                  {getModeLabel()}
                </span>
                <span className="text-[hsl(172_20%_80%)]">•</span>
                <span className="text-sm text-[hsl(172_15%_50%)] font-mono">
                  #{runId.toString().padStart(3, '0')}
                </span>
              </div>
            </div>
          </div>

          {/* Right: Status Badge */}
          <Badge
            variant="outline"
            className="self-start sm:self-auto bg-[hsl(152_60%_96%)] border-[hsl(152_50%_85%)] text-[hsl(152_60%_32%)] px-3 py-1 text-xs font-medium"
          >
            Complete
          </Badge>
        </div>
      </div>
    </div>
  );
};

export default ResultsHeader;
