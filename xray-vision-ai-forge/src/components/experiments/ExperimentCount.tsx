import { memo } from "react";
import { Activity } from "lucide-react";

interface ExperimentCountProps {
  count: number;
}

export const ExperimentCount = memo(({ count }: ExperimentCountProps) => (
  <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-[hsl(168_25%_96%)] to-[hsl(172_25%_94%)] border border-[hsl(168_20%_90%)] shadow-sm hover:shadow-md transition-all duration-300">
    <div className="flex items-center justify-center w-5 h-5 rounded-full bg-[hsl(172_63%_22%)]/10">
      <Activity className="h-3 w-3 text-[hsl(172_63%_35%)]" />
    </div>
    <span className="text-xs font-semibold text-[hsl(172_43%_20%)]">
      {count} experiment{count !== 1 ? "s" : ""}
    </span>
  </div>
));

ExperimentCount.displayName = "ExperimentCount";
