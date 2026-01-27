import { memo } from "react";
import { Users, Server } from "lucide-react";

interface TrainingModeBadgeProps {
  isFederated: boolean;
  compact?: boolean;
}

export const TrainingModeBadge = memo(({ isFederated, compact = false }: TrainingModeBadgeProps) => {
  const baseClasses = compact
    ? "inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full text-[8px] font-semibold"
    : "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-semibold";

  const variantClasses = isFederated
    ? "bg-gradient-to-r from-[hsl(210_60%_94%)] to-[hsl(210_60%_92%)] text-[hsl(210_60%_35%)] border border-[hsl(210_60%_85%)]"
    : "bg-gradient-to-r from-[hsl(152_50%_94%)] to-[hsl(152_50%_92%)] text-[hsl(152_60%_30%)] border border-[hsl(152_50%_85%)]";

  return (
    <div className={`${baseClasses} ${variantClasses} shadow-sm`}>
      {isFederated ? (
        <>
          <Users className={compact ? "h-2 w-2" : "h-2.5 w-2.5"} />
          <span>{compact ? "Fed" : "Federated"}</span>
        </>
      ) : (
        <>
          <Server className={compact ? "h-2 w-2" : "h-2.5 w-2.5"} />
          <span>{compact ? "Cen" : "Centralized"}</span>
        </>
      )}
    </div>
  );
});

TrainingModeBadge.displayName = "TrainingModeBadge";
