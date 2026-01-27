import { memo } from "react";
import { LayoutGrid, List, Minimize2 } from "lucide-react";
import { ViewMode } from "./useExperiments";

interface ViewModeToggleProps {
  viewMode: ViewMode;
  onChange: (mode: ViewMode) => void;
}

const buttonClasses = (isActive: boolean) =>
  `p-2 rounded-md transition-all duration-300 ${
    isActive
      ? "bg-white shadow-md text-[hsl(172_63%_22%)]"
      : "text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_22%)] hover:bg-white/50"
  }`;

export const ViewModeToggle = memo(({ viewMode, onChange }: ViewModeToggleProps) => (
  <div className="hidden sm:flex items-center gap-1 p-1 rounded-lg bg-gradient-to-r from-[hsl(168_25%_96%)] to-[hsl(172_25%_94%)] border border-[hsl(168_20%_90%)] shadow-sm">
    <button onClick={() => onChange("detailed")} className={buttonClasses(viewMode === "detailed")} title="Detailed view">
      <LayoutGrid className="h-4 w-4" />
    </button>
    <button onClick={() => onChange("concise")} className={buttonClasses(viewMode === "concise")} title="Concise view">
      <List className="h-4 w-4" />
    </button>
    <button onClick={() => onChange("compact")} className={buttonClasses(viewMode === "compact")} title="Compact view">
      <Minimize2 className="h-4 w-4" />
    </button>
  </div>
));

ViewModeToggle.displayName = "ViewModeToggle";
