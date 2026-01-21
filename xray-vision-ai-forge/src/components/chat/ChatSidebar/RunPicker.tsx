import React from "react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Loader2, BarChart, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { RunSummary } from "./types";

interface RunPickerProps {
  show: boolean;
  runs: RunSummary[];
  recentRuns: RunSummary[];
  federatedRuns: RunSummary[];
  centralizedRuns: RunSummary[];
  isLoading: boolean;
  onSelectRun: (run: RunSummary) => void;
  onClose: () => void;
}

const formatRunTime = (timeStr: string | null) => {
  if (!timeStr) return "Unknown";
  try {
    const date = new Date(timeStr);
    return date.toLocaleString();
  } catch {
    return timeStr;
  }
};

export const RunPicker: React.FC<RunPickerProps> = ({
  show,
  runs,
  recentRuns,
  federatedRuns,
  centralizedRuns,
  isLoading,
  onSelectRun,
  onClose,
}) => {
  if (!show) return null;

  return (
    <div className="border-b border-[hsl(210_15%_92%)] bg-[hsl(168_25%_98%)] max-h-80 overflow-hidden">
      <Command
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            onClose();
          }
        }}
        className="bg-transparent"
      >
        <div className="p-4 border-b border-[hsl(210_15%_92%)] bg-white sticky top-0">
          <CommandInput
            placeholder="Search runs by ID or training mode..."
            className="text-sm"
            autoFocus
          />
        </div>
        <CommandList
          className="max-h-72 overflow-y-auto p-2 scrollbar-none"
          role="listbox"
        >
          {isLoading ? (
            <div className="p-8 flex flex-col items-center justify-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin text-[hsl(172_63%_35%)]" />
              <p className="text-sm text-[hsl(215_15%_50%)]">Loading runs...</p>
            </div>
          ) : runs.length === 0 ? (
            <CommandEmpty className="p-8 text-center">
              <div className="flex flex-col items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-[hsl(210_15%_95%)] flex items-center justify-center">
                  <BarChart className="h-6 w-6 text-[hsl(215_15%_55%)]" />
                </div>
                <div>
                  <p className="text-sm font-medium text-[hsl(172_43%_20%)]">
                    No training runs found
                  </p>
                  <p className="text-xs text-[hsl(215_15%_55%)] mt-1">
                    Start a training run first
                  </p>
                </div>
              </div>
            </CommandEmpty>
          ) : (
            <>
              {/* Recent Runs */}
              {recentRuns.length > 0 && (
                <CommandGroup heading="Recent Runs">
                  {recentRuns.map((run) => (
                    <CommandItem
                      key={`recent-${run.id}`}
                      value={`run ${run.id} ${run.training_mode}`}
                      onSelect={() => onSelectRun(run)}
                      role="option"
                      className="flex items-center gap-3 p-3 cursor-pointer data-[selected='true']:bg-[hsl(172_40%_94%)] data-[selected='true']:text-[hsl(172_63%_22%)] rounded-xl transition-colors group"
                    >
                      <RunItem run={run} />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Federated Runs */}
              {federatedRuns.length > 0 && (
                <CommandGroup heading="Federated Runs">
                  {federatedRuns.map((run) => (
                    <CommandItem
                      key={`fed-${run.id}`}
                      value={`run ${run.id} federated`}
                      onSelect={() => onSelectRun(run)}
                      role="option"
                      className="flex items-center gap-3 p-3 cursor-pointer data-[selected='true']:bg-[hsl(172_40%_94%)] data-[selected='true']:text-[hsl(172_63%_22%)] rounded-xl transition-colors group"
                    >
                      <RunItem run={run} />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}

              {/* Centralized Runs */}
              {centralizedRuns.length > 0 && (
                <CommandGroup heading="Centralized Runs">
                  {centralizedRuns.map((run) => (
                    <CommandItem
                      key={`central-${run.id}`}
                      value={`run ${run.id} centralized`}
                      onSelect={() => onSelectRun(run)}
                      role="option"
                      className="flex items-center gap-3 p-3 cursor-pointer data-[selected='true']:bg-[hsl(172_40%_94%)] data-[selected='true']:text-[hsl(172_63%_22%)] rounded-xl transition-colors group"
                    >
                      <RunItem run={run} />
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </>
          )}
        </CommandList>
      </Command>
    </div>
  );
};

const RunItem: React.FC<{ run: RunSummary }> = ({ run }) => (
  <>
    <div
      className={cn(
        "h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors",
        run.training_mode === "federated"
          ? "bg-[hsl(210_60%_92%)] text-[hsl(210_60%_40%)]"
          : "bg-[hsl(152_50%_92%)] text-[hsl(152_60%_35%)]",
      )}
    >
      <BarChart className="h-5 w-5" />
    </div>
    <div className="flex-1 min-w-0">
      <p className="text-sm font-semibold">Run #{run.id}</p>
      <p className="text-xs opacity-70 truncate mt-0.5">
        {run.training_mode} • {formatRunTime(run.start_time)}
      </p>
      {run.best_val_recall > 0 && (
        <p className="text-xs text-[hsl(152_60%_35%)] font-medium mt-1">
          Best Recall: {(run.best_val_recall * 100).toFixed(2)}%
        </p>
      )}
    </div>
    <div className="opacity-0 group-data-[selected='true']:opacity-100 transition-opacity">
      <div className="w-8 h-8 rounded-lg bg-[hsl(172_63%_22%)] flex items-center justify-center">
        <Send className="h-4 w-4 text-white" />
      </div>
    </div>
  </>
);
