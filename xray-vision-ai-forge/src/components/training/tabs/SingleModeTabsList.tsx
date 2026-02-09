import React from "react";
import { TabsList, TabsTrigger } from "@/components/ui/tabs";

interface SingleModeTabsListProps {
  trainingMode: string;
  hasServerEvaluation: boolean;
  hasClientMetrics?: boolean;
}

const SingleModeTabsList: React.FC<SingleModeTabsListProps> = ({
  trainingMode,
  hasServerEvaluation,
  hasClientMetrics = false,
}) => {
  // Calculate number of tabs dynamically
  let tabCount = 3; // metrics, charts, metadata
  if (hasServerEvaluation) tabCount++;
  if (hasClientMetrics) tabCount++;
  // Hide Progress tab in federated mode
  if (trainingMode === "federated") tabCount--;

  const gridColsMap: Record<number, string> = {
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-4",
    5: "grid-cols-5",
  };
  const gridCols = gridColsMap[tabCount] || "grid-cols-5";

  return (
    <TabsList className={`grid w-full max-w-2xl mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl ${gridCols}`}>
      <TabsTrigger value="metrics" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        {trainingMode === "federated" ? "Final Metrics" : "Metrics"}
      </TabsTrigger>
      {trainingMode !== "federated" && (
        <TabsTrigger value="charts" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
          Progress
        </TabsTrigger>
      )}
      {hasClientMetrics && (
        <TabsTrigger value="clients" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
          Clients
        </TabsTrigger>
      )}
      {hasServerEvaluation && (
        <TabsTrigger value="server-evaluation" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
          Server Eval
        </TabsTrigger>
      )}
      <TabsTrigger value="metadata" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
        Metadata
      </TabsTrigger>
    </TabsList>
  );
};

export default SingleModeTabsList;
