import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MetadataDisplay } from "@/components/analytics";
import {
  ComparisonTab,
  CentralizedTab,
  FederatedTab,
  DetailsTab,
  MetricsTab,
  ChartsTab,
  ServerEvaluationTab,
} from "./tabs";
import type { ExperimentResults } from "@/types/api";

interface MetricData {
  name: string;
  value: number;
}

interface HistoryData {
  epoch: number;
  trainLoss?: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  valPrecision?: number;
  valRecall?: number;
  valF1?: number;
  valAuroc?: number;
}

interface ComparisonMetricsData {
  name: string;
  centralized: number;
  federated: number;
  difference: string;
}

interface ServerEvaluationChartData {
  round: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auroc?: number;
}

interface ServerEvaluationData {
  has_server_evaluation: boolean;
}

interface TabsOrchestratorProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  showComparison: boolean;
  trainingMode: string;
  centralizedResults: ExperimentResults | null;
  federatedResults: ExperimentResults | null;
  activeResults: ExperimentResults | null;
  metricsChartData: MetricData[];
  confusionMatrix: number[][] | null;
  trainingHistoryData: HistoryData[];
  comparisonMetricsData: ComparisonMetricsData[];
  centralizedMetricsData: MetricData[];
  centralizedHistoryData: HistoryData[];
  centralizedConfusionMatrix: number[][] | null;
  federatedMetricsData: MetricData[];
  federatedHistoryData: HistoryData[];
  federatedConfusionMatrix: number[][] | null;
  serverEvaluation: ServerEvaluationData | null;
  serverEvaluationChartData: ServerEvaluationChartData[];
  serverEvaluationLatestMetrics: MetricData[] | null;
  serverEvaluationConfusionMatrix: number[][] | null;
}

const TabsOrchestrator: React.FC<TabsOrchestratorProps> = ({
  activeTab,
  setActiveTab,
  showComparison,
  trainingMode,
  centralizedResults,
  federatedResults,
  activeResults,
  metricsChartData,
  confusionMatrix,
  trainingHistoryData,
  comparisonMetricsData,
  centralizedMetricsData,
  centralizedHistoryData,
  centralizedConfusionMatrix,
  federatedMetricsData,
  federatedHistoryData,
  federatedConfusionMatrix,
  serverEvaluation,
  serverEvaluationChartData,
  serverEvaluationLatestMetrics,
  serverEvaluationConfusionMatrix,
}) => {
  if (showComparison) {
    return (
      <Tabs
        defaultValue="comparison"
        value={activeTab}
        onValueChange={setActiveTab}
        className="space-y-6"
      >
        <TabsList className="grid grid-cols-4 w-full max-w-lg mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl">
          <TabsTrigger
            value="comparison"
            className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
          >
            Comparison
          </TabsTrigger>
          <TabsTrigger
            value="centralized"
            className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
          >
            Centralized
          </TabsTrigger>
          <TabsTrigger
            value="federated"
            className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
          >
            Federated
          </TabsTrigger>
          <TabsTrigger
            value="details"
            className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
          >
            Details
          </TabsTrigger>
        </TabsList>

        <TabsContent value="comparison" className="space-y-6">
          <ComparisonTab
            comparisonMetricsData={comparisonMetricsData}
            centralizedHistoryData={centralizedHistoryData}
            federatedHistoryData={federatedHistoryData}
          />
        </TabsContent>

        <TabsContent value="centralized" className="space-y-6">
          <CentralizedTab
            centralizedResults={centralizedResults}
            centralizedMetricsData={centralizedMetricsData}
            centralizedConfusionMatrix={centralizedConfusionMatrix}
          />
        </TabsContent>

        <TabsContent value="federated" className="space-y-6">
          <FederatedTab
            federatedResults={federatedResults}
            federatedMetricsData={federatedMetricsData}
            federatedConfusionMatrix={federatedConfusionMatrix}
          />
        </TabsContent>

        <TabsContent value="details" className="space-y-6">
          <DetailsTab
            centralizedResults={centralizedResults}
            federatedResults={federatedResults}
          />
        </TabsContent>
      </Tabs>
    );
  }

  const hasServerEvaluation =
    trainingMode === "federated" && serverEvaluation?.has_server_evaluation;
  const tabCount = hasServerEvaluation ? 4 : 3;

  return (
    <Tabs
      defaultValue="metrics"
      value={activeTab}
      onValueChange={setActiveTab}
      className="space-y-6"
    >
      <TabsList
        className={`grid w-full max-w-lg mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl grid-cols-${tabCount}`}
      >
        <TabsTrigger
          value="metrics"
          className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
        >
          {trainingMode === "federated" ? "Final Metrics" : "Metrics"}
        </TabsTrigger>
        <TabsTrigger
          value="charts"
          className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
        >
          Progress
        </TabsTrigger>
        {hasServerEvaluation && (
          <TabsTrigger
            value="server-evaluation"
            className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
          >
            Server Eval
          </TabsTrigger>
        )}
        <TabsTrigger
          value="metadata"
          className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]"
        >
          Metadata
        </TabsTrigger>
      </TabsList>

      <TabsContent value="metrics" className="space-y-6">
        <MetricsTab
          metricsChartData={metricsChartData}
          confusionMatrix={confusionMatrix}
          trainingMode={trainingMode}
        />
      </TabsContent>

      <TabsContent value="charts" className="space-y-6">
        <ChartsTab trainingHistoryData={trainingHistoryData} />
      </TabsContent>

      {hasServerEvaluation && (
        <TabsContent value="server-evaluation" className="space-y-6">
          <ServerEvaluationTab
            serverEvaluationLatestMetrics={serverEvaluationLatestMetrics}
            serverEvaluationChartData={serverEvaluationChartData}
            serverEvaluationConfusionMatrix={serverEvaluationConfusionMatrix}
          />
        </TabsContent>
      )}

      <TabsContent value="metadata">
        {activeResults && (
          <MetadataDisplay
            metadata={activeResults.metadata}
            title="Experiment Metadata"
          />
        )}
      </TabsContent>
    </Tabs>
  );
};

export default TabsOrchestrator;
