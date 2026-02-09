import React from "react";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { ExperimentConfiguration } from "@/types/experiment";
import { MetadataDisplay } from "@/components/analytics";
import { useResultsVisualization } from "@/hooks/useResultsVisualization";
import LoadingState from "./LoadingState";
import ErrorState from "./ErrorState";
import ResultsHeader from "./ResultsHeader";
import ResultsFooter from "./ResultsFooter";
import ComparisonTab from "./tabs/ComparisonTab";
import CentralizedTab from "./tabs/CentralizedTab";
import MetricsTab from "./tabs/MetricsTab";
import ChartsTab from "./tabs/ChartsTab";
import ServerEvaluationTab from "./tabs/ServerEvaluationTab";
import ClientMetricsTab from "./tabs/ClientMetricsTab";
import ComparisonTabsList from "./tabs/ComparisonTabsList";
import SingleModeTabsList from "./tabs/SingleModeTabsList";

interface ResultsVisualizationProps {
  config: ExperimentConfiguration;
  runId: number;
  onReset: () => void;
}

const ResultsVisualization = ({
  config,
  runId,
  onReset,
}: ResultsVisualizationProps) => {
  const {
    activeTab, setActiveTab, loading, error, centralizedResults, federatedResults,
    activeResults, showComparison, trainingHistoryData, confusionMatrix, metricsChartData,
    comparisonMetricsData, centralizedMetricsData, centralizedHistoryData,
    centralizedConfusionMatrix, federatedMetricsData, federatedHistoryData,
    federatedConfusionMatrix, serverEvaluation, serverEvaluationChartData,
    serverEvaluationLatestMetrics, serverEvaluationConfusionMatrix,
    clientMetrics, clientMetricsChartData, clientTrainingHistories, aggregatedRoundMetrics,
    handleDownload,
  } = useResultsVisualization({ config, runId });

  const hasClientMetrics = Boolean(
    clientMetrics?.is_federated && clientMetrics.num_clients > 0
  );

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} onReset={onReset} />;

  return (
    <div className="space-y-8" style={{ animation: "fadeIn 0.5s ease-out" }}>
      <div className="w-full max-w-5xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 overflow-hidden">
          <ResultsHeader config={config} runId={runId} showComparison={showComparison} />

          <div className="p-8">
            {showComparison ? (
              <Tabs defaultValue="comparison" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <ComparisonTabsList />

                <TabsContent value="comparison">
                  <ComparisonTab
                    comparisonMetricsData={comparisonMetricsData}
                    centralizedHistoryData={centralizedHistoryData}
                    federatedHistoryData={federatedHistoryData}
                  />
                </TabsContent>

                <TabsContent value="centralized">
                  <CentralizedTab
                    centralizedResults={centralizedResults}
                    centralizedMetricsData={centralizedMetricsData}
                    centralizedConfusionMatrix={centralizedConfusionMatrix}
                  />
                </TabsContent>

                <TabsContent value="federated">
                  <CentralizedTab
                    centralizedResults={federatedResults}
                    centralizedMetricsData={federatedMetricsData}
                    centralizedConfusionMatrix={federatedConfusionMatrix}
                  />
                </TabsContent>

                <TabsContent value="details" className="focus-visible:outline-none focus-visible:ring-0">
                  <div className="p-2 md:p-4 space-y-6">
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                      {centralizedResults && (
                        <MetadataDisplay metadata={centralizedResults.metadata} title="Centralized Experiment" />
                      )}
                      {federatedResults && (
                        <MetadataDisplay metadata={federatedResults.metadata} title="Federated Experiment" />
                      )}
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              <Tabs defaultValue="metrics" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <SingleModeTabsList
                  trainingMode={config.trainingMode}
                  hasServerEvaluation={serverEvaluation?.has_server_evaluation || false}
                  hasClientMetrics={hasClientMetrics}
                />

                <TabsContent value="metrics">
                  {activeResults && (
                    <MetricsTab
                      metricsChartData={metricsChartData}
                      confusionMatrix={confusionMatrix}
                      trainingMode={config.trainingMode}
                    />
                  )}
                </TabsContent>

                {config.trainingMode !== "federated" && (
                  <TabsContent value="charts">
                    {activeResults && <ChartsTab trainingHistoryData={trainingHistoryData} />}
                  </TabsContent>
                )}

                {hasClientMetrics && (
                  <TabsContent value="clients">
                    <ClientMetricsTab
                      clientMetricsChartData={clientMetricsChartData}
                      clientTrainingHistories={clientTrainingHistories}
                      aggregatedRoundMetrics={aggregatedRoundMetrics}
                      numClients={clientMetrics?.num_clients || 0}
                    />
                  </TabsContent>
                )}

                {serverEvaluation?.has_server_evaluation && (
                  <TabsContent value="server-evaluation">
                    <ServerEvaluationTab
                      serverEvaluationChartData={serverEvaluationChartData}
                      serverEvaluationLatestMetrics={serverEvaluationLatestMetrics}
                      serverEvaluationConfusionMatrix={serverEvaluationConfusionMatrix}
                    />
                  </TabsContent>
                )}

                <TabsContent value="metadata" className="focus-visible:outline-none focus-visible:ring-0">
                  {activeResults ? (
                    <div className="p-2 md:p-4">
                      <MetadataDisplay metadata={activeResults.metadata} title="Experiment Metadata" />
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-64 text-muted-foreground">
                      No metadata available
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </div>

          <ResultsFooter onReset={onReset} handleDownload={handleDownload} />
        </div>
      </div>
    </div>
  );
};

export default ResultsVisualization;
