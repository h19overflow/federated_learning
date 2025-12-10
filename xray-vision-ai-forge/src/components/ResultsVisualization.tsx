import React from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ExperimentConfiguration } from '@/types/experiment';
import { BarChart, LineChart, Line, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Download, Check, ArrowLeftRight, BarChart2, BarChart3, Loader, AlertCircle } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import MetadataDisplay from '@/components/MetadataDisplay';
import { useResultsVisualization } from '@/hooks/useResultsVisualization';
import api from '@/services/api';

interface ResultsVisualizationProps {
  config: ExperimentConfiguration;
  runId: number;  // Changed from experimentId - comes from WebSocket
  onReset: () => void;
  // Federated learning props (optional - only for federated training)
  isFederatedTraining?: boolean;
  federatedRounds?: Array<{
    round: number;
    metrics: Record<string, number>;
  }>;
  federatedContext?: {
    numRounds: number;
    numClients: number;
  };
}

// Chart Colors
const chartColors = {
  accuracy: '#0A9396',
  precision: '#94D2BD',
  recall: '#E9C46A',
  f1Score: '#F4A261',
  auc: '#E76F51',
  trainLoss: '#0A9396',
  valLoss: '#005F73',
  trainAcc: '#94D2BD',
  valAcc: '#52B788',
  trainF1: '#F4A261',
  valPrecision: '#94D2BD',
  valRecall: '#E9C46A',
  valF1: '#F4A261',
  valAuroc: '#E76F51',
  normal: '#0A9396',
  pneumonia: '#005F73',
  centralized: '#0A9396',
  federated: '#E76F51'
};

const ResultsVisualization = ({
  config,
  runId,
  onReset,
  isFederatedTraining = false,
  federatedRounds = [],
  federatedContext,
}: ResultsVisualizationProps) => {
  // Local state for federated data in case not provided via props
  const [localFederatedData, setLocalFederatedData] = React.useState<{
    isFederated: boolean;
    rounds: Array<{ round: number; metrics: Record<string, number> }>;
    context?: { numRounds: number; numClients: number };
  }>({
    isFederated: isFederatedTraining,
    rounds: federatedRounds,
    context: federatedContext,
  });

  // Fetch federated data if not provided via props
  React.useEffect(() => {
    const fetchFederatedData = async () => {
      // Skip if we already have data from props
      if (federatedRounds.length > 0) {
        return;
      }

      // Fetch federated rounds data if this is a federated training run
      if (config.trainingMode === 'federated') {
        try {
          const fedData = await api.results.getFederatedRounds(runId);
          console.log('[ResultsVisualization] Fetched federated rounds:', fedData);
          if (fedData.is_federated && fedData.rounds.length > 0) {
            setLocalFederatedData({
              isFederated: true,
              rounds: fedData.rounds,
              context: {
                numRounds: fedData.num_rounds,
                numClients: fedData.num_clients,
              },
            });
          }
        } catch (error) {
          console.error('[ResultsVisualization] Error fetching federated data:', error);
        }
      }
    };

    fetchFederatedData();
  }, [runId, config.trainingMode, federatedRounds.length]);

  // Use local state if props are not provided
  // Priority: props data > fetched data > default empty
  const effectiveFederatedData = React.useMemo(() => {
    if (federatedRounds.length > 0) {
      console.log('[ResultsVisualization] Using federated rounds from props:', federatedRounds.length);
      return { 
        isFederated: true, 
        rounds: federatedRounds, 
        context: federatedContext 
      };
    }
    console.log('[ResultsVisualization] Using local federated data:', localFederatedData.rounds.length);
    return localFederatedData;
  }, [federatedRounds, federatedContext, localFederatedData]);

  const {
    activeTab,
    setActiveTab,
    loading,
    error,
    centralizedResults,
    federatedResults,
    comparisonData,
    activeResults,
    showComparison,
    trainingHistoryData,
    confusionMatrixData,
    confusionMatrix,
    metricsChartData,
    metricsBarData,
    comparisonBarData,
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
    getConfusionMatrixColor,
    handleDownload,
  } = useResultsVisualization({ config, runId });

  // Debug logging
  React.useEffect(() => {
    console.log('[ResultsVisualization] Training mode:', config.trainingMode);
    console.log('[ResultsVisualization] Effective federated data:', effectiveFederatedData);
    console.log('[ResultsVisualization] Server evaluation:', serverEvaluation);
  }, [config.trainingMode, effectiveFederatedData, serverEvaluation]);

  // Loading state
  if (loading) {
    return (
      <div className="animate-fade-in">
        <Card className="w-full max-w-4xl mx-auto">
          <CardContent className="p-12">
            <div className="flex flex-col items-center justify-center space-y-4">
              <Loader className="h-12 w-12 text-medical animate-spin" />
              <p className="text-lg text-muted-foreground">Loading experiment results...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="animate-fade-in">
        <Card className="w-full max-w-4xl mx-auto">
          <CardContent className="p-12">
            <div className="flex flex-col items-center justify-center space-y-4">
              <AlertCircle className="h-12 w-12 text-status-error" />
              <p className="text-lg font-medium">Failed to Load Results</p>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button onClick={onReset} variant="outline">
                Start New Experiment
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Main render
  return (
    <div className="animate-fade-in">
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl text-medical-dark">
            {showComparison ? 'Comparison of Training Methods' : 'Experiment Results'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {showComparison && comparisonData ? (
            <Tabs defaultValue="comparison" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <TabsList className="grid grid-cols-4 w-full max-w-lg mx-auto">
                <TabsTrigger value="comparison">Comparison</TabsTrigger>
                <TabsTrigger value="centralized">Centralized</TabsTrigger>
                <TabsTrigger value="federated">Federated</TabsTrigger>
                <TabsTrigger value="details">Details</TabsTrigger>
              </TabsList>

              {/* Comparison Tab */}
              <TabsContent value="comparison" className="animate-fade-in space-y-6">
                <div className="space-y-8">
                  {/* Metrics Comparison Table */}
                  <Card>
                    <CardHeader className="py-3">
                      <h3 className="text-md font-medium flex items-center">
                        <BarChart2 className="mr-2 h-4 w-4" /> Performance Metrics Comparison
                      </h3>
                    </CardHeader>
                    <CardContent className="p-4">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Metric</TableHead>
                            <TableHead>Centralized</TableHead>
                            <TableHead>Federated</TableHead>
                            <TableHead>Difference</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {comparisonMetricsData.map(metric => (
                            <TableRow key={metric.name}>
                              <TableCell className="font-medium">{metric.name}</TableCell>
                              <TableCell>{(metric.centralized * 100).toFixed(1)}%</TableCell>
                              <TableCell>{(metric.federated * 100).toFixed(1)}%</TableCell>
                              <TableCell className={parseFloat(metric.difference) > 0 ? "text-green-600" : "text-red-600"}>
                                {parseFloat(metric.difference) > 0 ? "+" : ""}{(parseFloat(metric.difference) * 100).toFixed(1)}%
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>

                  {/* Comparison Bar Chart */}
                  <Card>
                    <CardHeader className="py-3">
                      <h3 className="text-md font-medium flex items-center text-base">
                        <BarChart3 className="mr-2 h-4 w-4" /> Side-by-Side Performance Comparison
                      </h3>
                    </CardHeader>
                    <CardContent className="p-4 px-[100px]">
                      <div className="h-80">
                        <ChartContainer config={{
                          centralized: { color: chartColors.centralized },
                          federated: { color: chartColors.federated }
                        }}>
                          <BarChart data={comparisonMetricsData} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} />
                            <ChartTooltip content={<ChartTooltipContent />} />
                            <Legend />
                            <Bar name="Centralized" dataKey="centralized" fill={chartColors.centralized} />
                            <Bar name="Federated" dataKey="federated" fill={chartColors.federated} />
                          </BarChart>
                        </ChartContainer>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Training History Comparison */}
                  <Card>
                    <CardHeader className="py-3">
                      <h3 className="text-md font-medium flex items-center">
                        <ArrowLeftRight className="mr-2 h-4 w-4" /> Training Progress Comparison
                      </h3>
                    </CardHeader>
                    <CardContent className="p-4">
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="epoch" allowDuplicatedCategory={false} />
                            <YAxis domain={[0.5, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} />
                            <Tooltip formatter={value => `${(Number(value) * 100).toFixed(1)}%`} />
                            <Legend />
                            <Line data={centralizedHistoryData} type="monotone" dataKey="valAcc" name="Centralized Accuracy" stroke={chartColors.centralized} activeDot={{ r: 8 }} />
                            <Line data={federatedHistoryData} type="monotone" dataKey="valAcc" name="Federated Accuracy" stroke={chartColors.federated} activeDot={{ r: 8 }} strokeDasharray="5 5" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Centralized Results Tab */}
              <TabsContent value="centralized" className="animate-fade-in space-y-6">
                {centralizedResults ? (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      {centralizedMetricsData.map((metric) => (
                        <Card key={metric.name} className="result-card">
                          <CardContent className="p-4 text-center">
                            <h4 className="text-sm font-medium text-muted-foreground mb-1">{metric.name}</h4>
                            <p className="text-3xl font-bold text-medical-dark">{(metric.value * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader className="py-3">
                          <h3 className="text-md font-medium">Performance Metrics</h3>
                        </CardHeader>
                        <CardContent className="p-4">
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={centralizedMetricsData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} />
                                <Tooltip formatter={value => `${(Number(value) * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score">
                                  {centralizedMetricsData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>

                      {centralizedConfusionMatrix && (
                        <Card>
                          <CardHeader className="py-3">
                            <h3 className="text-md font-medium">Confusion Matrix</h3>
                          </CardHeader>
                          <CardContent className="p-4">
                            <div className="h-64 flex items-center justify-center">
                              <div className="grid grid-cols-2 grid-rows-2 gap-1 w-full max-w-xs">
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{centralizedConfusionMatrix[0][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{centralizedConfusionMatrix[0][1]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{centralizedConfusionMatrix[1][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{centralizedConfusionMatrix[1][1]}</span>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-muted-foreground">No centralized results available</p>
                  </div>
                )}
              </TabsContent>

              {/* Federated Results Tab */}
              <TabsContent value="federated" className="animate-fade-in space-y-6">
                {federatedResults ? (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      {federatedMetricsData.map((metric) => (
                        <Card key={metric.name} className="result-card">
                          <CardContent className="p-4 text-center">
                            <h4 className="text-sm font-medium text-muted-foreground mb-1">{metric.name}</h4>
                            <p className="text-3xl font-bold text-medical-dark">{(metric.value * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader className="py-3">
                          <h3 className="text-md font-medium">Performance Metrics</h3>
                        </CardHeader>
                        <CardContent className="p-4">
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={federatedMetricsData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} />
                                <Tooltip formatter={value => `${(Number(value) * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score">
                                  {federatedMetricsData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>

                      {federatedConfusionMatrix && (
                        <Card>
                          <CardHeader className="py-3">
                            <h3 className="text-md font-medium">Confusion Matrix</h3>
                          </CardHeader>
                          <CardContent className="p-4">
                            <div className="h-64 flex items-center justify-center">
                              <div className="grid grid-cols-2 grid-rows-2 gap-1 w-full max-w-xs">
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{federatedConfusionMatrix[0][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{federatedConfusionMatrix[0][1]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{federatedConfusionMatrix[1][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{federatedConfusionMatrix[1][1]}</span>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-muted-foreground">No federated results available</p>
                  </div>
                )}
              </TabsContent>

              {/* Details Tab */}
              <TabsContent value="details" className="animate-fade-in">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {centralizedResults && (
                    <div>
                      <MetadataDisplay
                        metadata={centralizedResults.metadata}
                        title="Centralized Experiment Metadata"
                      />
                    </div>
                  )}

                  {federatedResults && (
                    <div>
                      <MetadataDisplay
                        metadata={federatedResults.metadata}
                        title="Federated Experiment Metadata"
                      />
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          ) : (
            <Tabs defaultValue="metrics" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <TabsList className={`grid w-full max-w-lg mx-auto ${config.trainingMode === 'federated' ? (serverEvaluation?.has_server_evaluation ? 'grid-cols-4' : 'grid-cols-3') : 'grid-cols-3'}`}>
                <TabsTrigger value="metrics">
                  {config.trainingMode === 'federated' ? 'Final Metrics' : 'Performance Metrics'}
                </TabsTrigger>
                <TabsTrigger value="charts">
                  {config.trainingMode === 'federated' ? 'Client Progress' : 'Training Progress'}
                </TabsTrigger>
                {serverEvaluation?.has_server_evaluation && (
                  <TabsTrigger value="server-evaluation">Server Evaluation</TabsTrigger>
                )}
                <TabsTrigger value="metadata">Metadata</TabsTrigger>
              </TabsList>

              {/* Performance Metrics Tab */}
              <TabsContent value="metrics" className="animate-fade-in space-y-6">
                {activeResults && (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      {metricsChartData.map((metric) => (
                        <Card key={metric.name} className="result-card">
                          <CardContent className="p-4 text-center">
                            <h4 className="text-sm font-medium text-muted-foreground mb-1">{metric.name}</h4>
                            <p className="text-3xl font-bold text-medical-dark">{(metric.value * 100).toFixed(1)}%</p>
                          </CardContent>
                        </Card>
                      ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader className="py-3">
                          <h3 className="text-md font-medium">Performance Metrics</h3>
                        </CardHeader>
                        <CardContent className="p-4">
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={metricsChartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} />
                                <Tooltip formatter={value => `${(Number(value) * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score">
                                  {metricsChartData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>

                      {confusionMatrix && (
                        <Card>
                          <CardHeader className="py-3">
                            <h3 className="text-md font-medium">Confusion Matrix</h3>
                          </CardHeader>
                          <CardContent className="p-4">
                            <div className="h-64 flex items-center justify-center">
                              <div className="grid grid-cols-2 grid-rows-2 gap-1 w-full max-w-xs">
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{confusionMatrix[0][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{confusionMatrix[0][1]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{confusionMatrix[1][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{confusionMatrix[1][1]}</span>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  </>
                )}
              </TabsContent>

              {/* Training Progress Charts Tab */}
              <TabsContent value="charts" className="animate-fade-in space-y-6">
                {activeResults && (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader className="py-3">
                          <h3 className="text-md font-medium">Training & Validation Loss</h3>
                        </CardHeader>
                        <CardContent className="p-4">
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingHistoryData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke={chartColors.trainLoss} activeDot={{ r: 8 }} />
                                <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke={chartColors.valLoss} strokeDasharray="5 5" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader className="py-3">
                          <h3 className="text-md font-medium">Training & Validation Accuracy</h3>
                        </CardHeader>
                        <CardContent className="p-4">
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingHistoryData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                                <YAxis domain={[0.5, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} />
                                <Tooltip formatter={value => `${(Number(value) * 100).toFixed(1)}%`} />
                                <Legend />
                                <Line type="monotone" dataKey="trainAcc" name="Training Accuracy" stroke={chartColors.trainAcc} activeDot={{ r: 8 }} />
                                <Line type="monotone" dataKey="valAcc" name="Validation Accuracy" stroke={chartColors.valAcc} strokeDasharray="5 5" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* All Validation Metrics Chart */}
                    <Card>
                      <CardHeader className="py-3">
                        <h3 className="text-md font-medium">All Validation Metrics Over Time</h3>
                      </CardHeader>
                      <CardContent className="p-4">
                        <div className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trainingHistoryData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                              <YAxis domain={[0, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} />
                              <Tooltip formatter={value => `${(Number(value) * 100).toFixed(2)}%`} />
                              <Legend />
                              <Line type="monotone" dataKey="valAcc" name="Accuracy" stroke={chartColors.valAcc} strokeWidth={2} activeDot={{ r: 6 }} />
                              <Line type="monotone" dataKey="valPrecision" name="Precision" stroke={chartColors.valPrecision} strokeWidth={2} activeDot={{ r: 6 }} />
                              <Line type="monotone" dataKey="valRecall" name="Recall" stroke={chartColors.valRecall} strokeWidth={2} activeDot={{ r: 6 }} />
                              <Line type="monotone" dataKey="valF1" name="F1 Score" stroke={chartColors.valF1} strokeWidth={2} activeDot={{ r: 6 }} />
                              <Line type="monotone" dataKey="valAuroc" name="AUC-ROC" stroke={chartColors.valAuroc} strokeWidth={2} activeDot={{ r: 6 }} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                )}
              </TabsContent>

              {/* Federated Rounds Tab - Client Aggregated Metrics */}
              {config.trainingMode === 'federated' && (
                <TabsContent value="federated-rounds" className="animate-fade-in space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">
                        📊 Client Aggregated Metrics (Federated Learning)
                        {effectiveFederatedData.context && (
                          <span className="text-sm font-normal text-muted-foreground ml-2">
                            ({effectiveFederatedData.context.numClients} clients × {effectiveFederatedData.context.numRounds} rounds)
                          </span>
                        )}
                      </CardTitle>
                      <p className="text-sm text-muted-foreground mt-2">
                        These metrics represent the aggregated performance from all clients after each federated round.
                        The global model is updated by combining client models using FedAvg strategy.
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {effectiveFederatedData.rounds.length === 0 ? (
                        <div className="text-center py-12">
                          <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                          <p className="text-lg font-medium">No Client Aggregation Data Available</p>
                          <p className="text-sm text-muted-foreground mt-2">
                            Client aggregated metrics will appear here after federated training completes.
                          </p>
                        </div>
                      ) : (
                        <>
                          {/* Line chart: Round # vs Metrics */}
                          <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={effectiveFederatedData.rounds}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="round" 
                              label={{ value: 'Round', position: 'insideBottom', offset: -5 }} 
                            />
                            <YAxis 
                              domain={[0, 1]} 
                              tickFormatter={v => `${(v * 100).toFixed(0)}%`} 
                            />
                            <Tooltip 
                              formatter={(value: number) => `${(value * 100).toFixed(1)}%`} 
                            />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="metrics.accuracy" 
                              name="Accuracy" 
                              stroke={chartColors.accuracy} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="metrics.precision" 
                              name="Precision" 
                              stroke={chartColors.precision} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="metrics.recall" 
                              name="Recall" 
                              stroke={chartColors.recall} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="metrics.f1" 
                              name="F1 Score" 
                              stroke={chartColors.f1Score} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="metrics.auroc" 
                              name="AUC-ROC" 
                              stroke={chartColors.auc} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Metrics table */}
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Round</TableHead>
                            <TableHead>Accuracy</TableHead>
                            <TableHead>Precision</TableHead>
                            <TableHead>Recall</TableHead>
                            <TableHead>F1 Score</TableHead>
                            <TableHead>AUC-ROC</TableHead>
                            <TableHead>Loss</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {effectiveFederatedData.rounds.map(row => (
                            <TableRow key={row.round}>
                              <TableCell className="font-medium">{row.round}</TableCell>
                              <TableCell>
                                {row.metrics.accuracy !== undefined
                                  ? `${(row.metrics.accuracy * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.metrics.precision !== undefined
                                  ? `${(row.metrics.precision * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.metrics.recall !== undefined
                                  ? `${(row.metrics.recall * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.metrics.f1 !== undefined
                                  ? `${(row.metrics.f1 * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.metrics.auroc !== undefined
                                  ? `${(row.metrics.auroc * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.metrics.loss !== undefined
                                  ? row.metrics.loss.toFixed(4)
                                  : 'N/A'}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              )}

              {/* Server Evaluation Tab */}
              {serverEvaluation?.has_server_evaluation && (
                <TabsContent value="server-evaluation" className="animate-fade-in space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">
                        🖥️ Server-Side Evaluation (Centralized Test Set)
                        <span className="text-sm font-normal text-muted-foreground ml-2">
                          Global model evaluated on server after each round
                        </span>
                      </CardTitle>
                      <p className="text-sm text-muted-foreground mt-2">
                        The server independently evaluates the global model on a held-out centralized test set after each round.
                        This provides an objective measure of the global model's performance on data not seen by clients.
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Latest Metrics Cards */}
                      {serverEvaluationLatestMetrics && (
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                          {serverEvaluationLatestMetrics.map((metric) => (
                            <Card key={metric.name} className="result-card">
                              <CardContent className="p-4 text-center">
                                <h4 className="text-sm font-medium text-muted-foreground mb-1">{metric.name}</h4>
                                <p className="text-3xl font-bold text-medical-dark">{(metric.value * 100).toFixed(1)}%</p>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      )}

                      {/* Metrics Over Rounds Chart */}
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={serverEvaluationChartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="round" 
                              label={{ value: 'Round', position: 'insideBottom', offset: -5 }} 
                            />
                            <YAxis 
                              domain={[0, 1]} 
                              tickFormatter={v => `${(v * 100).toFixed(0)}%`} 
                            />
                            <Tooltip 
                              formatter={(value: number) => `${(value * 100).toFixed(1)}%`} 
                            />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="accuracy" 
                              name="Accuracy" 
                              stroke={chartColors.accuracy} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="precision" 
                              name="Precision" 
                              stroke={chartColors.precision} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="recall" 
                              name="Recall" 
                              stroke={chartColors.recall} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="f1_score" 
                              name="F1 Score" 
                              stroke={chartColors.f1Score} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="auroc" 
                              name="AUC-ROC" 
                              stroke={chartColors.auc} 
                              strokeWidth={2}
                              activeDot={{ r: 6 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Confusion Matrix */}
                      {serverEvaluationConfusionMatrix && (
                        <Card>
                          <CardHeader className="py-3">
                            <h3 className="text-md font-medium">Confusion Matrix (Latest Round)</h3>
                          </CardHeader>
                          <CardContent className="p-4">
                            <div className="h-64 flex items-center justify-center">
                              <div className="grid grid-cols-2 grid-rows-2 gap-1 w-full max-w-xs">
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{serverEvaluationConfusionMatrix[0][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{serverEvaluationConfusionMatrix[0][1]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  False Negative<br />
                                  <span className="text-xl md:text-2xl font-bold text-status-error">{serverEvaluationConfusionMatrix[1][0]}</span>
                                </div>
                                <div className="bg-gray-100 p-2 text-center text-xs md:text-sm font-medium">
                                  True Positive<br />
                                  <span className="text-xl md:text-2xl font-bold text-medical">{serverEvaluationConfusionMatrix[1][1]}</span>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Metrics Table */}
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Round</TableHead>
                            <TableHead>Loss</TableHead>
                            <TableHead>Accuracy</TableHead>
                            <TableHead>Precision</TableHead>
                            <TableHead>Recall</TableHead>
                            <TableHead>F1 Score</TableHead>
                            <TableHead>AUC-ROC</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {serverEvaluationChartData.map(row => (
                            <TableRow key={row.round}>
                              <TableCell className="font-medium">{row.round}</TableCell>
                              <TableCell>{row.loss.toFixed(4)}</TableCell>
                              <TableCell>
                                {row.accuracy !== undefined && row.accuracy !== null
                                  ? `${(row.accuracy * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.precision !== undefined && row.precision !== null
                                  ? `${(row.precision * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.recall !== undefined && row.recall !== null
                                  ? `${(row.recall * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.f1_score !== undefined && row.f1_score !== null
                                  ? `${(row.f1_score * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                              <TableCell>
                                {row.auroc !== undefined && row.auroc !== null
                                  ? `${(row.auroc * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </TabsContent>
              )}

              {/* Metadata Tab */}
              <TabsContent value="metadata" className="animate-fade-in">
                {activeResults && (
                  <MetadataDisplay
                    metadata={activeResults.metadata}
                    title="Experiment Metadata"
                  />
                )}
              </TabsContent>
            </Tabs>
          )}

          {/* Download Section */}
          <div className="mt-8 border-t pt-6">
            <h3 className="text-lg font-medium mb-4">Download Results</h3>
            <div className="flex flex-wrap gap-4">
              <Button variant="outline" className="flex items-center" onClick={() => handleDownload('csv')}>
                <Download className="mr-2 h-4 w-4" /> Metrics CSV
              </Button>
              <Button variant="outline" className="flex items-center" onClick={() => handleDownload('json')}>
                <Download className="mr-2 h-4 w-4" /> Metrics JSON
              </Button>
              <Button variant="outline" className="flex items-center" onClick={() => handleDownload('summary')}>
                <Download className="mr-2 h-4 w-4" /> Summary Report
              </Button>
            </div>
          </div>

        </CardContent>
        <CardFooter>
          <Button onClick={onReset} className="ml-auto bg-medical hover:bg-medical-dark">
            <Check className="mr-2 h-4 w-4" /> Start New Experiment
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default ResultsVisualization;
