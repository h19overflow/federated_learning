import React from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ExperimentConfiguration } from '@/types/experiment';
import { BarChart, LineChart, Line, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Download, Check, ArrowLeftRight, BarChart2, BarChart3, Loader2, AlertCircle, Activity, TrendingUp, Grid3X3, FileText, Sparkles, HelpCircle, ChevronDown } from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import MetadataDisplay from '@/components/MetadataDisplay';
import { useResultsVisualization } from '@/hooks/useResultsVisualization';
import api from '@/services/api';

interface ResultsVisualizationProps {
  config: ExperimentConfiguration;
  runId: number;
  onReset: () => void;
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

// Clinical Clarity Theme Chart Colors
const chartColors = {
  accuracy: 'hsl(172, 63%, 28%)',       // Primary teal
  precision: 'hsl(168, 40%, 45%)',      // Mint
  recall: 'hsl(35, 70%, 50%)',          // Amber
  f1Score: 'hsl(172, 45%, 35%)',        // Light teal
  auc: 'hsl(210, 60%, 50%)',            // Trust blue
  trainLoss: 'hsl(172, 63%, 35%)',      // Teal
  valLoss: 'hsl(172, 43%, 20%)',        // Dark teal
  trainAcc: 'hsl(168, 40%, 50%)',       // Mint green
  valAcc: 'hsl(172, 50%, 40%)',         // Medium teal
  trainF1: 'hsl(35, 60%, 50%)',         // Warm amber
  valPrecision: 'hsl(168, 35%, 45%)',   // Sage
  valRecall: 'hsl(35, 65%, 55%)',       // Light amber
  valF1: 'hsl(172, 40%, 40%)',          // Teal variant
  valAuroc: 'hsl(210, 55%, 50%)',       // Blue
  normal: 'hsl(172, 63%, 28%)',         // Primary
  pneumonia: 'hsl(172, 43%, 18%)',      // Dark
  centralized: 'hsl(172, 63%, 28%)',    // Teal
  federated: 'hsl(210, 60%, 50%)'       // Blue
};

/**
 * ResultsVisualization Component
 * Displays training results and metrics with charts and visualizations
 * Redesigned with Clinical Clarity theme
 */
const ResultsVisualization = ({
  config,
  runId,
  onReset,
  isFederatedTraining = false,
  federatedRounds = [],
  federatedContext,
}: ResultsVisualizationProps) => {
  const [localFederatedData, setLocalFederatedData] = React.useState<{
    isFederated: boolean;
    rounds: Array<{ round: number; metrics: Record<string, number> }>;
    context?: { numRounds: number; numClients: number };
  }>({
    isFederated: isFederatedTraining,
    rounds: federatedRounds,
    context: federatedContext,
  });

  React.useEffect(() => {
    const fetchFederatedData = async () => {
      if (federatedRounds.length > 0) return;
      if (config.trainingMode === 'federated') {
        try {
          const fedData = await api.results.getFederatedRounds(runId);
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

  const effectiveFederatedData = React.useMemo(() => {
    if (federatedRounds.length > 0) {
      return { isFederated: true, rounds: federatedRounds, context: federatedContext };
    }
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
    confusionMatrix,
    metricsChartData,
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
    handleDownload,
  } = useResultsVisualization({ config, runId });

  // Loading state
  if (loading) {
    return (
      <div className="space-y-8" style={{ animation: 'fadeIn 0.5s ease-out' }}>
        <div className="w-full max-w-4xl mx-auto">
          <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 p-12">
            <div className="flex flex-col items-center justify-center space-y-5">
              <div className="p-4 rounded-2xl bg-[hsl(172_40%_94%)]">
                <Loader2 className="h-10 w-10 text-[hsl(172_63%_28%)] animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-xl font-semibold text-[hsl(172_43%_20%)]">Loading Results</p>
                <p className="text-[hsl(215_15%_50%)] mt-1">Analyzing experiment data...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-8" style={{ animation: 'fadeIn 0.5s ease-out' }}>
        <div className="w-full max-w-4xl mx-auto">
          <div className="bg-white rounded-[2rem] border border-[hsl(0_50%_90%)] shadow-lg p-12">
            <div className="flex flex-col items-center justify-center space-y-5">
              <div className="p-4 rounded-2xl bg-[hsl(0_60%_95%)]">
                <AlertCircle className="h-10 w-10 text-[hsl(0_72%_51%)]" />
              </div>
              <div className="text-center">
                <p className="text-xl font-semibold text-[hsl(0_72%_40%)]">Failed to Load Results</p>
                <p className="text-[hsl(215_15%_50%)] mt-1">{error}</p>
              </div>
              <Button
                onClick={onReset}
                variant="outline"
                className="rounded-xl border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
              >
                Start New Experiment
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Metric explanations for educational tooltips
  const metricExplanations: Record<string, { description: string; relevance: string }> = {
    'F1-Score': {
      description: 'The harmonic mean of precision and recall (2 × (precision × recall) / (precision + recall)). It balances both false positives and false negatives into a single metric.',
      relevance: 'For pneumonia detection, F1-Score ensures the model doesn\'t sacrifice recall (catching pneumonia cases) for precision (avoiding false alarms) or vice versa. A high F1-Score (>0.85) indicates reliable clinical performance with balanced error rates.'
    },
    'F1 Score': {
      description: 'The harmonic mean of precision and recall (2 × (precision × recall) / (precision + recall)). It balances both false positives and false negatives into a single metric.',
      relevance: 'For pneumonia detection, F1 Score ensures the model doesn\'t sacrifice recall (catching pneumonia cases) for precision (avoiding false alarms) or vice versa. A high F1 Score (>0.85) indicates reliable clinical performance with balanced error rates.'
    },
    'AUC': {
      description: 'Area Under the ROC Curve - measures the model\'s ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).',
      relevance: 'In medical imaging, AUC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it\'s robust to class imbalance, making it the gold standard for clinical validation and FDA approval.'
    },
    'AUC-ROC': {
      description: 'Area Under the ROC Curve - measures the model\'s ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).',
      relevance: 'In medical imaging, AUC-ROC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it\'s robust to class imbalance, making it the gold standard for clinical validation and FDA approval.'
    },
    'AUROC': {
      description: 'Area Under the ROC Curve - measures the model\'s ability to distinguish between pneumonia and normal X-rays across all possible classification thresholds (0.0 = worst, 1.0 = perfect).',
      relevance: 'In medical imaging, AUROC (>0.90 is excellent, >0.95 is outstanding) shows how well the model separates normal X-rays from pneumonia cases. Unlike accuracy, it\'s robust to class imbalance, making it the gold standard for clinical validation and FDA approval.'
    },
    'Precision': {
      description: 'Proportion of positive predictions that are actually correct (TP / (TP + FP)).',
      relevance: 'High precision means fewer false alarms - when the model flags pneumonia, it\'s usually right. This reduces unnecessary follow-up tests and patient anxiety.'
    },
    'Recall': {
      description: 'Proportion of actual positive cases that are correctly identified (TP / (TP + FN)).',
      relevance: 'High recall is critical in medical diagnosis - it means the model catches most pneumonia cases. Missing a pneumonia case (low recall) could delay life-saving treatment.'
    },
    'Accuracy': {
      description: 'Overall proportion of correct predictions ((TP + TN) / Total).',
      relevance: 'While accuracy gives a general sense of performance, it can be misleading with imbalanced datasets. Use F1 Score and AUC-ROC for better clinical assessment.'
    }
  };

  // Enhanced Metric Card Component with tooltips
  const MetricCard = ({ name, value, index = 0, total = 5 }: { name: string; value: number; index?: number; total?: number }) => {
    const [showTooltip, setShowTooltip] = React.useState(false);
    const explanation = metricExplanations[name];

    // Determine tooltip position to prevent overflow
    const isFirst = index === 0;
    const isLast = index === total - 1;
    const tooltipPositionClass = isFirst
      ? 'left-0'
      : isLast
      ? 'right-0'
      : 'left-1/2 -translate-x-1/2';

    const arrowPositionClass = isFirst
      ? 'left-8'
      : isLast
      ? 'right-8'
      : 'left-1/2 -translate-x-1/2';

    return (
      <div
        className="bg-white p-5 rounded-xl border border-[hsl(210_15%_92%)] shadow-sm hover:shadow-md transition-shadow relative group"
        onMouseEnter={() => explanation && setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide">{name}</p>
          {explanation && (
            <HelpCircle className="h-3 w-3 text-[hsl(172_63%_35%)] cursor-help" />
          )}
        </div>
        <p className="text-3xl font-bold text-[hsl(172_63%_25%)]">{(value * 100).toFixed(1)}%</p>

        {/* Educational Tooltip - positioned to avoid overflow */}
        {explanation && showTooltip && (
          <div className={`absolute z-50 w-72 p-4 bg-white border-2 border-[hsl(172_40%_85%)] rounded-xl shadow-2xl bottom-full mb-2 ${tooltipPositionClass} pointer-events-none`}>
            <div className="space-y-3">
              <div>
                <p className="text-xs font-bold text-[hsl(172_63%_28%)] mb-1.5 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  What is {name}?
                </p>
                <p className="text-xs text-[hsl(215_15%_40%)] leading-relaxed">{explanation.description}</p>
              </div>
              <div className="pt-2 border-t border-[hsl(210_15%_92%)]">
                <p className="text-xs font-bold text-[hsl(172_63%_28%)] mb-1.5 flex items-center gap-1">
                  <Activity className="h-3 w-3" />
                  Clinical Relevance
                </p>
                <p className="text-xs text-[hsl(215_15%_40%)] leading-relaxed">{explanation.relevance}</p>
              </div>
            </div>
            {/* Tooltip arrow pointing down */}
            <div className={`absolute top-full ${arrowPositionClass} border-8 border-transparent border-t-white`} style={{ marginTop: '-2px' }}></div>
            <div className={`absolute top-full ${arrowPositionClass} border-[9px] border-transparent border-t-[hsl(172_40%_85%)]`}></div>
          </div>
        )}
      </div>
    );
  };

  // Confusion Matrix Component with educational content
  const ConfusionMatrixDisplay = ({ matrix, title }: { matrix: number[][]; title?: string }) => {
    const [isOpen, setIsOpen] = React.useState(false);

    return (
      <div className="bg-[hsl(168_25%_98%)] rounded-xl p-5 border border-[hsl(168_20%_92%)]">
        {title && (
          <h4 className="text-sm font-semibold text-[hsl(172_43%_20%)] mb-4 flex items-center gap-2">
            <Grid3X3 className="h-4 w-4 text-[hsl(172_63%_35%)]" />
            {title}
          </h4>
        )}

        {/* Matrix Grid */}
        <div className="grid grid-cols-2 gap-2 max-w-xs mx-auto mb-4">
          <div className="bg-[hsl(172_50%_95%)] p-3 rounded-lg text-center">
            <p className="text-xs text-[hsl(215_15%_55%)] mb-1">True Negative</p>
            <p className="text-2xl font-bold text-[hsl(172_63%_28%)]">{matrix[0][0]}</p>
          </div>
          <div className="bg-[hsl(0_60%_97%)] p-3 rounded-lg text-center">
            <p className="text-xs text-[hsl(215_15%_55%)] mb-1">False Positive</p>
            <p className="text-2xl font-bold text-[hsl(0_72%_51%)]">{matrix[0][1]}</p>
          </div>
          <div className="bg-[hsl(0_60%_97%)] p-3 rounded-lg text-center">
            <p className="text-xs text-[hsl(215_15%_55%)] mb-1">False Negative</p>
            <p className="text-2xl font-bold text-[hsl(0_72%_51%)]">{matrix[1][0]}</p>
          </div>
          <div className="bg-[hsl(172_50%_95%)] p-3 rounded-lg text-center">
            <p className="text-xs text-[hsl(215_15%_55%)] mb-1">True Positive</p>
            <p className="text-2xl font-bold text-[hsl(172_63%_28%)]">{matrix[1][1]}</p>
          </div>
        </div>

        {/* Educational Collapsible */}
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger asChild>
            <button className="flex items-center gap-2 text-sm text-[hsl(172_63%_30%)] hover:text-[hsl(172_63%_25%)] transition-colors w-full justify-center py-2 rounded-lg hover:bg-[hsl(172_40%_94%)]">
              <HelpCircle className="h-4 w-4" />
              <span>Understanding the Confusion Matrix</span>
              <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3">
            <div className="bg-white rounded-lg p-4 border border-[hsl(168_20%_90%)] space-y-3">
              <p className="text-sm text-[hsl(215_15%_40%)]">
                A confusion matrix shows how well the model classifies X-rays as <strong>Normal</strong> or <strong>Pneumonia</strong>.
              </p>

              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-sm bg-[hsl(172_50%_85%)] mt-1 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(172_43%_20%)]">True Negative (TN)</p>
                    <p className="text-xs text-[hsl(215_15%_50%)]">Correctly identified as Normal (no pneumonia)</p>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-sm bg-[hsl(172_50%_85%)] mt-1 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(172_43%_20%)]">True Positive (TP)</p>
                    <p className="text-xs text-[hsl(215_15%_50%)]">Correctly identified as Pneumonia</p>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-sm bg-[hsl(0_60%_90%)] mt-1 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(0_60%_40%)]">False Positive (FP)</p>
                    <p className="text-xs text-[hsl(215_15%_50%)]">Incorrectly flagged as Pneumonia (false alarm)</p>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <div className="w-3 h-3 rounded-sm bg-[hsl(0_60%_90%)] mt-1 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(0_60%_40%)]">False Negative (FN)</p>
                    <p className="text-xs text-[hsl(215_15%_50%)]">Missed Pneumonia case (most critical error)</p>
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
                <p className="text-xs text-[hsl(215_15%_45%)] italic">
                  💡 In medical diagnosis, minimizing False Negatives is critical—missing a pneumonia case could delay treatment.
                </p>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  // Main render
  return (
    <div className="space-y-8" style={{ animation: 'fadeIn 0.5s ease-out' }}>
      <div className="w-full max-w-5xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 overflow-hidden">
          {/* Header */}
          <div className="px-8 py-6 border-b border-[hsl(210_15%_94%)] bg-gradient-to-r from-[hsl(168_25%_98%)] to-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                  <TrendingUp className="h-6 w-6 text-[hsl(172_63%_28%)]" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                    {showComparison ? 'Training Methods Comparison' : 'Experiment Results'}
                  </h2>
                  <p className="text-[hsl(215_15%_50%)] mt-1">
                    {config.trainingMode === 'federated' ? 'Federated Learning' : 'Centralized Training'} - Run #{runId}
                  </p>
                </div>
              </div>
              <Badge className="bg-[hsl(172_50%_92%)] text-[hsl(172_63%_25%)] border-0 px-3 py-1">
                <Sparkles className="w-3 h-3 mr-2" />
                Complete
              </Badge>
            </div>
          </div>

          {/* Content */}
          <div className="p-8">
            {showComparison && comparisonData ? (
              <Tabs defaultValue="comparison" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <TabsList className="grid grid-cols-4 w-full max-w-lg mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl">
                  <TabsTrigger value="comparison" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Comparison
                  </TabsTrigger>
                  <TabsTrigger value="centralized" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Centralized
                  </TabsTrigger>
                  <TabsTrigger value="federated" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Federated
                  </TabsTrigger>
                  <TabsTrigger value="details" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Details
                  </TabsTrigger>
                </TabsList>

                {/* Comparison Tab Content */}
                <TabsContent value="comparison" className="space-y-6">
                  {/* Metrics Comparison Table */}
                  <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                    <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                      <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                        <BarChart2 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                      </div>
                      Performance Metrics Comparison
                    </h3>
                    <Table>
                      <TableHeader>
                        <TableRow className="hover:bg-transparent">
                          <TableHead className="text-[hsl(215_15%_45%)]">Metric</TableHead>
                          <TableHead className="text-[hsl(215_15%_45%)]">Centralized</TableHead>
                          <TableHead className="text-[hsl(215_15%_45%)]">Federated</TableHead>
                          <TableHead className="text-[hsl(215_15%_45%)]">Difference</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {comparisonMetricsData.map(metric => (
                          <TableRow key={metric.name} className="hover:bg-[hsl(168_25%_97%)]">
                            <TableCell className="font-medium text-[hsl(172_43%_20%)]">{metric.name}</TableCell>
                            <TableCell className="font-mono text-[hsl(172_63%_28%)]">{(metric.centralized * 100).toFixed(1)}%</TableCell>
                            <TableCell className="font-mono text-[hsl(210_60%_50%)]">{(metric.federated * 100).toFixed(1)}%</TableCell>
                            <TableCell className={`font-mono ${parseFloat(metric.difference) > 0 ? "text-[hsl(172_63%_28%)]" : "text-[hsl(0_72%_51%)]"}`}>
                              {parseFloat(metric.difference) > 0 ? "+" : ""}{(parseFloat(metric.difference) * 100).toFixed(1)}%
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  {/* Comparison Bar Chart */}
                  <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                    <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                      <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                        <BarChart3 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                      </div>
                      Side-by-Side Performance
                    </h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={comparisonMetricsData} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                          <XAxis dataKey="name" tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                          <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'white',
                              border: '1px solid hsl(168, 20%, 90%)',
                              borderRadius: '12px',
                              boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                            }}
                            formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                          />
                          <Legend />
                          <Bar name="Centralized" dataKey="centralized" fill={chartColors.centralized} radius={[4, 4, 0, 0]} />
                          <Bar name="Federated" dataKey="federated" fill={chartColors.federated} radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Training Progress Comparison */}
                  <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                    <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                      <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                        <ArrowLeftRight className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                      </div>
                      Training Progress Comparison
                    </h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                          <XAxis dataKey="epoch" allowDuplicatedCategory={false} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                          <YAxis domain={[0.5, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'white',
                              border: '1px solid hsl(168, 20%, 90%)',
                              borderRadius: '12px'
                            }}
                            formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                          />
                          <Legend />
                          <Line data={centralizedHistoryData} type="monotone" dataKey="valAcc" name="Centralized" stroke={chartColors.centralized} strokeWidth={2} dot={{ r: 4 }} />
                          <Line data={federatedHistoryData} type="monotone" dataKey="valAcc" name="Federated" stroke={chartColors.federated} strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </TabsContent>

                {/* Centralized Tab */}
                <TabsContent value="centralized" className="space-y-6">
                  {centralizedResults ? (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {centralizedMetricsData.map((metric, idx) => (
                          <MetricCard key={metric.name} name={metric.name} value={metric.value} index={idx} total={centralizedMetricsData.length} />
                        ))}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">Performance Metrics</h3>
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={centralizedMetricsData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                                <XAxis dataKey="name" tick={{ fill: 'hsl(215, 15%, 45%)', fontSize: 12 }} />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score" radius={[4, 4, 0, 0]}>
                                  {centralizedMetricsData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                        {centralizedConfusionMatrix && (
                          <ConfusionMatrixDisplay matrix={centralizedConfusionMatrix} title="Confusion Matrix" />
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-[hsl(215_15%_55%)]">No centralized results available</p>
                    </div>
                  )}
                </TabsContent>

                {/* Federated Tab */}
                <TabsContent value="federated" className="space-y-6">
                  {federatedResults ? (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {federatedMetricsData.map((metric, idx) => (
                          <MetricCard key={metric.name} name={metric.name} value={metric.value} index={idx} total={federatedMetricsData.length} />
                        ))}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">Performance Metrics</h3>
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={federatedMetricsData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                                <XAxis dataKey="name" tick={{ fill: 'hsl(215, 15%, 45%)', fontSize: 12 }} />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score" radius={[4, 4, 0, 0]}>
                                  {federatedMetricsData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                        {federatedConfusionMatrix && (
                          <ConfusionMatrixDisplay matrix={federatedConfusionMatrix} title="Confusion Matrix" />
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-[hsl(215_15%_55%)]">No federated results available</p>
                    </div>
                  )}
                </TabsContent>

                {/* Details Tab */}
                <TabsContent value="details" className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {centralizedResults && (
                      <MetadataDisplay metadata={centralizedResults.metadata} title="Centralized Experiment" />
                    )}
                    {federatedResults && (
                      <MetadataDisplay metadata={federatedResults.metadata} title="Federated Experiment" />
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              // Single mode results
              <Tabs defaultValue="metrics" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <TabsList className={`grid w-full max-w-lg mx-auto bg-[hsl(168_20%_95%)] p-1 rounded-xl ${config.trainingMode === 'federated' && serverEvaluation?.has_server_evaluation ? 'grid-cols-4' : 'grid-cols-3'}`}>
                  <TabsTrigger value="metrics" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    {config.trainingMode === 'federated' ? 'Final Metrics' : 'Metrics'}
                  </TabsTrigger>
                  <TabsTrigger value="charts" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Progress
                  </TabsTrigger>
                  {serverEvaluation?.has_server_evaluation && (
                    <TabsTrigger value="server-evaluation" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                      Server Eval
                    </TabsTrigger>
                  )}
                  <TabsTrigger value="metadata" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-[hsl(172_43%_20%)]">
                    Metadata
                  </TabsTrigger>
                </TabsList>

                {/* Metrics Tab */}
                <TabsContent value="metrics" className="space-y-6">
                  {activeResults && (
                    <>
                      {config.trainingMode === 'federated' && (
                        <div className="bg-[hsl(210_100%_97%)] rounded-xl p-4 border border-[hsl(210_60%_85%)] mb-6">
                          <div className="flex gap-3">
                            <HelpCircle className="h-5 w-5 text-[hsl(210_60%_45%)] flex-shrink-0 mt-0.5" />
                            <div>
                              <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">Final Metrics - Client Training Results</p>
                              <p className="text-sm text-[hsl(210_50%_35%)]">
                                These are the averaged performance metrics from all participating clients' local models at the final training round. This represents the client-side training performance before global model aggregation.
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {metricsChartData.map((metric, idx) => (
                          <MetricCard key={metric.name} name={metric.name} value={metric.value} index={idx} total={metricsChartData.length} />
                        ))}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                          <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                            <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                              <BarChart3 className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                            </div>
                            Performance Metrics
                          </h3>
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={metricsChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                                <XAxis dataKey="name" tick={{ fill: 'hsl(215, 15%, 45%)', fontSize: 12 }} />
                                <YAxis domain={[0, 1]} tickFormatter={tick => `${tick * 100}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                                <Bar dataKey="value" name="Score" radius={[4, 4, 0, 0]}>
                                  {metricsChartData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={Object.values(chartColors)[index]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                        {confusionMatrix && (
                          <ConfusionMatrixDisplay matrix={confusionMatrix} title="Confusion Matrix" />
                        )}
                      </div>
                    </>
                  )}
                </TabsContent>

                {/* Charts Tab */}
                <TabsContent value="charts" className="space-y-6">
                  {activeResults && (
                    <>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                          <h3 className="text-md font-semibold text-[hsl(172_43%_15%)] mb-4">Training & Validation Loss</h3>
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingHistoryData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                                <XAxis dataKey="epoch" tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <YAxis tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <Tooltip contentStyle={{ borderRadius: '12px' }} />
                                <Legend />
                                <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke={chartColors.trainLoss} strokeWidth={2} />
                                <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke={chartColors.valLoss} strokeWidth={2} strokeDasharray="5 5" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                        <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                          <h3 className="text-md font-semibold text-[hsl(172_43%_15%)] mb-4">Training & Validation Accuracy</h3>
                          <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingHistoryData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                                <XAxis dataKey="epoch" tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <YAxis domain={[0.5, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                                <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                                <Legend />
                                <Line type="monotone" dataKey="trainAcc" name="Train Acc" stroke={chartColors.trainAcc} strokeWidth={2} />
                                <Line type="monotone" dataKey="valAcc" name="Val Acc" stroke={chartColors.valAcc} strokeWidth={2} strokeDasharray="5 5" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </div>

                      {/* All Metrics Over Time */}
                      <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                        <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                          <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                            <Activity className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                          </div>
                          All Validation Metrics Over Time
                        </h3>
                        <div className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trainingHistoryData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                              <XAxis dataKey="epoch" tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                              <YAxis domain={[0, 1]} tickFormatter={tick => `${(tick * 100).toFixed(0)}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                              <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                              <Legend />
                              <Line type="monotone" dataKey="valAcc" name="Accuracy" stroke={chartColors.valAcc} strokeWidth={2} />
                              <Line type="monotone" dataKey="valPrecision" name="Precision" stroke={chartColors.valPrecision} strokeWidth={2} />
                              <Line type="monotone" dataKey="valRecall" name="Recall" stroke={chartColors.valRecall} strokeWidth={2} />
                              <Line type="monotone" dataKey="valF1" name="F1 Score" stroke={chartColors.valF1} strokeWidth={2} />
                              <Line type="monotone" dataKey="valAuroc" name="AUC-ROC" stroke={chartColors.valAuroc} strokeWidth={2} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </>
                  )}
                </TabsContent>

                {/* Server Evaluation Tab */}
                {serverEvaluation?.has_server_evaluation && (
                  <TabsContent value="server-evaluation" className="space-y-6">
                    <div className="bg-[hsl(210_100%_97%)] rounded-xl p-4 border border-[hsl(210_60%_85%)] mb-6">
                      <div className="flex gap-3">
                        <HelpCircle className="h-5 w-5 text-[hsl(210_60%_45%)] flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-sm font-semibold text-[hsl(210_70%_30%)] mb-1">Server Evaluation - Global Model Performance</p>
                          <p className="text-sm text-[hsl(210_50%_35%)]">
                            The aggregated global model is evaluated on a held-out centralized test set after each training round. This provides an objective measure of the global model's generalization performance independent of client-side variations.
                          </p>
                        </div>
                      </div>
                    </div>

                    {serverEvaluationLatestMetrics && (
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {serverEvaluationLatestMetrics.map((metric, idx) => (
                          <MetricCard key={metric.name} name={metric.name} value={metric.value} index={idx} total={serverEvaluationLatestMetrics.length} />
                        ))}
                      </div>
                    )}

                    <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
                      <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-4">Metrics Over Rounds</h3>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={serverEvaluationChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                            <XAxis dataKey="round" tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                            <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'hsl(215, 15%, 45%)' }} />
                            <Tooltip contentStyle={{ borderRadius: '12px' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke={chartColors.accuracy} strokeWidth={2} />
                            <Line type="monotone" dataKey="precision" name="Precision" stroke={chartColors.precision} strokeWidth={2} />
                            <Line type="monotone" dataKey="recall" name="Recall" stroke={chartColors.recall} strokeWidth={2} />
                            <Line type="monotone" dataKey="f1_score" name="F1 Score" stroke={chartColors.f1Score} strokeWidth={2} />
                            <Line type="monotone" dataKey="auroc" name="AUC-ROC" stroke={chartColors.auc} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {serverEvaluationConfusionMatrix && (
                      <>
                        <ConfusionMatrixDisplay matrix={serverEvaluationConfusionMatrix} title="Confusion Matrix (Latest Round)" />
                      </>
                    )}
                  </TabsContent>
                )}

                {/* Metadata Tab */}
                <TabsContent value="metadata">
                  {activeResults && (
                    <MetadataDisplay metadata={activeResults.metadata} title="Experiment Metadata" />
                  )}
                </TabsContent>
              </Tabs>
            )}

            {/* Download Section */}
            <div className="mt-8 pt-6 border-t border-[hsl(168_20%_92%)]">
              <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                  <Download className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                </div>
                Download Results
              </h3>
              <div className="flex flex-wrap gap-3">
                <Button
                  variant="outline"
                  onClick={() => handleDownload('csv')}
                  className="rounded-xl border-[hsl(172_30%_80%)] text-[hsl(172_43%_25%)] hover:bg-[hsl(168_25%_94%)]"
                >
                  <FileText className="mr-2 h-4 w-4" /> Export Training Metrics (CSV)
                </Button>
              </div>
              <p className="text-xs text-[hsl(215_15%_55%)] mt-2">
                Download epoch-by-epoch training and validation metrics for further analysis.
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="px-8 py-6 border-t border-[hsl(210_15%_94%)] bg-[hsl(168_25%_99%)]">
            <Button
              onClick={onReset}
              className="ml-auto flex bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-base px-8 py-6 rounded-xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5"
            >
              <Check className="mr-2 h-5 w-5" /> Start New Experiment
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsVisualization;
