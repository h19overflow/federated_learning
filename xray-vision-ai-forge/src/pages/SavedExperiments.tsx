
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowLeft, BarChart, Database, Loader, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import api from '@/services/api';

interface FederatedInfo {
  num_rounds: number;
  num_clients: number;
  has_server_evaluation: boolean;
  best_accuracy?: number;
  best_recall?: number;
  latest_round?: number;
  latest_accuracy?: number;
}

interface RunSummary {
  id: number;
  training_mode: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  best_val_recall: number;
  metrics_count: number;
  run_description: string | null;
  federated_info: FederatedInfo | null;
}

const SavedExperiments = () => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  // Fetch runs on mount
  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setLoading(true);
        const response = await api.results.listRuns();
        console.log('[SavedExperiments] Fetched runs:', response.runs);
        
        // Debug: Log federated runs specifically
        const federatedRuns = response.runs.filter((r: RunSummary) => r.training_mode === 'federated');
        console.log('[SavedExperiments] Federated runs:', federatedRuns);
        federatedRuns.forEach((run: RunSummary) => {
          console.log(`[SavedExperiments] Run ${run.id} federated_info:`, run.federated_info);
        });
        
        setRuns(response.runs);
      } catch (err: any) {
        console.error('Error fetching runs:', err);
        setError(err.message || 'Failed to load saved experiments');
        toast.error('Failed to load experiments');
      } finally {
        setLoading(false);
      }
    };

    fetchRuns();
  }, []);

  const viewRunResults = (runId: number) => {
    // Navigate to experiment page with state to show results
    navigate('/experiment', {
      state: {
        viewRunId: runId,
        goToStep: 4 // Go directly to results step
      }
    });
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header />

      <main className="flex-1 overflow-y-auto py-8">
        <div className="container px-4 md:px-6 max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-8 animate-fade-in">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2 transition-all duration-300 hover:bg-medical/5 hover:border-medical hover:-translate-x-1 group"
              onClick={() => navigate('/')}
            >
              <ArrowLeft size={16} className="transition-transform group-hover:-translate-x-1" />
              Back to Dashboard
            </Button>
            <h1 className="text-3xl font-bold text-medical-dark">Saved Experiments</h1>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader className="h-12 w-12 text-medical animate-spin mb-4" />
              <p className="text-muted-foreground">Loading experiments...</p>
            </div>
          )}

          {/* Error State */}
          {error && !loading && (
            <div className="flex flex-col items-center justify-center py-12">
              <AlertCircle className="h-12 w-12 text-status-error mb-4" />
              <p className="text-lg font-medium">Failed to Load Experiments</p>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button
                onClick={() => window.location.reload()}
                variant="outline"
                className="mt-4"
              >
                Retry
              </Button>
            </div>
          )}

          {/* Runs List */}
          {!loading && !error && runs.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {runs.map(run => {
                const isFederated = run.training_mode === 'federated';
                const federatedInfo = run.federated_info;
                
                return (
                  <Card key={run.id} className="result-card animate-fade-in hover:shadow-lg transition-all duration-300 hover:-translate-y-1 card-hover">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span>{run.run_description || `Run #${run.id}`}</span>
                        {isFederated && (
                          <span className="text-xs font-normal px-2 py-1 bg-medical/10 text-medical rounded">
                            Federated
                          </span>
                        )}
                      </CardTitle>
                      <CardDescription>
                        {run.start_time ? new Date(run.start_time).toLocaleString() : 'No date'}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium">Training Mode: </span>
                          <span className="capitalize">{run.training_mode || 'Unknown'}</span>
                        </div>

                        {/* Centralized Training Metrics */}
                        {!isFederated && run.best_val_recall > 0 && (
                          <div className="pt-2 border-t">
                            <div className="font-medium mb-1">Best Validation Recall:</div>
                            <div className="text-2xl font-bold text-medical">
                              {(run.best_val_recall * 100).toFixed(1)}%
                            </div>
                          </div>
                        )}

                        {/* Federated Training Metrics */}
                        {isFederated && federatedInfo && (
                          <div className="pt-2 border-t space-y-2">
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>
                                <span className="text-muted-foreground">Rounds:</span>
                                <span className="ml-1 font-medium">{federatedInfo.num_rounds}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Clients:</span>
                                <span className="ml-1 font-medium">{federatedInfo.num_clients}</span>
                              </div>
                            </div>
                            
                            {federatedInfo.has_server_evaluation && federatedInfo.best_recall !== null && federatedInfo.best_recall !== undefined && (
                              <div className="pt-2">
                                <div className="font-medium mb-1 text-xs">Best Server Recall:</div>
                                <div className="text-2xl font-bold text-medical">
                                  {(federatedInfo.best_recall * 100).toFixed(1)}%
                                </div>
                              </div>
                            )}

                            {!federatedInfo.has_server_evaluation && (
                              <div className="text-xs text-amber-600">
                                ⚠️ No server evaluation data available
                              </div>
                            )}
                          </div>
                        )}

                        <div className="text-xs text-muted-foreground pt-1">
                          {run.metrics_count} metrics collected
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-end">
                      <Button
                        size="sm"
                        onClick={() => viewRunResults(run.id)}
                        className="bg-medical hover:bg-medical-dark transition-all duration-300 hover:shadow-md hover:-translate-y-0.5 button-hover group"
                      >
                        <BarChart className="h-4 w-4 mr-1 transition-transform group-hover:scale-110" />
                        View Results
                      </Button>
                    </CardFooter>
                  </Card>
                );
              })}
            </div>
          )}

          {/* Empty State */}
          {!loading && !error && runs.length === 0 && (
            <div className="text-center py-12">
              <p className="text-lg text-muted-foreground">No saved experiments yet</p>
              <Button
                onClick={() => navigate('/')}
                className="mt-4 bg-medical hover:bg-medical-dark transition-all duration-300 hover:shadow-md button-hover"
              >
                Create New Experiment
              </Button>
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default SavedExperiments;
