import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { ArrowLeft, BarChart, Loader2, AlertCircle, Users, Server, ChevronRight, Calendar, Activity } from 'lucide-react';
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

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setLoading(true);
        const response = await api.results.listRuns();
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
    navigate('/experiment', {
      state: {
        viewRunId: runId,
        goToStep: 4
      }
    });
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Unknown date';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header />

      <main className="flex-1 overflow-y-auto bg-trust-gradient">
        <div className="max-w-6xl mx-auto px-6 py-12">
          {/* Page Header */}
          <div className="mb-12">
            <button
              onClick={() => navigate('/')}
              className="inline-flex items-center gap-2 text-[hsl(215_15%_50%)] hover:text-[hsl(172_63%_28%)] transition-colors mb-6 group"
            >
              <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
              <span className="text-sm font-medium">Back to Home</span>
            </button>

            <div className="flex items-end justify-between">
              <div>
                <h1 className="text-4xl font-semibold text-[hsl(172_43%_15%)] tracking-tight mb-2">
                  Saved Experiments
                </h1>
                <p className="text-lg text-[hsl(215_15%_50%)]">
                  Review and analyze your previous training runs.
                </p>
              </div>

              {!loading && runs.length > 0 && (
                <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-[hsl(168_25%_96%)] border border-[hsl(168_20%_90%)]">
                  <Activity className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                  <span className="text-sm font-medium text-[hsl(172_43%_20%)]">
                    {runs.length} experiment{runs.length !== 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="w-16 h-16 rounded-2xl bg-[hsl(172_40%_94%)] flex items-center justify-center mb-6">
                <Loader2 className="h-8 w-8 text-[hsl(172_63%_35%)] animate-spin" />
              </div>
              <p className="text-lg font-medium text-[hsl(172_43%_20%)]">Loading experiments</p>
              <p className="text-sm text-[hsl(215_15%_55%)] mt-1">Please wait...</p>
            </div>
          )}

          {/* Error State */}
          {error && !loading && (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="w-16 h-16 rounded-2xl bg-[hsl(0_60%_95%)] flex items-center justify-center mb-6">
                <AlertCircle className="h-8 w-8 text-[hsl(0_72%_51%)]" />
              </div>
              <p className="text-lg font-medium text-[hsl(172_43%_15%)] mb-2">Failed to Load Experiments</p>
              <p className="text-sm text-[hsl(215_15%_50%)] mb-6 max-w-md text-center">{error}</p>
              <Button
                onClick={() => window.location.reload()}
                className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl px-6 py-2.5 shadow-md shadow-[hsl(172_63%_22%)]/20"
              >
                Try Again
              </Button>
            </div>
          )}

          {/* Experiments Grid */}
          {!loading && !error && runs.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {runs.map((run, index) => {
                const isFederated = run.training_mode === 'federated';
                const federatedInfo = run.federated_info;

                return (
                  <div
                    key={run.id}
                    className="group relative bg-white rounded-[1.5rem] border border-[hsl(210_15%_92%)] p-6 hover:shadow-xl hover:shadow-[hsl(172_40%_85%)]/25 transition-all duration-500 hover:-translate-y-1"
                    style={{
                      animation: 'fadeIn 0.4s ease-out forwards',
                      animationDelay: `${index * 0.05}s`,
                      opacity: 0
                    }}
                  >
                    {/* Card Header */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-[hsl(172_43%_15%)] text-lg truncate">
                          {run.run_description || `Run #${run.id}`}
                        </h3>
                        <div className="flex items-center gap-2 mt-1.5 text-[hsl(215_15%_55%)]">
                          <Calendar className="h-3.5 w-3.5" />
                          <span className="text-xs">{formatDate(run.start_time)}</span>
                        </div>
                      </div>

                      {/* Training Mode Badge */}
                      <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                        isFederated
                          ? 'bg-[hsl(210_60%_94%)] text-[hsl(210_60%_40%)]'
                          : 'bg-[hsl(152_50%_94%)] text-[hsl(152_60%_35%)]'
                      }`}>
                        {isFederated ? (
                          <>
                            <Users className="h-3.5 w-3.5" />
                            <span>Federated</span>
                          </>
                        ) : (
                          <>
                            <Server className="h-3.5 w-3.5" />
                            <span>Centralized</span>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Metrics Section */}
                    <div className="bg-[hsl(168_25%_98%)] rounded-xl p-4 mb-4 border border-[hsl(168_20%_94%)]">
                      {/* Centralized Metrics */}
                      {!isFederated && run.best_val_recall > 0 && (
                        <div>
                          <p className="text-xs text-[hsl(215_15%_55%)] mb-1">Best Validation Recall</p>
                          <p className="text-3xl font-semibold text-[hsl(172_63%_28%)]">
                            {(run.best_val_recall * 100).toFixed(1)}
                            <span className="text-lg text-[hsl(172_63%_40%)]">%</span>
                          </p>
                        </div>
                      )}

                      {/* Federated Metrics */}
                      {isFederated && federatedInfo && (
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-3">
                            <div className="bg-white rounded-lg p-2.5 border border-[hsl(210_15%_92%)]">
                              <p className="text-[10px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5">Rounds</p>
                              <p className="text-lg font-semibold text-[hsl(172_43%_20%)]">{federatedInfo.num_rounds}</p>
                            </div>
                            <div className="bg-white rounded-lg p-2.5 border border-[hsl(210_15%_92%)]">
                              <p className="text-[10px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5">Clients</p>
                              <p className="text-lg font-semibold text-[hsl(172_43%_20%)]">{federatedInfo.num_clients}</p>
                            </div>
                          </div>

                          {federatedInfo.has_server_evaluation && federatedInfo.best_recall !== null && federatedInfo.best_recall !== undefined && (
                            <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
                              <p className="text-xs text-[hsl(215_15%_55%)] mb-1">Best Server Recall</p>
                              <p className="text-2xl font-semibold text-[hsl(172_63%_28%)]">
                                {(federatedInfo.best_recall * 100).toFixed(1)}
                                <span className="text-sm text-[hsl(172_63%_40%)]">%</span>
                              </p>
                            </div>
                          )}

                          {!federatedInfo.has_server_evaluation && (
                            <div className="flex items-center gap-2 pt-2 border-t border-[hsl(168_20%_92%)]">
                              <div className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)]" />
                              <p className="text-xs text-[hsl(35_70%_45%)]">No server evaluation data</p>
                            </div>
                          )}
                        </div>
                      )}

                      {/* No metrics fallback */}
                      {!isFederated && run.best_val_recall === 0 && (
                        <div className="text-center py-2">
                          <p className="text-sm text-[hsl(215_15%_55%)]">No metrics available</p>
                        </div>
                      )}
                    </div>

                    {/* Footer */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[hsl(215_15%_55%)]">
                        {run.metrics_count} metrics collected
                      </span>

                      <Button
                        onClick={() => viewRunResults(run.id)}
                        className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl px-4 py-2 text-sm shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.02]"
                      >
                        <BarChart className="h-4 w-4 mr-2" />
                        View Results
                        <ChevronRight className="h-4 w-4 ml-1 opacity-50 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Empty State */}
          {!loading && !error && runs.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="w-20 h-20 rounded-3xl bg-[hsl(168_25%_96%)] flex items-center justify-center mb-6">
                <svg className="w-10 h-10 text-[hsl(172_63%_35%)]" viewBox="0 0 40 40" fill="none">
                  <rect x="6" y="10" width="28" height="24" rx="3" stroke="currentColor" strokeWidth="2" />
                  <path d="M6 16h28" stroke="currentColor" strokeWidth="2" />
                  <circle cx="11" cy="13" r="1.5" fill="currentColor" />
                  <circle cx="16" cy="13" r="1.5" fill="currentColor" />
                  <path d="M14 26l5-5 4 4 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-[hsl(172_43%_15%)] mb-2">No experiments yet</h3>
              <p className="text-[hsl(215_15%_50%)] mb-8 max-w-sm text-center">
                Start your first training run to see it appear here.
              </p>
              <Button
                onClick={() => navigate('/experiment')}
                className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white rounded-xl px-8 py-3 text-base shadow-lg shadow-[hsl(172_63%_22%)]/20 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 transition-all duration-300 hover:-translate-y-0.5"
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
