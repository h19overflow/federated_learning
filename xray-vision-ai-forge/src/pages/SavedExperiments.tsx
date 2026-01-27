import { memo, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ArrowLeft, BarChart, TrendingUp } from "lucide-react";
import { AnalyticsTab } from "@/components/analytics";
import {
  useExperiments,
  ViewModeToggle,
  ExperimentCount,
  ExperimentCard,
  viewModeGridCols,
  LoadingState,
  ErrorState,
  EmptyState,
} from "@/components/experiments";

const SavedExperiments = memo(() => {
  const navigate = useNavigate();
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);
  const { runs, loading, error, activeTab, setActiveTab, viewMode, setViewMode, refresh } = useExperiments();

  const viewRunResults = useCallback(
    (runId: number) => navigate("/experiment", { state: { viewRunId: runId, goToStep: 4 } }),
    [navigate]
  );

  const goHome = useCallback(() => navigate("/"), [navigate]);
  const createExperiment = useCallback(() => navigate("/experiment"), [navigate]);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {showWelcomeGuide && <WelcomeGuide onClose={() => setShowWelcomeGuide(false)} />}
      <Header onShowHelp={() => setShowWelcomeGuide(true)} />

      <main className="flex-1 overflow-y-auto bg-trust-gradient">
        <div className="max-w-6xl mx-auto px-6 py-6">
          {/* Page Header */}
          <div className="mb-6 animate-fade-in">
            <button
              onClick={goHome}
              className="inline-flex items-center gap-2 text-[hsl(215_15%_50%)] hover:text-[hsl(172_63%_28%)] transition-all duration-300 mb-3 group hover:gap-3"
            >
              <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
              <span className="text-sm font-medium">Back to Home</span>
            </button>

            <div className="flex items-end justify-between gap-4">
              <div className="flex-1">
                <h1 className="text-3xl font-semibold text-[hsl(172_43%_15%)] tracking-tight mb-1.5 bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_43%_15%)] bg-clip-text text-transparent">
                  Saved Experiments
                </h1>
                <p className="text-sm text-[hsl(215_15%_50%)] font-light">Review and analyze your previous training runs.</p>
              </div>

              <div className="flex items-center gap-3">
                {!loading && runs.length > 0 && <ViewModeToggle viewMode={viewMode} onChange={setViewMode} />}
                {!loading && runs.length > 0 && <ExperimentCount count={runs.length} />}
              </div>
            </div>
          </div>

          {/* Tabs Navigation */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full max-w-md mx-auto mb-4 grid-cols-2 bg-gradient-to-r from-[hsl(168,20%,95%)] to-[hsl(210,15%,94%)] p-1.5 rounded-2xl shadow-sm border border-[hsl(168_20%_90%)]/50 backdrop-blur-sm">
              <TabsTrigger
                value="experiments"
                className="rounded-xl data-[state=active]:bg-white data-[state=active]:shadow-md data-[state=active]:text-[hsl(172,43%,20%)] flex items-center gap-2 font-medium transition-all duration-300 data-[state=inactive]:text-[hsl(215_15%_55%)]"
              >
                <BarChart className="w-4 h-4" />
                Experiments
              </TabsTrigger>
              <TabsTrigger
                value="analytics"
                className="rounded-xl data-[state=active]:bg-white data-[state=active]:shadow-md data-[state=active]:text-[hsl(172,43%,20%)] flex items-center gap-2 font-medium transition-all duration-300 data-[state=inactive]:text-[hsl(215_15%_55%)]"
              >
                <TrendingUp className="w-4 h-4" />
                Analytics
              </TabsTrigger>
            </TabsList>

            {/* Experiments Tab Content */}
            <TabsContent value="experiments" className="mt-0">
              {loading && <LoadingState />}
              {error && !loading && <ErrorState message={error} onRetry={refresh} />}
              {!loading && !error && runs.length > 0 && (
                <div key={`grid-${viewMode}`} className={`gap-4 auto-rows-fr transition-all duration-500 grid ${viewModeGridCols[viewMode]}`}>
                  {runs.map((run, index) => (
                    <ExperimentCard key={run.id} run={run} viewMode={viewMode} index={index} onViewResults={viewRunResults} />
                  ))}
                </div>
              )}
              {!loading && !error && runs.length === 0 && <EmptyState onCreate={createExperiment} />}
            </TabsContent>

            {/* Analytics Tab Content */}
            <TabsContent value="analytics" className="mt-0">
              <AnalyticsTab />
            </TabsContent>
          </Tabs>
        </div>
      </main>

      <Footer />
    </div>
  );
});

export default SavedExperiments;
