import React, { useState, useEffect, useCallback, memo } from "react";
import { useNavigate } from "react-router-dom";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  ArrowLeft,
  BarChart,
  Loader2,
  AlertCircle,
  Users,
  Server,
  ChevronRight,
  Calendar,
  Activity,
  TrendingUp,
  Zap,
  CheckCircle2,
  LayoutGrid,
  List,
  Minimize2,
} from "lucide-react";
import { toast } from "sonner";
import api from "@/services/api";
import { AnalyticsTab } from "@/components/analytics";
import { RunSummary } from "@/types/runs";

type ViewMode = "detailed" | "concise" | "compact";

const SavedExperiments = memo(() => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("experiments");
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    const saved = localStorage.getItem("experiments-view-mode");
    return (saved as ViewMode) || "detailed";
  });
  const navigate = useNavigate();

  // Persist view mode to localStorage
  useEffect(() => {
    console.log("View mode changed to:", viewMode);
    localStorage.setItem("experiments-view-mode", viewMode);
  }, [viewMode]);

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setLoading(true);
        const response = await api.results.listRuns();
        setRuns(response.runs);
      } catch (err) {
        console.error("Error fetching runs:", err);
        if (err instanceof Error) {
          setError(err.message || "Failed to load saved experiments");
        } else {
          setError("Failed to load saved experiments");
        }
        toast.error("Failed to load experiments");
      } finally {
        setLoading(false);
      }
    };

    fetchRuns();
  }, []);

   const viewRunResults = useCallback((runId: number) => {
     navigate("/experiment", {
       state: {
         viewRunId: runId,
         goToStep: 4,
       },
     });
   }, [navigate]);

   const formatDate = useCallback((dateString: string | null) => {
     if (!dateString) return "Unknown date";
     try {
       const date = new Date(dateString);
       return date.toLocaleDateString("en-US", {
         month: "short",
         day: "numeric",
         year: "numeric",
         hour: "2-digit",
         minute: "2-digit",
       });
     } catch {
       return dateString;
     }
   }, []);

    // Helper function: Check if centralized metrics should be displayed
    const shouldShowCentralizedMetrics = (run: RunSummary): boolean => {
      return run.best_val_recall > 0 || run.best_val_accuracy > 0;
    };

    // Helper function: Check if federated evaluation metrics should be displayed
    const shouldShowFederatedEvalMetrics = (federatedInfo?: RunSummary['federated_info']): boolean => {
      if (!federatedInfo?.has_server_evaluation) return false;
      return federatedInfo.best_accuracy !== null || federatedInfo.best_recall !== null;
    };

    // Helper function: Check if accuracy metric is valid
    const hasValidAccuracy = (value: number | null | undefined): boolean => {
      return value !== null && value !== undefined;
    };

    // Helper function: Check if recall metric is valid
    const hasValidRecall = (value: number | null | undefined): boolean => {
      return value !== null && value !== undefined;
    };

     // Helper function: Check if no metrics are available
     const hasNoMetrics = (run: RunSummary): boolean => {
       return run.best_val_recall === 0 && run.best_val_accuracy === 0;
     };

     // Metric indicator component with gradient background
     const MetricIndicator = memo(({
       label,
       value,
       unit = "%",
       variant = "primary",
     }: {
       label: string;
       value: number;
       unit?: string;
       variant?: "primary" | "secondary" | "accent";
     }) => {
       const variants = {
         primary: "from-[hsl(172_63%_22%)]/5 to-[hsl(172_63%_22%)]/0 border-[hsl(172_63%_22%)]/20",
         secondary: "from-[hsl(210_60%_40%)]/5 to-[hsl(210_60%_40%)]/0 border-[hsl(210_60%_40%)]/20",
         accent: "from-[hsl(152_60%_35%)]/5 to-[hsl(152_60%_35%)]/0 border-[hsl(152_60%_35%)]/20",
       };

        return (
          <div className={`bg-gradient-to-br ${variants[variant]} rounded-lg p-2 border backdrop-blur-sm transition-all duration-300 hover:shadow-md`}>
            <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5">
              {label}
            </p>
            <div className="flex items-baseline gap-0.5">
              <p className="text-base font-semibold text-[hsl(172_43%_20%)]">
                {value.toFixed(1)}
              </p>
              <p className="text-[10px] text-[hsl(215_15%_50%)]">{unit}</p>
            </div>
          </div>
        );
     });

     const SummaryStatisticsPreview = memo(({
       stats,
     }: {
       stats?: RunSummary['final_epoch_stats'];
     }) => {
       if (!stats) return null;

        return (
          <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
            <p className="text-[10px] text-[hsl(215_15%_55%)] mb-1.5 uppercase tracking-wide font-medium">
              Final Epoch Statistics
            </p>
            <div className="grid grid-cols-2 gap-1.5 text-xs">
              <MetricIndicator label="Recall" value={stats.sensitivity * 100} />
              <MetricIndicator label="Specificity" value={stats.specificity * 100} variant="secondary" />
              <MetricIndicator label="Precision" value={stats.precision_cm * 100} variant="accent" />
              <MetricIndicator label="F1 Score" value={stats.f1_cm * 100} />
            </div>
          </div>
        );
     });

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {showWelcomeGuide && (
        <WelcomeGuide onClose={() => setShowWelcomeGuide(false)} />
      )}

      <Header onShowHelp={() => setShowWelcomeGuide(true)} />

       <main className="flex-1 overflow-y-auto bg-trust-gradient">
         <div className="max-w-6xl mx-auto px-6 py-6">
            {/* Page Header */}
            <div className="mb-6 animate-fade-in">
              <button
                onClick={() => navigate("/")}
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
                  <p className="text-sm text-[hsl(215_15%_50%)] font-light">
                    Review and analyze your previous training runs.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  {/* View Mode Toggle */}
                  {!loading && runs.length > 0 && (
                    <div className="hidden sm:flex items-center gap-1 p-1 rounded-lg bg-gradient-to-r from-[hsl(168_25%_96%)] to-[hsl(172_25%_94%)] border border-[hsl(168_20%_90%)] shadow-sm">
                      <button
                        onClick={() => setViewMode("detailed")}
                        className={`p-2 rounded-md transition-all duration-300 ${
                          viewMode === "detailed"
                            ? "bg-white shadow-md text-[hsl(172_63%_22%)]"
                            : "text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_22%)] hover:bg-white/50"
                        }`}
                        title="Detailed view"
                      >
                        <LayoutGrid className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => setViewMode("concise")}
                        className={`p-2 rounded-md transition-all duration-300 ${
                          viewMode === "concise"
                            ? "bg-white shadow-md text-[hsl(172_63%_22%)]"
                            : "text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_22%)] hover:bg-white/50"
                        }`}
                        title="Concise view"
                      >
                        <List className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => setViewMode("compact")}
                        className={`p-2 rounded-md transition-all duration-300 ${
                          viewMode === "compact"
                            ? "bg-white shadow-md text-[hsl(172_63%_22%)]"
                            : "text-[hsl(215_15%_55%)] hover:text-[hsl(172_63%_22%)] hover:bg-white/50"
                        }`}
                        title="Compact view"
                      >
                        <Minimize2 className="h-4 w-4" />
                      </button>
                    </div>
                  )}

                  {!loading && runs.length > 0 && (
                    <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-[hsl(168_25%_96%)] to-[hsl(172_25%_94%)] border border-[hsl(168_20%_90%)] shadow-sm hover:shadow-md transition-all duration-300">
                      <div className="flex items-center justify-center w-5 h-5 rounded-full bg-[hsl(172_63%_22%)]/10">
                        <Activity className="h-3 w-3 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="text-xs font-semibold text-[hsl(172_43%_20%)]">
                        {runs.length} experiment{runs.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>

           {/* Tabs Navigation */}
           <Tabs
             value={activeTab}
             onValueChange={setActiveTab}
             className="w-full"
           >
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
               {/* Loading State */}
               {loading && (
                 <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
                   <div className="relative w-20 h-20 mb-8">
                     <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-[hsl(172_63%_22%)]/10 to-[hsl(172_63%_22%)]/5 blur-xl" />
                     <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(172_40%_94%)] to-[hsl(168_25%_96%)] flex items-center justify-center border border-[hsl(172_40%_88%)] shadow-lg">
                       <Loader2 className="h-9 w-9 text-[hsl(172_63%_35%)] animate-spin" />
                     </div>
                   </div>
                   <p className="text-lg font-semibold text-[hsl(172_43%_20%)] mb-2">
                     Loading experiments
                   </p>
                   <p className="text-sm text-[hsl(215_15%_55%)]">
                     Fetching your training runs...
                   </p>
                 </div>
               )}

               {/* Error State */}
               {error && !loading && (
                 <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
                   <div className="relative w-20 h-20 mb-8">
                     <div className="absolute inset-0 rounded-3xl bg-[hsl(0_72%_51%)]/10 blur-xl" />
                     <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(0_60%_95%)] to-[hsl(0_50%_93%)] flex items-center justify-center border border-[hsl(0_60%_85%)] shadow-lg">
                       <AlertCircle className="h-9 w-9 text-[hsl(0_72%_51%)]" />
                     </div>
                   </div>
                   <p className="text-lg font-semibold text-[hsl(172_43%_15%)] mb-2">
                     Failed to Load Experiments
                   </p>
                   <p className="text-sm text-[hsl(215_15%_50%)] mb-8 max-w-md text-center leading-relaxed">
                     {error}
                   </p>
                   <Button
                     onClick={() => window.location.reload()}
                     className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-xl px-6 py-2.5 shadow-lg shadow-[hsl(172_63%_22%)]/25 hover:shadow-xl transition-all duration-300 font-medium"
                   >
                     Try Again
                   </Button>
                 </div>
               )}

                {/* Experiments Grid */}
                {!loading && !error && runs.length > 0 && (
                  <div 
                    key={`grid-${viewMode}`}
                    className={`gap-4 auto-rows-fr transition-all duration-500 ${
                      viewMode === "detailed" 
                        ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3" 
                        : viewMode === "concise"
                        ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
                        : "grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5"
                    }`}>
                    {runs.map((run, index) => {
                      const isFederated = run.training_mode === "federated";
                      const federatedInfo = run.federated_info;
                      const bestAccuracy = isFederated 
                        ? federatedInfo?.best_accuracy ?? 0 
                        : run.best_val_accuracy;

                      return (
                        <div
                          key={`${run.id}-${viewMode}`}
                          className={`group relative bg-white rounded-xl border border-[hsl(210_15%_92%)] hover:border-[hsl(172_63%_22%)]/30 hover:shadow-2xl hover:shadow-[hsl(172_40%_85%)]/20 transition-all duration-500 hover:-translate-y-1 flex flex-col h-full overflow-hidden ${
                            viewMode === "detailed" ? "p-3" : viewMode === "concise" ? "p-2.5" : "p-2"
                          }`}
                          style={{
                            animation: "fadeIn 0.4s ease-out forwards, scaleIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) forwards",
                            animationDelay: `${index * 0.04}s`,
                            opacity: 0,
                            transformOrigin: "center",
                          }}
                        >
                          {/* Gradient accent bar */}
                          <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${
                            isFederated
                              ? "from-[hsl(210_60%_40%)] via-[hsl(210_60%_40%)]/50 to-transparent"
                              : "from-[hsl(152_60%_35%)] via-[hsl(152_60%_35%)]/50 to-transparent"
                          }`} />

                          {/* DETAILED MODE */}
                          {viewMode === "detailed" && (
                            <>
                              {/* Card Header */}
                              <div className="flex items-start justify-between mb-3">
                                <div className="flex-1 min-w-0">
                                  <h3 className="font-semibold text-[hsl(172_43%_15%)] text-base truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300">
                                    {run.run_description || `Run #${run.id}`}
                                  </h3>
                                  <div className="flex items-center gap-1.5 mt-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300">
                                    <Calendar className="h-3 w-3 flex-shrink-0" />
                                    <span className="text-[10px]">
                                      {formatDate(run.start_time)}
                                    </span>
                                  </div>
                                </div>

                                {/* Training Mode Badge */}
                                <div
                                  className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-semibold whitespace-nowrap transition-all duration-300 ${
                                    isFederated
                                      ? "bg-gradient-to-r from-[hsl(210_60%_94%)] to-[hsl(210_60%_92%)] text-[hsl(210_60%_35%)] border border-[hsl(210_60%_85%)] shadow-sm"
                                      : "bg-gradient-to-r from-[hsl(152_50%_94%)] to-[hsl(152_50%_92%)] text-[hsl(152_60%_30%)] border border-[hsl(152_50%_85%)] shadow-sm"
                                  } group-hover:shadow-md`}
                                >
                                  {isFederated ? (
                                    <>
                                      <Users className="h-3 w-3" />
                                      <span>Federated</span>
                                    </>
                                  ) : (
                                    <>
                                      <Server className="h-3 w-3" />
                                      <span>Centralized</span>
                                    </>
                                  )}
                                </div>
                              </div>

                               {/* Metrics Section */}
                               <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_15%_97%)] rounded-lg p-2.5 mb-2.5 border border-[hsl(168_20%_94%)] flex-grow transition-all duration-300 group-hover:border-[hsl(168_20%_88%)] animate-content-fade">
                                {/* Centralized Metrics */}
                                {!isFederated && shouldShowCentralizedMetrics(run) && (
                                  <div className="space-y-2">
                                    <div className="grid grid-cols-2 gap-2">
                                      <MetricIndicator label="Best Val Accuracy" value={run.best_val_accuracy * 100} />
                                      <MetricIndicator label="Best Val Recall" value={run.best_val_recall * 100} variant="secondary" />
                                    </div>
                                    <SummaryStatisticsPreview stats={run.final_epoch_stats} />
                                  </div>
                                )}

                                {/* Federated Metrics */}
                                {isFederated && federatedInfo && (
                                  <div className="space-y-2">
                                    <div className="grid grid-cols-2 gap-2">
                                      <div className="bg-gradient-to-br from-[hsl(210_60%_40%)]/5 to-[hsl(210_60%_40%)]/0 rounded-lg p-2 border border-[hsl(210_60%_40%)]/20 backdrop-blur-sm transition-all duration-300 hover:shadow-md">
                                        <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">
                                          Rounds
                                        </p>
                                        <div className="flex items-baseline gap-0.5">
                                          <p className="text-base font-semibold text-[hsl(172_43%_20%)]">
                                            {federatedInfo.num_rounds}
                                          </p>
                                          <Zap className="h-3 w-3 text-[hsl(210_60%_40%)]" />
                                        </div>
                                      </div>
                                      <div className="bg-gradient-to-br from-[hsl(152_60%_35%)]/5 to-[hsl(152_60%_35%)]/0 rounded-lg p-2 border border-[hsl(152_60%_35%)]/20 backdrop-blur-sm transition-all duration-300 hover:shadow-md">
                                        <p className="text-[9px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">
                                          Clients
                                        </p>
                                        <div className="flex items-baseline gap-0.5">
                                          <p className="text-base font-semibold text-[hsl(172_43%_20%)]">
                                            {federatedInfo.num_clients}
                                          </p>
                                          <Users className="h-3 w-3 text-[hsl(152_60%_35%)]" />
                                        </div>
                                      </div>
                                    </div>

                                    {shouldShowFederatedEvalMetrics(federatedInfo) && (
                                      <div className="pt-2 border-t border-[hsl(168_20%_92%)]">
                                        <p className="text-[10px] text-[hsl(215_15%_55%)] mb-1.5 uppercase tracking-wide font-medium">
                                          Server Evaluation
                                        </p>
                                        <div className="grid grid-cols-2 gap-2">
                                          {hasValidAccuracy(federatedInfo.best_accuracy) && (
                                            <MetricIndicator label="Best Val Accuracy" value={(federatedInfo.best_accuracy ?? 0) * 100} />
                                          )}
                                          {hasValidRecall(federatedInfo.best_recall) && (
                                            <MetricIndicator label="Best Val Recall" value={(federatedInfo.best_recall ?? 0) * 100} variant="secondary" />
                                          )}
                                        </div>
                                      </div>
                                    )}

                                    {!federatedInfo.has_server_evaluation && (
                                      <div className="flex items-center gap-1.5 pt-2 border-t border-[hsl(168_20%_92%)]">
                                        <div className="w-1 h-1 rounded-full bg-[hsl(35_70%_50%)]" />
                                        <p className="text-[10px] text-[hsl(35_70%_45%)] font-medium">
                                          No server evaluation data
                                        </p>
                                      </div>
                                    )}
                                    <SummaryStatisticsPreview stats={run.final_epoch_stats} />
                                  </div>
                                )}

                                {!isFederated && hasNoMetrics(run) && (
                                  <div className="text-center py-2">
                                    <p className="text-sm text-[hsl(215_15%_55%)]">
                                      No metrics available
                                    </p>
                                  </div>
                                )}
                              </div>

                              {/* Footer */}
                              <div className="flex items-center justify-between mt-auto pt-2.5 border-t border-[hsl(210_15%_92%)] group-hover:border-[hsl(172_63%_22%)]/20 transition-colors duration-300">
                                <div className="flex items-center gap-1">
                                  <CheckCircle2 className="h-3 w-3 text-[hsl(152_60%_35%)]" />
                                  <span className="text-[10px] text-[hsl(215_15%_55%)] font-medium">
                                    {run.metrics_count} metrics
                                  </span>
                                </div>

                                <Button
                                  onClick={() => viewRunResults(run.id)}
                                  className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-3 py-1.5 text-xs shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.03] font-medium flex items-center gap-1"
                                >
                                  <BarChart className="h-3.5 w-3.5" />
                                  View Results
                                  <ChevronRight className="h-3.5 w-3.5 opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                                </Button>
                              </div>
                            </>
                          )}

                          {/* CONCISE MODE */}
                          {viewMode === "concise" && (
                            <>
                              <div className="flex-1">
                                <h3 className="font-semibold text-[hsl(172_43%_15%)] text-sm truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300 mb-1.5">
                                  {run.run_description || `Run #${run.id}`}
                                </h3>
                                <div className="flex items-center gap-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300 mb-2">
                                  <Calendar className="h-3 w-3 flex-shrink-0" />
                                  <span className="text-[9px]">
                                    {formatDate(run.start_time)}
                                  </span>
                                </div>

                                {/* Training Mode Badge */}
                                <div
                                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-semibold whitespace-nowrap transition-all duration-300 mb-2 ${
                                    isFederated
                                      ? "bg-gradient-to-r from-[hsl(210_60%_94%)] to-[hsl(210_60%_92%)] text-[hsl(210_60%_35%)] border border-[hsl(210_60%_85%)] shadow-sm"
                                      : "bg-gradient-to-r from-[hsl(152_50%_94%)] to-[hsl(152_50%_92%)] text-[hsl(152_60%_30%)] border border-[hsl(152_50%_85%)] shadow-sm"
                                  }`}
                                >
                                  {isFederated ? (
                                    <>
                                      <Users className="h-2.5 w-2.5" />
                                      <span>Federated</span>
                                    </>
                                  ) : (
                                    <>
                                      <Server className="h-2.5 w-2.5" />
                                      <span>Centralized</span>
                                    </>
                                  )}
                                </div>

                                {/* Key Metric */}
                                <div className="bg-gradient-to-br from-[hsl(168_25%_98%)] to-[hsl(210_15%_97%)] rounded-lg p-2 border border-[hsl(168_20%_94%)] transition-all duration-300 group-hover:border-[hsl(168_20%_88%)]">
                                  <p className="text-[8px] uppercase tracking-wide text-[hsl(215_15%_55%)] mb-0.5 font-medium">
                                    Best Accuracy
                                  </p>
                                  <div className="flex items-baseline gap-0.5">
                                    <p className="text-sm font-semibold text-[hsl(172_43%_20%)]">
                                      {(bestAccuracy * 100).toFixed(1)}
                                    </p>
                                    <p className="text-[9px] text-[hsl(215_15%_50%)]">%</p>
                                  </div>
                                </div>
                              </div>

                              {/* Footer */}
                              <Button
                                onClick={() => viewRunResults(run.id)}
                                className="w-full bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-2 py-1.5 text-xs shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.02] font-medium mt-2"
                              >
                                View Results
                              </Button>
                            </>
                          )}

                          {/* COMPACT MODE */}
                          {viewMode === "compact" && (
                            <>
                              <div className="flex-1 min-w-0">
                                <h3 className="font-semibold text-[hsl(172_43%_15%)] text-xs truncate group-hover:text-[hsl(172_63%_22%)] transition-colors duration-300 mb-1">
                                  {run.run_description || `Run #${run.id}`}
                                </h3>
                                <div className="flex items-center gap-1 text-[hsl(215_15%_55%)] group-hover:text-[hsl(215_15%_50%)] transition-colors duration-300 mb-1.5">
                                  <Calendar className="h-2.5 w-2.5 flex-shrink-0" />
                                  <span className="text-[8px]">
                                    {formatDate(run.start_time).split(",")[0]}
                                  </span>
                                </div>

                                {/* Training Mode Badge */}
                                <div
                                  className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full text-[8px] font-semibold whitespace-nowrap transition-all duration-300 ${
                                    isFederated
                                      ? "bg-gradient-to-r from-[hsl(210_60%_94%)] to-[hsl(210_60%_92%)] text-[hsl(210_60%_35%)] border border-[hsl(210_60%_85%)]"
                                      : "bg-gradient-to-r from-[hsl(152_50%_94%)] to-[hsl(152_50%_92%)] text-[hsl(152_60%_30%)] border border-[hsl(152_50%_85%)]"
                                  }`}
                                >
                                  {isFederated ? (
                                    <>
                                      <Users className="h-2 w-2" />
                                      <span>Fed</span>
                                    </>
                                  ) : (
                                    <>
                                      <Server className="h-2 w-2" />
                                      <span>Cen</span>
                                    </>
                                  )}
                                </div>
                              </div>

                              {/* Footer */}
                              <Button
                                onClick={() => viewRunResults(run.id)}
                                className="w-full bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-lg px-2 py-1 text-[10px] shadow-md shadow-[hsl(172_63%_22%)]/15 hover:shadow-lg hover:shadow-[hsl(172_63%_22%)]/25 transition-all duration-300 group-hover:scale-[1.02] font-medium mt-2"
                              >
                                View
                              </Button>
                            </>
                          )}
                       </div>
                     );
                   })}
                 </div>
               )}

               {/* Empty State */}
               {!loading && !error && runs.length === 0 && (
                 <div className="flex flex-col items-center justify-center py-32 animate-fade-in">
                   <div className="relative w-24 h-24 mb-8">
                     <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-[hsl(172_63%_22%)]/10 to-[hsl(168_25%_96%)]/50 blur-xl" />
                     <div className="relative w-full h-full rounded-3xl bg-gradient-to-br from-[hsl(168_25%_96%)] to-[hsl(210_15%_95%)] flex items-center justify-center border border-[hsl(168_20%_90%)] shadow-lg">
                       <svg
                         className="w-12 h-12 text-[hsl(172_63%_35%)]"
                         viewBox="0 0 40 40"
                         fill="none"
                       >
                         <rect
                           x="6"
                           y="10"
                           width="28"
                           height="24"
                           rx="3"
                           stroke="currentColor"
                           strokeWidth="2"
                         />
                         <path
                           d="M6 16h28"
                           stroke="currentColor"
                           strokeWidth="2"
                         />
                         <circle cx="11" cy="13" r="1.5" fill="currentColor" />
                         <circle cx="16" cy="13" r="1.5" fill="currentColor" />
                         <path
                           d="M14 26l5-5 4 4 7-7"
                           stroke="currentColor"
                           strokeWidth="2"
                           strokeLinecap="round"
                           strokeLinejoin="round"
                         />
                       </svg>
                     </div>
                   </div>
                   <h3 className="text-2xl font-semibold text-[hsl(172_43%_15%)] mb-3">
                     No experiments yet
                   </h3>
                   <p className="text-[hsl(215_15%_50%)] mb-10 max-w-sm text-center leading-relaxed">
                     Start your first training run to see it appear here. Create a new experiment to begin.
                   </p>
                   <Button
                     onClick={() => navigate("/experiment")}
                     className="bg-gradient-to-r from-[hsl(172_63%_22%)] to-[hsl(172_63%_18%)] hover:from-[hsl(172_63%_18%)] hover:to-[hsl(172_63%_14%)] text-white rounded-xl px-8 py-3 text-base shadow-lg shadow-[hsl(172_63%_22%)]/25 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/35 transition-all duration-300 hover:-translate-y-1 font-semibold flex items-center gap-2"
                   >
                     <Zap className="h-5 w-5" />
                     Create New Experiment
                   </Button>
                 </div>
               )}
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
