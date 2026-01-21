import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Header, Footer, WelcomeGuide } from "@/components/layout";
import {
  ProgressIndicator,
  StepContent,
  DatasetUpload,
  ExperimentConfig,
  TrainingExecution,
  ResultsVisualization,
} from "@/components/training";
import { Button } from "@/components/ui/button";
import { SavedExperiment, ExperimentConfiguration } from "@/types/experiment";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import api from "@/services/api";

// Define the workflow steps
const steps = [
  {
    id: 1,
    name: "Dataset",
    description: "Upload and configure dataset",
  },
  {
    id: 2,
    name: "Configuration",
    description: "Configure experiment parameters",
  },
  {
    id: 3,
    name: "Training",
    description: "Run the experiment",
  },
  {
    id: 4,
    name: "Results",
    description: "View and analyze results",
  },
];

// Define the application state interfaces
interface DatasetState {
  file: File | null;
  trainSplit: number;
}

const Index = () => {
  // Current step in the workflow
  const [currentStep, setCurrentStep] = useState(1);
  const location = useLocation();
  const navigate = useNavigate();

  // Welcome guide state
  const [showWelcomeGuide, setShowWelcomeGuide] = useState(false);

  // Check if this is the user's first visit
  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem("hasSeenWelcome");
    if (!hasSeenWelcome) {
      setShowWelcomeGuide(true);
      localStorage.setItem("hasSeenWelcome", "true");
    }
  }, []);

  // State for each step
  const [datasetState, setDatasetState] = useState<DatasetState>({
    file: null,
    trainSplit: 80,
  });

  const [experimentConfig, setExperimentConfig] =
    useState<ExperimentConfiguration>({
      trainingMode: "centralized",
      fineTuneLayers: 5,
      learningRate: 0.001,
      weightDecay: 0.0001,
      epochs: 10,
      batchSize: 32,
    });

  // State for saving experiments
  const [experimentName, setExperimentName] = useState<string>("");
  const [runId, setRunId] = useState<number | null>(null); // Changed from experimentId (string) to runId (number)
  const [experimentResults, setExperimentResults] = useState<any>(null);
  const [federatedResults, setFederatedResults] = useState<any>(null);

  // Federated training state (captured from TrainingExecution)
  const [isFederatedTraining, setIsFederatedTraining] = useState(false);
  const [federatedRounds, setFederatedRounds] = useState<
    Array<{
      round: number;
      metrics: Record<string, number>;
    }>
  >([]);
  const [federatedContext, setFederatedContext] = useState<
    | {
        numRounds: number;
        numClients: number;
      }
    | undefined
  >(undefined);

  // Handle loading a saved experiment
  useEffect(() => {
    // Handle viewing results directly from run ID
    if (location.state && location.state.viewRunId) {
      const viewRunId = location.state.viewRunId as number;

      console.log("[Index] Setting runId from navigation:", viewRunId);

      setRunId(viewRunId);

      // Fetch federated state for this run and update config
      const loadFederatedState = async () => {
        try {
          const fedData = await api.results.getFederatedRounds(viewRunId);
          console.log("[Index] Fetched federated data:", fedData);

          if (
            fedData.is_federated &&
            fedData.rounds &&
            fedData.rounds.length > 0
          ) {
            setIsFederatedTraining(true);
            setFederatedRounds(fedData.rounds);
            setFederatedContext({
              numRounds: fedData.num_rounds,
              numClients: fedData.num_clients,
            });

            // IMPORTANT: Update experiment config to reflect federated mode
            setExperimentConfig((prev) => ({
              ...prev,
              trainingMode: "federated",
            }));

            console.log("[Index] ✅ Loaded federated state:", {
              isFederated: true,
              numRounds: fedData.num_rounds,
              numClients: fedData.num_clients,
              roundsCount: fedData.rounds.length,
            });
          } else {
            console.log(
              "[Index] ⚠️ No federated data found or run is not federated:",
              {
                is_federated: fedData.is_federated,
                rounds_count: fedData.rounds?.length || 0,
              },
            );
          }
        } catch (error) {
          console.error("[Index] ❌ Error loading federated state:", error);
          // Not a critical error - continue anyway
        }
      };
      loadFederatedState();

      // Go to results step
      const targetStep =
        location.state.goToStep && typeof location.state.goToStep === "number"
          ? location.state.goToStep
          : 4;

      setCurrentStep(targetStep);

      toast.info(`Loading run #${viewRunId} results...`);

      // Clear navigation state to prevent reload issues (after a short delay to ensure state is set)
      setTimeout(() => {
        navigate(location.pathname, { replace: true, state: {} });
      }, 100);
      return;
    }

    // Handle loading a saved experiment (legacy)
    if (location.state && location.state.loadedExperiment) {
      const experiment = location.state.loadedExperiment as SavedExperiment;

      // Set experiment data
      setExperimentName(experiment.name);
      setDatasetState({
        file: null, // Can't restore actual file object
        trainSplit: experiment.trainSplit,
      });
      setExperimentConfig(experiment.configuration);

      if (experiment.results) {
        setExperimentResults(experiment.results);
      }

      if (experiment.federatedResults) {
        setFederatedResults(experiment.federatedResults);
      }

      // Go to specific step if specified
      if (
        location.state.goToStep &&
        typeof location.state.goToStep === "number"
      ) {
        setCurrentStep(location.state.goToStep);
      }

      toast.info(`Loaded experiment: ${experiment.name}`);
    }
  }, [location.state, navigate, location.pathname]);

  // Handle the completion of each step
  const handleDatasetComplete = (data: DatasetState) => {
    setDatasetState(data);
    setCurrentStep(2);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleConfigComplete = (config: ExperimentConfiguration) => {
    setExperimentConfig(config);
    setCurrentStep(3);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleTrainingComplete = (runIdValue: number) => {
    setRunId(runIdValue);
    setCurrentStep(4);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleReset = () => {
    setCurrentStep(1);
    setDatasetState({
      file: null,
      trainSplit: 80,
    });
    setExperimentConfig({
      trainingMode: "centralized",
      fineTuneLayers: 5,
      learningRate: 0.001,
      weightDecay: 0.0001,
      epochs: 10,
      batchSize: 32,
    });
    setExperimentResults(null);
    setFederatedResults(null);
    setExperimentName("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Handle going back a step
  const handleGoBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  // Handle saving the experiment
  const handleSaveExperiment = () => {
    const experimentToSave: SavedExperiment = {
      id: `${Date.now()}`,
      name: experimentName || `Experiment ${new Date().toLocaleDateString()}`,
      date: new Date().toISOString(),
      datasetName: datasetState.file?.name || "Unknown Dataset",
      trainSplit: datasetState.trainSplit,
      configuration: experimentConfig,
      results: experimentResults,
      federatedResults: federatedResults,
    };

    // In a real app, this would save to a database
    // For now, we'll just show a success toast and navigate to saved experiments
    toast.success("Experiment saved successfully!");
    navigate("/saved-experiments", {
      state: { newExperiment: experimentToSave },
    });
  };

  // Render the current step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <DatasetUpload
            onComplete={handleDatasetComplete}
            initialData={datasetState}
          />
        );
      case 2:
        return (
          <ExperimentConfig
            onComplete={handleConfigComplete}
            initialConfig={experimentConfig}
          />
        );
      case 3:
        return (
          <TrainingExecution
            config={experimentConfig}
            datasetFile={datasetState.file}
            trainSplit={datasetState.trainSplit}
            onComplete={handleTrainingComplete}
            onFederatedUpdate={(fedData) => {
              setIsFederatedTraining(fedData.isFederated);
              setFederatedRounds(fedData.rounds);
              setFederatedContext(fedData.context);
            }}
          />
        );
      case 4:
        console.log("[Index] Rendering step 4 (Results). runId:", runId);
        return runId ? (
          <ResultsVisualization
            config={experimentConfig}
            runId={runId}
            onReset={handleReset}
            isFederatedTraining={isFederatedTraining}
            federatedRounds={federatedRounds}
            federatedContext={federatedContext}
          />
        ) : (
          <div className="text-center p-8">
            <p className="text-muted-foreground">
              No run ID available. Please start training first.
            </p>
            <Button onClick={handleReset} className="mt-4">
              Start New Experiment
            </Button>
          </div>
        );
      default:
        return (
          <DatasetUpload
            onComplete={handleDatasetComplete}
            initialData={datasetState}
          />
        );
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Welcome Guide */}
      {showWelcomeGuide && (
        <WelcomeGuide onClose={() => setShowWelcomeGuide(false)} />
      )}

      <Header onShowHelp={() => setShowWelcomeGuide(true)} />

      <main className="flex-1 overflow-y-auto py-8">
        <div className="container px-4 md:px-6 max-w-7xl mx-auto">
          {/* Enhanced Progress Indicator with Breadcrumb & Step Circles */}
          <div className="animate-fade-in">
            <ProgressIndicator currentStep={currentStep} steps={steps} />
          </div>

          {/* Navigation Controls with improved styling */}
          <div className="navigation-bar animate-slide-in">
            {/* Back Button */}
            {currentStep > 1 ? (
              <Button
                variant="outline"
                onClick={handleGoBack}
                className="back-button group transition-all duration-300 hover:bg-medical/5 hover:border-medical hover:-translate-x-1"
              >
                <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
                Back to {steps[currentStep - 2].name}
              </Button>
            ) : (
              <div /> /* Maintain spacing */
            )}
          </div>

          {/* Step Content with Animations */}
          <StepContent stepKey={currentStep} isLoading={false}>
            <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm transition-all duration-300 hover:shadow-md">
              {renderStepContent()}
            </div>
          </StepContent>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Index;
