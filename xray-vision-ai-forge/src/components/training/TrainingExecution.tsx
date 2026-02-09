import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  Play,
  Check,
  XCircle,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Terminal,
  Settings,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ExperimentConfiguration } from "./ExperimentConfig";
import { useTrainingExecution } from "@/hooks/useTrainingExecution";
import { useTrainingMetrics } from "@/hooks/useTrainingMetrics";
import { InstructionCard } from "@/components/shared";
import { TrainingObservabilityPanel } from "@/components/observability";

interface TrainingExecutionProps {
  config: ExperimentConfiguration;
  datasetFile: File | null;
  trainSplit: number;
  onComplete: (runId: number) => void;
  onFederatedUpdate?: (data: {
    isFederated: boolean;
    rounds: Array<{ round: number; metrics: Record<string, number> }>;
    context?: { numRounds: number; numClients: number };
  }) => void;
}

const TrainingExecution = ({
  config,
  datasetFile,
  trainSplit,
  onComplete,
  onFederatedUpdate,
}: TrainingExecutionProps) => {
  const {
    isRunning,
    statusMessages,
    overallStatus,
    logContainerRef,
    startTraining,
    handleCompleteStep,
    isFederatedTraining,
    federatedRounds,
    federatedContext,
    ws,
  } = useTrainingExecution(config, datasetFile, trainSplit, onComplete);

  const trainingMetrics = useTrainingMetrics(ws);

  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  useEffect(() => {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!datasetFile) {
      errors.push("No dataset uploaded. Please go back and upload a dataset.");
    }

    if (config.trainingMode === "federated") {
      if (!config.clients || config.clients < 2) {
        errors.push("Federated learning requires at least 2 clients.");
      }
      if (config.clients && config.clients > 20) {
        warnings.push(
          `Using ${config.clients} clients may significantly increase training time.`,
        );
      }
      if (!config.federatedRounds || config.federatedRounds < 1) {
        errors.push("At least 1 federated round is required.");
      }
      if (!config.localEpochs || config.localEpochs < 1) {
        errors.push("Local epochs must be at least 1.");
      }
    }

    if (config.trainingMode === "centralized") {
      if (!config.epochs || config.epochs < 1) {
        errors.push("At least 1 epoch is required for training.");
      }
    }

    if (config.batchSize < 8) {
      warnings.push("Small batch size may result in unstable training.");
    }

    setValidationErrors(errors);
    setValidationWarnings(warnings);
  }, [config, datasetFile]);

  React.useEffect(() => {
    if (onFederatedUpdate) {
      onFederatedUpdate({
        isFederated: isFederatedTraining,
        rounds: federatedRounds,
        context: federatedContext,
      });
    }
  }, [
    isFederatedTraining,
    federatedRounds,
    federatedContext,
    onFederatedUpdate,
  ]);

  const canStartTraining =
    validationErrors.length === 0 && overallStatus === "idle";

  const getStatusBadge = () => {
    switch (overallStatus) {
      case "idle":
        return (
          <Badge className="bg-[hsl(210_40%_94%)] text-[hsl(210_60%_40%)] border-0 px-3 py-1">
            <span className="w-2 h-2 rounded-full bg-[hsl(210_60%_50%)] mr-2" />
            Ready
          </Badge>
        );
      case "running":
        return (
          <Badge className="bg-[hsl(172_40%_92%)] text-[hsl(172_63%_25%)] border-0 px-3 py-1">
            <Loader2 className="w-3 h-3 mr-2 animate-spin" />
            Training
          </Badge>
        );
      case "completed":
        return (
          <Badge className="bg-[hsl(172_50%_90%)] text-[hsl(172_63%_22%)] border-0 px-3 py-1">
            <CheckCircle2 className="w-3 h-3 mr-2" />
            Completed
          </Badge>
        );
      case "error":
        return (
          <Badge className="bg-[hsl(0_60%_95%)] text-[hsl(0_72%_45%)] border-0 px-3 py-1">
            <XCircle className="w-3 h-3 mr-2" />
            Error
          </Badge>
        );
    }
  };

  return (
    <div className="space-y-8" style={{ animation: "fadeIn 0.5s ease-out" }}>
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 overflow-hidden">
          {/* Header */}
          <div className="px-8 py-6 border-b border-[hsl(210_15%_94%)] bg-gradient-to-r from-[hsl(168_25%_98%)] to-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className={cn(
                    "p-3 rounded-2xl transition-colors",
                    overallStatus === "running"
                      ? "bg-[hsl(172_40%_94%)]"
                      : overallStatus === "completed"
                        ? "bg-[hsl(172_50%_92%)]"
                        : overallStatus === "error"
                          ? "bg-[hsl(0_60%_95%)]"
                          : "bg-[hsl(168_25%_94%)]",
                  )}
                >
                  {overallStatus === "idle" && (
                    <Play className="h-6 w-6 text-[hsl(172_63%_28%)]" />
                  )}
                  {overallStatus === "running" && (
                    <Activity className="h-6 w-6 text-[hsl(172_63%_28%)] animate-pulse" />
                  )}
                  {overallStatus === "completed" && (
                    <CheckCircle2 className="h-6 w-6 text-[hsl(172_63%_25%)]" />
                  )}
                  {overallStatus === "error" && (
                    <XCircle className="h-6 w-6 text-[hsl(0_72%_51%)]" />
                  )}
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                    Train Model
                  </h2>
                  <p className="text-[hsl(215_15%_50%)] mt-1">
                    {overallStatus === "idle" && "Ready to begin training"}
                    {overallStatus === "running" && "Training in progress..."}
                    {overallStatus === "completed" &&
                      "Training completed successfully"}
                    {overallStatus === "error" &&
                      "Training encountered an error"}
                  </p>
                </div>
              </div>
              {getStatusBadge()}
            </div>
          </div>

          {/* Content */}
          <div className="p-8 space-y-6">
            {/* Validation Messages */}
            {(validationErrors.length > 0 || validationWarnings.length > 0) && (
              <div
                className="space-y-4"
                style={{ animation: "fadeIn 0.3s ease-out" }}
              >
                {validationErrors.length > 0 && (
                  <div className="bg-[hsl(0_60%_97%)] border border-[hsl(0_50%_85%)] rounded-xl p-5">
                    <div className="flex items-start gap-4">
                      <div className="p-2 rounded-xl bg-[hsl(0_60%_92%)]">
                        <AlertTriangle className="h-5 w-5 text-[hsl(0_72%_51%)]" />
                      </div>
                      <div className="flex-1">
                        <h5 className="font-semibold text-[hsl(0_72%_40%)] mb-2">
                          Validation Errors ({validationErrors.length})
                        </h5>
                        <ul className="text-sm text-[hsl(0_60%_40%)] space-y-1.5">
                          {validationErrors.map((error, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(0_72%_51%)] mt-1.5 flex-shrink-0" />
                              <span>{error}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {validationWarnings.length > 0 && (
                  <div className="bg-[hsl(35_60%_96%)] border border-[hsl(35_50%_85%)] rounded-xl p-5">
                    <div className="flex items-start gap-4">
                      <div className="p-2 rounded-xl bg-[hsl(35_50%_90%)]">
                        <AlertTriangle className="h-5 w-5 text-[hsl(35_70%_45%)]" />
                      </div>
                      <div className="flex-1">
                        <h5 className="font-semibold text-[hsl(35_70%_35%)] mb-2">
                          Warnings ({validationWarnings.length})
                        </h5>
                        <ul className="text-sm text-[hsl(35_60%_35%)] space-y-1.5">
                          {validationWarnings.map((warning, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(35_70%_50%)] mt-1.5 flex-shrink-0" />
                              <span>{warning}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {validationErrors.length === 0 &&
              datasetFile &&
              overallStatus === "idle" && (
                <InstructionCard variant="success">
                  <span className="font-medium">
                    Configuration is valid and ready for training
                  </span>
                </InstructionCard>
              )}

            {/* Configuration Summary */}
            <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
              <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-4">
                <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                  <Settings className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                </div>
                Configuration Summary
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                  <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                    Training Mode
                  </p>
                  <p className="font-semibold text-[hsl(172_43%_20%)]">
                    {config.trainingMode === "centralized"
                      ? "Centralized"
                      : "Federated Learning"}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                  <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                    Learning Rate
                  </p>
                  <p className="font-semibold font-mono text-[hsl(172_43%_20%)]">
                    {config.learningRate}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                  <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                    {config.trainingMode === "centralized"
                      ? "Epochs"
                      : "Local Epochs"}
                  </p>
                  <p className="font-semibold text-[hsl(172_43%_20%)]">
                    {config.trainingMode === "centralized"
                      ? config.epochs
                      : config.localEpochs}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                  <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                    Batch Size
                  </p>
                  <p className="font-semibold text-[hsl(172_43%_20%)]">
                    {config.batchSize}
                  </p>
                </div>
                {config.trainingMode === "federated" && (
                  <>
                    <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                      <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                        Clients
                      </p>
                      <p className="font-semibold text-[hsl(172_43%_20%)]">
                        {config.clients || "Not set"}
                      </p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-[hsl(210_15%_94%)]">
                      <p className="text-xs text-[hsl(215_15%_55%)] uppercase tracking-wide mb-1">
                        Federated Rounds
                      </p>
                      <p className="font-semibold text-[hsl(172_43%_20%)]">
                        {config.federatedRounds || "Not set"}
                      </p>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Training Log */}
            <div className="bg-[hsl(172_30%_12%)] rounded-2xl p-6 border border-[hsl(172_40%_20%)]">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-xl bg-[hsl(172_35%_18%)]">
                  <Terminal className="h-4 w-4 text-[hsl(168_40%_60%)]" />
                </div>
                <h4 className="text-sm font-medium text-[hsl(168_30%_70%)]">
                  Training Log
                </h4>
              </div>
              <div
                ref={logContainerRef}
                className="font-mono text-xs h-56 overflow-y-auto pr-2 space-y-1"
                style={{
                  scrollbarWidth: "thin",
                  scrollbarColor: "hsl(172 30% 25%) transparent",
                }}
              >
                {statusMessages.length === 0 ? (
                  <div className="text-[hsl(168_20%_50%)] italic py-4">
                    Ready to start experiment. Press the "Start Training" button
                    below.
                  </div>
                ) : (
                  statusMessages.map((msg, idx) => (
                    <div
                      key={`${msg.id}-${idx}`}
                      className={cn(
                        "leading-relaxed py-0.5",
                        msg.type === "error" &&
                          "text-[hsl(0_70%_65%)] font-semibold",
                        msg.type === "success" && "text-[hsl(172_60%_55%)]",
                        msg.type === "progress" && "text-[hsl(210_70%_70%)]",
                        msg.type === "info" && "text-[hsl(168_20%_70%)]",
                      )}
                      style={{ animation: "fadeIn 0.2s ease-out" }}
                    >
                      {!msg.message.startsWith("━") &&
                        !msg.message.startsWith("   ") && (
                          <span className="text-[hsl(168_20%_45%)]">
                            [
                            {new Date(
                              msg.timestamp || Date.now(),
                            ).toLocaleTimeString()}
                            ]{" "}
                          </span>
                        )}
                      {msg.message}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Live Observability Panel - below terminal (centralized training only) */}
            {isRunning && config.trainingMode !== "federated" && (
              <TrainingObservabilityPanel
                batchMetrics={trainingMetrics.batchMetrics}
                currentLoss={trainingMetrics.currentLoss}
                currentAccuracy={trainingMetrics.currentAccuracy}
                currentF1={trainingMetrics.currentF1}
                confusionMatrix={trainingMetrics.confusionMatrix}
                isReceiving={trainingMetrics.isReceiving}
                trainingMode={config.trainingMode}
              />
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-6 border-t border-[hsl(210_15%_94%)] bg-[hsl(168_25%_99%)]">
            <div className="flex flex-col items-center gap-3">
              {overallStatus === "idle" && (
                <>
                  <Button
                    onClick={startTraining}
                    disabled={!canStartTraining}
                    className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-base px-8 py-6 rounded-xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-lg"
                  >
                    <Play className="mr-2 h-5 w-5" /> Start Training
                  </Button>
                  {!canStartTraining && validationErrors.length > 0 && (
                    <p className="text-sm text-[hsl(0_72%_45%)] text-center">
                      Please fix the validation errors above before starting
                      training
                    </p>
                  )}
                </>
              )}

              {overallStatus === "running" && (
                <div className="flex items-center gap-3 text-[hsl(172_43%_30%)]">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span className="font-medium">Training in progress...</span>
                </div>
              )}

              {overallStatus === "completed" && (
                <Button
                  onClick={handleCompleteStep}
                  className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-base px-8 py-6 rounded-xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5"
                >
                  <Check className="mr-2 h-5 w-5" /> View Results
                  <ChevronRight className="ml-2 h-5 w-5" />
                </Button>
              )}

              {overallStatus === "error" && (
                <Button
                  onClick={startTraining}
                  variant="outline"
                  className="rounded-xl border-[hsl(0_50%_75%)] text-[hsl(0_72%_45%)] hover:bg-[hsl(0_50%_97%)] px-8 py-6"
                >
                  Retry Training
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingExecution;
