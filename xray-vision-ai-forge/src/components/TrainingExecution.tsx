import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Loader, Play, Check, XCircle, AlertCircle, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ExperimentConfiguration } from './ExperimentConfig';
import { useTrainingExecution } from '@/hooks/useTrainingExecution';

interface TrainingExecutionProps {
  config: ExperimentConfiguration;
  datasetFile: File | null;
  onComplete: (runId: number) => void;
  onFederatedUpdate?: (data: {
    isFederated: boolean;
    rounds: Array<{ round: number; metrics: Record<string, number> }>;
    context?: { numRounds: number; numClients: number };
  }) => void;
}

const TrainingExecution = ({ config, datasetFile, onComplete, onFederatedUpdate }: TrainingExecutionProps) => {
  const {
    isRunning,
    progress,
    statusMessages,
    overallStatus,
    logContainerRef,
    startTraining,
    handleCompleteStep,
    isFederatedTraining,
    federatedRounds,
    federatedContext,
  } = useTrainingExecution(config, datasetFile, onComplete);
  
  // Validation state
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  
  // Real-time validation
  useEffect(() => {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Dataset validation
    if (!datasetFile) {
      errors.push('No dataset uploaded. Please go back and upload a dataset.');
    }
    
    // Federated-specific validation
    if (config.trainingMode === 'federated') {
      if (!config.clients || config.clients < 2) {
        errors.push('Federated learning requires at least 2 clients.');
      }
      if (config.clients && config.clients > 20) {
        warnings.push(`Using ${config.clients} clients may significantly increase training time.`);
      }
      if (!config.federatedRounds || config.federatedRounds < 1) {
        errors.push('At least 1 federated round is required.');
      }
      if (config.federatedRounds && config.federatedRounds > 50) {
        warnings.push(`${config.federatedRounds} rounds is quite high and will take considerable time.`);
      }
      if (!config.localEpochs || config.localEpochs < 1) {
        errors.push('Local epochs must be at least 1.');
      }
    }
    
    // Centralized validation
    if (config.trainingMode === 'centralized') {
      if (!config.epochs || config.epochs < 1) {
        errors.push('At least 1 epoch is required for training.');
      }
      if (config.epochs && config.epochs > 100) {
        warnings.push(`${config.epochs} epochs may lead to overfitting. Consider using fewer epochs.`);
      }
    }
    
    // General validation
    if (config.batchSize < 8) {
      warnings.push('Small batch size may result in unstable training.');
    }
    if (config.batchSize > 128) {
      warnings.push('Large batch size may require more memory and affect convergence.');
    }
    
    if (config.learningRate > 0.01) {
      warnings.push('High learning rate may cause training instability.');
    }
    if (config.learningRate < 0.00001) {
      warnings.push('Very low learning rate will result in slow training.');
    }
    
    setValidationErrors(errors);
    setValidationWarnings(warnings);
  }, [config, datasetFile]);
  
  // Update parent component when federated state changes
  React.useEffect(() => {
    if (onFederatedUpdate) {
      onFederatedUpdate({
        isFederated: isFederatedTraining,
        rounds: federatedRounds,
        context: federatedContext,
      });
    }
  }, [isFederatedTraining, federatedRounds, federatedContext, onFederatedUpdate]);
  
  const canStartTraining = validationErrors.length === 0 && overallStatus === 'idle';

  return (
    <div className="animate-fade-in">
      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl text-medical-dark">Train Model</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex items-center mb-4">
              <h3 className="text-lg font-medium flex items-center gap-2">
                {overallStatus === 'idle' && (
                  <Play className="h-5 w-5 text-medical" />
                )}
                {overallStatus === 'running' && (
                  <Loader className="h-5 w-5 text-medical animate-spin" />
                )}
                {overallStatus === 'completed' && (
                  <Check className="h-5 w-5 text-status-success" />
                )}
                {overallStatus === 'error' && (
                  <XCircle className="h-5 w-5 text-status-error" />
                )}
                Experiment Status
              </h3>
              <div className="ml-auto text-sm font-medium">
                {overallStatus === 'idle' && 'Ready to start'}
                {overallStatus === 'running' && 'Running...'}
                {overallStatus === 'completed' && 'Completed'}
                {overallStatus === 'error' && 'Error'}
              </div>
            </div>

            <div className="mb-6">
              <Progress value={progress} className="h-2" />
              <div className="flex justify-between mt-1 text-xs text-muted-foreground">
                <span>0%</span>
                <span>{progress.toFixed(1)}% Complete</span>
                <span>100%</span>
              </div>
            </div>

            {/* Validation Messages */}
            {(validationErrors.length > 0 || validationWarnings.length > 0) && (
              <div className="mb-4 space-y-2 animate-fade-in">
                {validationErrors.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-3">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h5 className="font-semibold text-red-800 text-sm mb-1">
                          Validation Errors ({validationErrors.length})
                        </h5>
                        <ul className="text-sm text-red-700 space-y-1">
                          {validationErrors.map((error, idx) => (
                            <li key={idx} className="flex items-start gap-1">
                              <span className="text-red-500">•</span>
                              <span>{error}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
                
                {validationWarnings.length > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h5 className="font-semibold text-yellow-800 text-sm mb-1">
                          Warnings ({validationWarnings.length})
                        </h5>
                        <ul className="text-sm text-yellow-700 space-y-1">
                          {validationWarnings.map((warning, idx) => (
                            <li key={idx} className="flex items-start gap-1">
                              <span className="text-yellow-500">•</span>
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
            
            {/* Success indicator when no errors */}
            {validationErrors.length === 0 && datasetFile && (
              <div className="bg-green-50 border border-green-200 rounded-md p-3 mb-4 animate-fade-in">
                <div className="flex items-center gap-2 text-green-800">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  <span className="font-medium text-sm">Configuration is valid and ready for training</span>
                </div>
              </div>
            )}

            <div className="bg-white rounded-md p-3 mb-4 shadow-sm text-sm">
              <h4 className="font-medium mb-2">Configuration Summary:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-y-2 gap-x-4">
                <div>Mode: <span className="font-semibold">{config.trainingMode === 'centralized' ? 'Centralized Training' : 'Federated Learning'}</span></div>
                <div>Learning Rate: <span className="font-semibold">{config.learningRate}</span></div>
                <div>
                  {config.trainingMode === 'centralized' ? 'Epochs' : 'Local Epochs per Client'}: 
                  <span className="font-semibold">{config.trainingMode === 'centralized' ? config.epochs : config.localEpochs}</span>
                </div>
                <div>Batch Size: <span className="font-semibold">{config.batchSize}</span></div>
                {config.trainingMode === 'federated' && (
                  <>
                    <div>Clients: <span className="font-semibold">{config.clients || 'Not set'}</span></div>
                    <div>Federated Rounds: <span className="font-semibold">{config.federatedRounds || 'Not set'}</span></div>
                  </>
                )}
              </div>
            </div>

            <div
              ref={logContainerRef}
              className="bg-gray-900 text-gray-100 rounded-md font-mono text-xs p-4 h-64 overflow-y-auto whitespace-pre-wrap"
            >
              {statusMessages.length === 0 ? (
                <div className="text-gray-500 italic">Ready to start experiment. Press the "Start Training" button below.</div>
              ) : (
                statusMessages.map((msg, idx) => (
                  <div
                    key={`${msg.id}-${idx}`}
                    className={cn(
                      "leading-relaxed animate-slide-in",
                      msg.type === 'error' && "text-red-400 font-semibold",
                      msg.type === 'success' && "text-green-400",
                      msg.type === 'progress' && "text-blue-300",
                      msg.type === 'info' && "text-gray-300"
                    )}
                  >
                    {!msg.message.startsWith('━') && !msg.message.startsWith('🚀') && !msg.message.startsWith('   ') && (
                      <span className="text-gray-600">
                        [{new Date(msg.timestamp || Date.now()).toLocaleTimeString()}]{' '}
                      </span>
                    )}
                    {msg.message}
                  </div>
                ))
              )}
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex flex-col gap-3">
          {overallStatus === 'idle' && (
            <>
              <Button
                onClick={startTraining}
                className="bg-medical hover:bg-medical-dark mx-auto transition-all duration-300 hover:shadow-md button-hover"
                size="lg"
                disabled={!canStartTraining}
              >
                <Play className="mr-2 h-4 w-4" /> Start Training
              </Button>
              {!canStartTraining && validationErrors.length > 0 && (
                <p className="text-sm text-red-600 text-center mx-auto">
                  Please fix the validation errors above before starting training
                </p>
              )}
            </>
          )}

          {overallStatus === 'running' && (
            <div className="text-center mx-auto text-muted-foreground italic flex items-center">
              <Loader className="animate-spin mr-2 h-4 w-4" />
              Training in progress...
            </div>
          )}

          {overallStatus === 'completed' && (
            <Button
              onClick={handleCompleteStep}
              className="bg-status-success hover:bg-green-700 mx-auto"
              size="lg"
            >
              <Check className="mr-2 h-4 w-4" /> View Results
            </Button>
          )}

          {overallStatus === 'error' && (
            <Button
              onClick={startTraining}
              variant="outline"
              className="mx-auto"
              size="lg"
            >
              Retry Training
            </Button>
          )}
        </CardFooter>
      </Card>
    </div>
  );
};

export default TrainingExecution;
