
import React, { useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Info, Settings, Users, Brain, ChevronRight, Layers, Zap, Cpu, ImageIcon, Sparkles } from 'lucide-react';
import InstructionCard from './InstructionCard';
import HelpTooltip from './HelpTooltip';

interface ExperimentConfigProps {
  onComplete: (config: ExperimentConfiguration) => void;
  initialConfig?: ExperimentConfiguration;
}

export interface ExperimentConfiguration {
  // Basic required fields
  trainingMode: 'centralized' | 'federated';
  fineTuneLayers: number;
  learningRate: number;
  weightDecay: number;
  epochs: number;
  batchSize: number;

  // Federated learning fields
  clients?: number;
  federatedRounds?: number;
  localEpochs?: number;

  // Advanced model parameters
  dropoutRate?: number;
  freezeBackbone?: boolean;
  monitorMetric?: 'val_loss' | 'val_acc' | 'val_f1' | 'val_auroc';

  // Advanced training parameters
  earlyStoppingPatience?: number;
  reduceLrPatience?: number;
  reduceLrFactor?: number;
  minLr?: number;

  // System parameters
  device?: 'auto' | 'cpu' | 'cuda';
  numWorkers?: number;

  // Image processing parameters
  colorMode?: 'RGB' | 'L';
  useImagenetNorm?: boolean;
  augmentationStrength?: number;

  // Advanced preprocessing
  useCustomPreprocessing?: boolean;
  contrastStretch?: boolean;
  adaptiveHistogram?: boolean;
  edgeEnhancement?: boolean;
}

const ExperimentConfig = ({ onComplete, initialConfig }: ExperimentConfigProps) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [config, setConfig] = useState<ExperimentConfiguration>(initialConfig || {
    trainingMode: 'centralized',
    fineTuneLayers: 5,
    learningRate: 0.001,
    weightDecay: 0.0001,
    epochs: 10,
    batchSize: 128,
    clients: 3,
    federatedRounds: 3,
    localEpochs: 20,
    // Advanced defaults
    dropoutRate: 0.3,
    freezeBackbone: true,
    monitorMetric: 'val_loss',
    earlyStoppingPatience: 7,
    reduceLrPatience: 3,
    reduceLrFactor: 0.5,
    minLr: 0.0000001,
    device: 'auto',
    numWorkers: 0,
    colorMode: 'RGB',
    useImagenetNorm: true,
    augmentationStrength: 1.0,
    useCustomPreprocessing: false,
    contrastStretch: true,
    adaptiveHistogram: false,
    edgeEnhancement: false,
  });

  const [errors, setErrors] = useState<Partial<Record<keyof ExperimentConfiguration, string>>>({});

  const handleChange = (field: keyof ExperimentConfiguration, value: any) => {
    setErrors(prev => ({ ...prev, [field]: undefined }));
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const validateNumber = (value: number | undefined, min: number, max: number): boolean => {
    if (value === undefined) return true;
    return value >= min && value <= max;
  };

  const validateConfig = (): boolean => {
    const newErrors: Partial<Record<keyof ExperimentConfiguration, string>> = {};

    // Validate basic parameters
    if (!validateNumber(config.fineTuneLayers, 0, 50)) {
      newErrors.fineTuneLayers = 'Must be between 0 and 50';
    }

    if (!validateNumber(config.learningRate, 0.0001, 0.1)) {
      newErrors.learningRate = 'Must be between 0.0001 and 0.1';
    }

    if (!validateNumber(config.weightDecay, 0, 0.01)) {
      newErrors.weightDecay = 'Must be between 0 and 0.01';
    }

    if (!validateNumber(config.epochs, 1, 1000)) {
      newErrors.epochs = 'Must be between 1 and 1000';
    }

    if (!validateNumber(config.batchSize, 1, 512)) {
      newErrors.batchSize = 'Must be between 1 and 512';
    }

    // Validate federated parameters
    if (config.trainingMode === 'federated') {
      if (!config.clients || !validateNumber(config.clients, 2, 100)) {
        newErrors.clients = 'Must be between 2 and 100';
      }

      if (!config.federatedRounds || !validateNumber(config.federatedRounds, 1, 100)) {
        newErrors.federatedRounds = 'Must be between 1 and 100';
      }

      if (!config.localEpochs || !validateNumber(config.localEpochs, 1, 50)) {
        newErrors.localEpochs = 'Must be between 1 and 50';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (validateConfig()) {
      onComplete(config);
    }
  };

  const InfoTooltip = ({ text }: { text: string }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="h-4 w-4 text-[hsl(215_15%_55%)] cursor-help inline-block ml-1.5 hover:text-[hsl(172_63%_35%)] transition-colors" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs bg-white/95 backdrop-blur border-[hsl(168_20%_90%)] shadow-lg rounded-xl p-3">
          <p className="text-[hsl(172_43%_20%)]">{text}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div className="space-y-8" style={{ animation: 'fadeIn 0.5s ease-out' }}>
      {/* Instructions */}
      <InstructionCard
        variant="guide"
        title="Configuration Overview"
        className="max-w-4xl mx-auto"
      >
        <div className="space-y-3">
          <p className="font-medium text-[hsl(172_43%_20%)]">Choose your training approach:</p>
          <ul className="space-y-2 ml-1">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_28%)] mt-2 flex-shrink-0" />
              <span><strong className="text-[hsl(172_43%_20%)]">Centralized:</strong> Traditional training with all data in one location</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[hsl(172_63%_28%)] mt-2 flex-shrink-0" />
              <span><strong className="text-[hsl(172_43%_20%)]">Federated:</strong> Distributed training across multiple clients while keeping data private</span>
            </li>
          </ul>
        </div>
      </InstructionCard>

      {/* Main Config Card */}
      <div className="w-full max-w-4xl mx-auto">
        <div className="bg-white rounded-[2rem] border border-[hsl(210_15%_92%)] shadow-lg shadow-[hsl(172_40%_85%)]/20 overflow-hidden">
          {/* Header */}
          <div className="px-8 py-6 border-b border-[hsl(210_15%_94%)] bg-gradient-to-r from-[hsl(168_25%_98%)] to-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-2xl bg-[hsl(172_40%_94%)]">
                  <Settings className="h-6 w-6 text-[hsl(172_63%_28%)]" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                    Experiment Configuration
                  </h2>
                  <p className="text-[hsl(215_15%_50%)] mt-1">
                    Configure training parameters and model settings
                  </p>
                </div>
              </div>
              <HelpTooltip
                title="Configuration Tips"
                content={
                  <div className="space-y-2">
                    <p><strong className="text-[hsl(172_43%_20%)]">New to deep learning?</strong> Use the recommended defaults.</p>
                    <p><strong className="text-[hsl(172_43%_20%)]">Experimenting?</strong> Toggle "Show Advanced Options" for fine-grained control.</p>
                    <p className="text-xs mt-2 pt-2 border-t border-[hsl(168_20%_90%)]">
                      Hover over info icons for detailed explanations.
                    </p>
                  </div>
                }
                iconClassName="h-5 w-5"
              />
            </div>
          </div>

          {/* Content */}
          <div className="p-8 space-y-8">
            {/* Training Mode Selection - Apple-style Segmented Control */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)]">Training Mode</h3>
                <HelpTooltip
                  content={
                    <div className="space-y-2">
                      <p><strong className="text-[hsl(172_43%_20%)]">Centralized:</strong> All data is pooled and trained on a single machine. Faster and simpler.</p>
                      <p><strong className="text-[hsl(172_43%_20%)]">Federated:</strong> Data stays on individual clients; only model updates are shared. Preserves privacy.</p>
                    </div>
                  }
                />
              </div>

              {/* Segmented Control */}
              <div className="p-1.5 bg-[hsl(168_20%_95%)] rounded-2xl inline-flex">
                <button
                  onClick={() => handleChange('trainingMode', 'centralized')}
                  className={cn(
                    "relative px-8 py-4 rounded-xl font-medium transition-all duration-300 flex items-center gap-3",
                    config.trainingMode === 'centralized'
                      ? "bg-white text-[hsl(172_43%_20%)] shadow-md"
                      : "text-[hsl(215_15%_50%)] hover:text-[hsl(172_43%_25%)]"
                  )}
                >
                  <div className={cn(
                    "p-2 rounded-xl transition-colors",
                    config.trainingMode === 'centralized'
                      ? "bg-[hsl(172_40%_94%)]"
                      : "bg-transparent"
                  )}>
                    <Brain className={cn(
                      "h-5 w-5 transition-colors",
                      config.trainingMode === 'centralized'
                        ? "text-[hsl(172_63%_28%)]"
                        : "text-[hsl(215_15%_55%)]"
                    )} />
                  </div>
                  <div className="text-left">
                    <span className="block font-semibold">Centralized</span>
                    <span className="text-xs opacity-75">Single machine training</span>
                  </div>
                </button>

                <button
                  onClick={() => handleChange('trainingMode', 'federated')}
                  className={cn(
                    "relative px-8 py-4 rounded-xl font-medium transition-all duration-300 flex items-center gap-3",
                    config.trainingMode === 'federated'
                      ? "bg-white text-[hsl(172_43%_20%)] shadow-md"
                      : "text-[hsl(215_15%_50%)] hover:text-[hsl(172_43%_25%)]"
                  )}
                >
                  <div className={cn(
                    "p-2 rounded-xl transition-colors",
                    config.trainingMode === 'federated'
                      ? "bg-[hsl(172_40%_94%)]"
                      : "bg-transparent"
                  )}>
                    <Users className={cn(
                      "h-5 w-5 transition-colors",
                      config.trainingMode === 'federated'
                        ? "text-[hsl(172_63%_28%)]"
                        : "text-[hsl(215_15%_55%)]"
                    )} />
                  </div>
                  <div className="text-left">
                    <span className="block font-semibold">Federated</span>
                    <span className="text-xs opacity-75">Distributed & private</span>
                  </div>
                  <Badge className="absolute -top-2 -right-2 bg-[hsl(172_63%_28%)] text-white text-[10px] px-2 py-0.5 rounded-full">
                    Privacy
                  </Badge>
                </button>
              </div>
            </div>

            {/* Basic Model Parameters */}
            <div className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(168_20%_92%)]">
              <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-6">
                <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                  <Layers className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                </div>
                Basic Model Parameters
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="fineTuneLayers" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                    Number of Fine-Tune Layers
                    <InfoTooltip text="Number of layers to unfreeze for fine-tuning. 0 means freeze all layers (feature extraction only)." />
                  </Label>
                  <Input
                    id="fineTuneLayers"
                    type="number"
                    value={config.fineTuneLayers}
                    onChange={(e) => handleChange('fineTuneLayers', parseInt(e.target.value))}
                    className={cn(
                      "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                      errors.fineTuneLayers && "border-[hsl(0_72%_51%)]"
                    )}
                  />
                  {errors.fineTuneLayers && (
                    <p className="text-xs text-[hsl(0_72%_51%)]">{errors.fineTuneLayers}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="learningRate" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                    Learning Rate
                    <InfoTooltip text="Step size for gradient descent. Lower values train slower but more stable." />
                  </Label>
                  <Input
                    id="learningRate"
                    type="number"
                    step="0.0001"
                    value={config.learningRate}
                    onChange={(e) => handleChange('learningRate', parseFloat(e.target.value))}
                    className={cn(
                      "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                      errors.learningRate && "border-[hsl(0_72%_51%)]"
                    )}
                  />
                  {errors.learningRate && (
                    <p className="text-xs text-[hsl(0_72%_51%)]">{errors.learningRate}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="weightDecay" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                    Weight Decay
                    <InfoTooltip text="L2 regularization parameter to prevent overfitting." />
                  </Label>
                  <Input
                    id="weightDecay"
                    type="number"
                    step="0.0001"
                    value={config.weightDecay}
                    onChange={(e) => handleChange('weightDecay', parseFloat(e.target.value))}
                    className={cn(
                      "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                      errors.weightDecay && "border-[hsl(0_72%_51%)]"
                    )}
                  />
                  {errors.weightDecay && (
                    <p className="text-xs text-[hsl(0_72%_51%)]">{errors.weightDecay}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="epochs" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                    Number of Training Epochs
                    <InfoTooltip text="Complete passes through the training dataset. Training may stop early if validation metrics don't improve." />
                  </Label>
                  <Input
                    id="epochs"
                    type="number"
                    value={config.epochs}
                    onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
                    className={cn(
                      "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                      errors.epochs && "border-[hsl(0_72%_51%)]"
                    )}
                  />
                  {errors.epochs && (
                    <p className="text-xs text-[hsl(0_72%_51%)]">{errors.epochs}</p>
                  )}
                  <p className="text-xs text-[hsl(215_15%_55%)]">
                    Auto early-stop patience: {
                      config.epochs <= 10
                        ? Math.min(5, Math.max(3, config.epochs - 1))
                        : Math.min(15, Math.floor(config.epochs / 2))
                    } epochs without improvement
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="batchSize" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                    Batch Size
                    <InfoTooltip text="Number of samples processed before updating weights. Higher values use more memory." />
                  </Label>
                  <Input
                    id="batchSize"
                    type="number"
                    value={config.batchSize}
                    onChange={(e) => handleChange('batchSize', parseInt(e.target.value))}
                    className={cn(
                      "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                      errors.batchSize && "border-[hsl(0_72%_51%)]"
                    )}
                  />
                  {errors.batchSize && (
                    <p className="text-xs text-[hsl(0_72%_51%)]">{errors.batchSize}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Federated Learning Parameters */}
            {config.trainingMode === 'federated' && (
              <div
                className="bg-[hsl(168_25%_98%)] rounded-2xl p-6 border border-[hsl(172_40%_85%)]"
                style={{ animation: 'fadeIn 0.3s ease-out' }}
              >
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-[hsl(172_43%_15%)] flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-xl bg-[hsl(172_40%_94%)]">
                      <Users className="h-5 w-5 text-[hsl(172_63%_35%)]" />
                    </div>
                    Federated Learning Parameters
                    <Badge className="bg-[hsl(172_50%_92%)] text-[hsl(172_63%_25%)] text-xs ml-2">Federated</Badge>
                  </h3>
                  <p className="text-sm text-[hsl(215_15%_50%)]">
                    Configure how data is distributed across simulated clients and how often they communicate.
                  </p>
                </div>

                {/* How it works */}
                <div className="bg-white/60 backdrop-blur rounded-xl p-4 mb-6 border border-[hsl(168_20%_90%)]">
                  <h4 className="font-semibold text-[hsl(172_43%_20%)] mb-3 flex items-center gap-2">
                    <Info className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                    How Federated Learning Works
                  </h4>
                  <ol className="text-sm text-[hsl(215_15%_45%)] space-y-2 ml-4 list-decimal">
                    <li><strong className="text-[hsl(172_43%_25%)]">Data Partitioning:</strong> Dataset is split across the specified number of clients</li>
                    <li><strong className="text-[hsl(172_43%_25%)]">Local Training:</strong> Each client trains on their data for the local epochs</li>
                    <li><strong className="text-[hsl(172_43%_25%)]">Model Aggregation:</strong> Client updates are sent to server and averaged</li>
                    <li><strong className="text-[hsl(172_43%_25%)]">Repeat:</strong> Process repeats for the number of federated rounds</li>
                  </ol>
                  <p className="text-xs text-[hsl(172_43%_35%)] mt-3 pt-3 border-t border-[hsl(168_20%_90%)] italic">
                    This approach keeps data private at each client while benefiting from collective learning.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="clients" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                      Simulated Clients
                      <InfoTooltip text="Number of virtual clients to partition data across. Each client trains on its own subset independently." />
                    </Label>
                    <Input
                      id="clients"
                      type="number"
                      value={config.clients}
                      onChange={(e) => handleChange('clients', parseInt(e.target.value))}
                      className={cn(
                        "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                        errors.clients && "border-[hsl(0_72%_51%)]"
                      )}
                      placeholder="e.g., 5"
                    />
                    {errors.clients && (
                      <p className="text-xs text-[hsl(0_72%_51%)]">{errors.clients}</p>
                    )}
                    <p className="text-xs text-[hsl(215_15%_55%)]">
                      Simulates multiple hospitals/institutions
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="federatedRounds" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                      Federated Rounds
                      <InfoTooltip text="How many times clients will train locally and share model updates with the server." />
                    </Label>
                    <Input
                      id="federatedRounds"
                      type="number"
                      value={config.federatedRounds}
                      onChange={(e) => handleChange('federatedRounds', parseInt(e.target.value))}
                      className={cn(
                        "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                        errors.federatedRounds && "border-[hsl(0_72%_51%)]"
                      )}
                      placeholder="e.g., 10"
                    />
                    {errors.federatedRounds && (
                      <p className="text-xs text-[hsl(0_72%_51%)]">{errors.federatedRounds}</p>
                    )}
                    <p className="text-xs text-[hsl(215_15%_55%)]">
                      Communication cycles between clients/server
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="localEpochs" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                      Local Epochs per Client
                      <InfoTooltip text="Number of training epochs each client performs before sending updates to the server." />
                    </Label>
                    <Input
                      id="localEpochs"
                      type="number"
                      value={config.localEpochs}
                      onChange={(e) => handleChange('localEpochs', parseInt(e.target.value))}
                      className={cn(
                        "rounded-xl border-[hsl(210_15%_88%)] bg-white focus:border-[hsl(172_63%_35%)] focus:ring-[hsl(172_63%_35%)]/20",
                        errors.localEpochs && "border-[hsl(0_72%_51%)]"
                      )}
                      placeholder="e.g., 3"
                    />
                    {errors.localEpochs && (
                      <p className="text-xs text-[hsl(0_72%_51%)]">{errors.localEpochs}</p>
                    )}
                    <p className="text-xs text-[hsl(215_15%_55%)]">
                      Balances communication cost vs. performance
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Advanced Settings Toggle */}
            <div className="flex items-center space-x-3 py-2">
              <Switch
                id="advanced-mode"
                checked={showAdvanced}
                onCheckedChange={setShowAdvanced}
                className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
              />
              <Label htmlFor="advanced-mode" className="cursor-pointer text-[hsl(172_43%_20%)] font-medium">
                Show Advanced Settings
              </Label>
              {showAdvanced && (
                <Badge className="bg-[hsl(35_60%_92%)] text-[hsl(35_70%_35%)] text-xs">Advanced</Badge>
              )}
            </div>

            {/* Advanced Settings Accordion */}
            {showAdvanced && (
              <Accordion type="multiple" className="w-full space-y-3">
                {/* Model Settings */}
                <AccordionItem value="model" className="border border-[hsl(168_20%_92%)] rounded-xl overflow-hidden bg-white">
                  <AccordionTrigger className="px-5 py-4 hover:bg-[hsl(168_25%_98%)] hover:no-underline [&[data-state=open]]:bg-[hsl(168_25%_98%)]">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
                        <Sparkles className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="font-semibold text-[hsl(172_43%_20%)]">Advanced Model Settings</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5 pt-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="dropoutRate" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Dropout Rate
                          <InfoTooltip text="Probability of dropping neurons during training for regularization." />
                        </Label>
                        <Input
                          id="dropoutRate"
                          type="number"
                          step="0.1"
                          min="0"
                          max="0.9"
                          value={config.dropoutRate}
                          onChange={(e) => handleChange('dropoutRate', parseFloat(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="monitorMetric" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Monitor Metric
                          <InfoTooltip text="Metric to monitor for early stopping and model checkpointing." />
                        </Label>
                        <Select
                          value={config.monitorMetric}
                          onValueChange={(value) => handleChange('monitorMetric', value)}
                        >
                          <SelectTrigger className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="rounded-xl">
                            <SelectItem value="val_loss">Validation Loss</SelectItem>
                            <SelectItem value="val_acc">Validation Accuracy</SelectItem>
                            <SelectItem value="val_f1">Validation F1 Score</SelectItem>
                            <SelectItem value="val_auroc">Validation AUROC</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-center space-x-3 col-span-2 p-4 rounded-xl bg-[hsl(168_25%_98%)]">
                        <Switch
                          id="freezeBackbone"
                          checked={config.freezeBackbone}
                          onCheckedChange={(checked) => handleChange('freezeBackbone', checked)}
                          className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                        />
                        <Label htmlFor="freezeBackbone" className="cursor-pointer text-[hsl(172_43%_20%)]">
                          Freeze Backbone (Feature Extraction Mode)
                          <InfoTooltip text="When enabled, only train the classification head." />
                        </Label>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                {/* Training Settings */}
                <AccordionItem value="training" className="border border-[hsl(168_20%_92%)] rounded-xl overflow-hidden bg-white">
                  <AccordionTrigger className="px-5 py-4 hover:bg-[hsl(168_25%_98%)] hover:no-underline [&[data-state=open]]:bg-[hsl(168_25%_98%)]">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
                        <Zap className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="font-semibold text-[hsl(172_43%_20%)]">Advanced Training Settings</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5 pt-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="earlyStoppingPatience" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Early Stopping Patience
                          <InfoTooltip text="Number of epochs to wait without improvement before stopping training." />
                        </Label>
                        <Input
                          id="earlyStoppingPatience"
                          type="number"
                          value={config.earlyStoppingPatience}
                          onChange={(e) => handleChange('earlyStoppingPatience', parseInt(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                          placeholder={`Auto: ${
                            config.epochs <= 10
                              ? Math.min(5, Math.max(3, config.epochs - 1))
                              : Math.min(15, Math.floor(config.epochs / 2))
                          }`}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="reduceLrPatience" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Reduce LR Patience
                          <InfoTooltip text="Epochs to wait before reducing learning rate." />
                        </Label>
                        <Input
                          id="reduceLrPatience"
                          type="number"
                          value={config.reduceLrPatience}
                          onChange={(e) => handleChange('reduceLrPatience', parseInt(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="reduceLrFactor" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Reduce LR Factor
                          <InfoTooltip text="Factor to reduce learning rate by (new_lr = lr * factor)." />
                        </Label>
                        <Input
                          id="reduceLrFactor"
                          type="number"
                          step="0.1"
                          value={config.reduceLrFactor}
                          onChange={(e) => handleChange('reduceLrFactor', parseFloat(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="minLr" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Minimum Learning Rate
                          <InfoTooltip text="Lower bound for learning rate reduction." />
                        </Label>
                        <Input
                          id="minLr"
                          type="number"
                          step="0.0000001"
                          value={config.minLr}
                          onChange={(e) => handleChange('minLr', parseFloat(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                {/* System Settings */}
                <AccordionItem value="system" className="border border-[hsl(168_20%_92%)] rounded-xl overflow-hidden bg-white">
                  <AccordionTrigger className="px-5 py-4 hover:bg-[hsl(168_25%_98%)] hover:no-underline [&[data-state=open]]:bg-[hsl(168_25%_98%)]">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
                        <Cpu className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="font-semibold text-[hsl(172_43%_20%)]">System Settings</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5 pt-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="device" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Compute Device
                          <InfoTooltip text="Hardware to use for training. Auto selects GPU if available." />
                        </Label>
                        <Select
                          value={config.device}
                          onValueChange={(value) => handleChange('device', value)}
                        >
                          <SelectTrigger className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="rounded-xl">
                            <SelectItem value="auto">Auto (Recommended)</SelectItem>
                            <SelectItem value="cuda">CUDA (GPU)</SelectItem>
                            <SelectItem value="cpu">CPU</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="numWorkers" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Data Loader Workers
                          <InfoTooltip text="Parallel workers for data loading. 0 recommended for Windows." />
                        </Label>
                        <Input
                          id="numWorkers"
                          type="number"
                          value={config.numWorkers}
                          onChange={(e) => handleChange('numWorkers', parseInt(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                {/* Image Processing */}
                <AccordionItem value="image" className="border border-[hsl(168_20%_92%)] rounded-xl overflow-hidden bg-white">
                  <AccordionTrigger className="px-5 py-4 hover:bg-[hsl(168_25%_98%)] hover:no-underline [&[data-state=open]]:bg-[hsl(168_25%_98%)]">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
                        <ImageIcon className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="font-semibold text-[hsl(172_43%_20%)]">Image Processing</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5 pt-2 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="colorMode" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Color Mode
                          <InfoTooltip text="Image color format. RGB for color, L for grayscale." />
                        </Label>
                        <Select
                          value={config.colorMode}
                          onValueChange={(value) => handleChange('colorMode', value)}
                        >
                          <SelectTrigger className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="rounded-xl">
                            <SelectItem value="RGB">RGB (Color)</SelectItem>
                            <SelectItem value="L">Grayscale</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="augmentationStrength" className="text-sm font-medium text-[hsl(172_43%_20%)] flex items-center">
                          Augmentation Strength
                          <InfoTooltip text="Data augmentation intensity. 0=none, 2=maximum." />
                        </Label>
                        <Input
                          id="augmentationStrength"
                          type="number"
                          step="0.1"
                          min="0"
                          max="2"
                          value={config.augmentationStrength}
                          onChange={(e) => handleChange('augmentationStrength', parseFloat(e.target.value))}
                          className="rounded-xl border-[hsl(210_15%_88%)] focus:border-[hsl(172_63%_35%)]"
                        />
                      </div>

                      <div className="flex items-center space-x-3 col-span-2 p-4 rounded-xl bg-[hsl(168_25%_98%)]">
                        <Switch
                          id="useImagenetNorm"
                          checked={config.useImagenetNorm}
                          onCheckedChange={(checked) => handleChange('useImagenetNorm', checked)}
                          className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                        />
                        <Label htmlFor="useImagenetNorm" className="cursor-pointer text-[hsl(172_43%_20%)]">
                          Use ImageNet Normalization
                          <InfoTooltip text="Apply ImageNet mean/std normalization for transfer learning." />
                        </Label>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                {/* Custom Preprocessing */}
                <AccordionItem value="preprocessing" className="border border-[hsl(168_20%_92%)] rounded-xl overflow-hidden bg-white">
                  <AccordionTrigger className="px-5 py-4 hover:bg-[hsl(168_25%_98%)] hover:no-underline [&[data-state=open]]:bg-[hsl(168_25%_98%)]">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-[hsl(172_40%_94%)]">
                        <Settings className="h-4 w-4 text-[hsl(172_63%_35%)]" />
                      </div>
                      <span className="font-semibold text-[hsl(172_43%_20%)]">Custom Preprocessing</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-5 pb-5 pt-2 space-y-4">
                    <div className="space-y-4">
                      <div className="flex items-center space-x-3 p-4 rounded-xl bg-[hsl(168_25%_98%)]">
                        <Switch
                          id="useCustomPreprocessing"
                          checked={config.useCustomPreprocessing}
                          onCheckedChange={(checked) => handleChange('useCustomPreprocessing', checked)}
                          className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                        />
                        <Label htmlFor="useCustomPreprocessing" className="cursor-pointer text-[hsl(172_43%_20%)]">
                          Enable Custom Preprocessing
                          <InfoTooltip text="Apply custom image preprocessing techniques." />
                        </Label>
                      </div>

                      {config.useCustomPreprocessing && (
                        <div className="grid grid-cols-1 gap-3 pl-6 border-l-2 border-[hsl(172_40%_85%)]" style={{ animation: 'fadeIn 0.3s ease-out' }}>
                          <div className="flex items-center space-x-3">
                            <Switch
                              id="contrastStretch"
                              checked={config.contrastStretch}
                              onCheckedChange={(checked) => handleChange('contrastStretch', checked)}
                              className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                            />
                            <Label htmlFor="contrastStretch" className="cursor-pointer text-[hsl(172_43%_25%)]">
                              Contrast Stretching
                            </Label>
                          </div>

                          <div className="flex items-center space-x-3">
                            <Switch
                              id="adaptiveHistogram"
                              checked={config.adaptiveHistogram}
                              onCheckedChange={(checked) => handleChange('adaptiveHistogram', checked)}
                              className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                            />
                            <Label htmlFor="adaptiveHistogram" className="cursor-pointer text-[hsl(172_43%_25%)]">
                              Adaptive Histogram Equalization
                            </Label>
                          </div>

                          <div className="flex items-center space-x-3">
                            <Switch
                              id="edgeEnhancement"
                              checked={config.edgeEnhancement}
                              onCheckedChange={(checked) => handleChange('edgeEnhancement', checked)}
                              className="data-[state=checked]:bg-[hsl(172_63%_28%)]"
                            />
                            <Label htmlFor="edgeEnhancement" className="cursor-pointer text-[hsl(172_43%_25%)]">
                              Edge Enhancement
                            </Label>
                          </div>
                        </div>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-6 border-t border-[hsl(210_15%_94%)] bg-[hsl(168_25%_99%)]">
            <Button
              onClick={handleSubmit}
              className="ml-auto flex bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] text-white text-base px-8 py-6 rounded-xl shadow-lg shadow-[hsl(172_63%_22%)]/20 transition-all duration-300 hover:shadow-xl hover:shadow-[hsl(172_63%_22%)]/30 hover:-translate-y-0.5"
            >
              Configure and Continue
              <ChevronRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExperimentConfig;
