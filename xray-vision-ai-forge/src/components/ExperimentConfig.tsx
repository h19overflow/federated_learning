
import React, { useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Info, Settings, Users, Zap, Brain, TrendingUp } from 'lucide-react';
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
          <Info className="h-4 w-4 text-muted-foreground cursor-help inline-block ml-1" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          <p>{text}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div className="animate-fade-in space-y-6">
      {/* Instructions */}
      <InstructionCard 
        variant="info" 
        title="Configuration Overview"
        className="max-w-4xl mx-auto"
      >
        <div className="space-y-2">
          <p className="font-medium">Choose your training approach:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>Centralized:</strong> Traditional training with all data in one location</li>
            <li><strong>Federated:</strong> Distributed training across multiple clients while keeping data private</li>
            <li><strong>Both:</strong> Run both modes sequentially for comparison</li>
          </ul>
        </div>
      </InstructionCard>

      <Card className="w-full max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl text-medical-dark flex items-center gap-2">
                <Settings className="h-6 w-6" />
                Experiment Configuration
              </CardTitle>
              <CardDescription className="mt-2">
                Configure training parameters and model settings
              </CardDescription>
            </div>
            <HelpTooltip
              title="Configuration Tips"
              content={
                <div className="space-y-2">
                  <p><strong>New to deep learning?</strong> Use the recommended defaults.</p>
                  <p><strong>Experimenting?</strong> Toggle "Show Advanced Options" for fine-grained control.</p>
                  <p className="text-xs mt-2 pt-2 border-t">
                    Hover over the info icons (ℹ️) next to each setting for detailed explanations.
                  </p>
                </div>
              }
              iconClassName="h-5 w-5"
            />
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Training Mode Selection */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Label className="text-lg">Training Mode</Label>
              <HelpTooltip
                content={
                  <div className="space-y-2">
                    <p><strong>Centralized:</strong> All data is pooled and trained on a single machine. Faster and simpler, but requires data sharing.</p>
                    <p><strong>Federated:</strong> Data stays on individual clients; only model updates are shared. Preserves privacy but takes longer.</p>
                    <p><strong>Both:</strong> Runs both approaches to compare performance and training dynamics.</p>
                  </div>
                }
              />
            </div>
            <RadioGroup
              value={config.trainingMode}
              onValueChange={(value) => handleChange('trainingMode', value as any)}
              className="grid grid-cols-1 md:grid-cols-3 gap-4"
            >
              <div className={cn(
                "flex items-center space-x-3 border-2 rounded-lg p-4 cursor-pointer transition-all",
                config.trainingMode === 'centralized' ? 'border-medical bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              )}>
                <RadioGroupItem value="centralized" id="centralized" />
                <Label htmlFor="centralized" className="cursor-pointer flex-1">
                  <div className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-medical" />
                    <span className="font-semibold">Centralized</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Single machine training</p>
                </Label>
              </div>
              <div className={cn(
                "flex items-center space-x-3 border-2 rounded-lg p-4 cursor-pointer transition-all",
                config.trainingMode === 'federated' ? 'border-medical bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              )}>
                <RadioGroupItem value="federated" id="federated" />
                <Label htmlFor="federated" className="cursor-pointer flex-1">
                  <div className="flex items-center gap-2">
                    <Users className="h-5 w-5 text-medical" />
                    <span className="font-semibold">Federated</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Distributed & private</p>
                </Label>
              </div>
            </RadioGroup>
          </div>

          {/* Basic Model Parameters */}
          <div className="bg-muted/50 p-4 rounded-lg">
            <h3 className="text-lg font-medium mb-4">Basic Model Parameters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="fineTuneLayers">
                  Number of Fine-Tune Layers
                  <InfoTooltip text="Number of layers to unfreeze for fine-tuning. 0 means freeze all layers (feature extraction only)." />
                </Label>
                <Input
                  id="fineTuneLayers"
                  type="number"
                  value={config.fineTuneLayers}
                  onChange={(e) => handleChange('fineTuneLayers', parseInt(e.target.value))}
                  className={cn(errors.fineTuneLayers ? "border-destructive" : "")}
                />
                {errors.fineTuneLayers && (
                  <p className="text-xs text-destructive">{errors.fineTuneLayers}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="learningRate">
                  Learning Rate
                  <InfoTooltip text="Step size for gradient descent. Lower values train slower but more stable." />
                </Label>
                <Input
                  id="learningRate"
                  type="number"
                  step="0.0001"
                  value={config.learningRate}
                  onChange={(e) => handleChange('learningRate', parseFloat(e.target.value))}
                  className={cn(errors.learningRate ? "border-destructive" : "")}
                />
                {errors.learningRate && (
                  <p className="text-xs text-destructive">{errors.learningRate}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="weightDecay">
                  Weight Decay
                  <InfoTooltip text="L2 regularization parameter to prevent overfitting." />
                </Label>
                <Input
                  id="weightDecay"
                  type="number"
                  step="0.0001"
                  value={config.weightDecay}
                  onChange={(e) => handleChange('weightDecay', parseFloat(e.target.value))}
                  className={cn(errors.weightDecay ? "border-destructive" : "")}
                />
                {errors.weightDecay && (
                  <p className="text-xs text-destructive">{errors.weightDecay}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="epochs">
                  Number of Training Epochs
                  <InfoTooltip text="Complete passes through the training dataset. Note: Training may stop early if validation metrics don't improve (controlled by Early Stopping Patience in Advanced Settings)." />
                </Label>
                <Input
                  id="epochs"
                  type="number"
                  value={config.epochs}
                  onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
                  className={cn(errors.epochs ? "border-destructive" : "")}
                />
                {errors.epochs && (
                  <p className="text-xs text-destructive">{errors.epochs}</p>
                )}
                <p className="text-xs text-muted-foreground">
                  Auto early-stop patience: {
                    config.epochs <= 10
                      ? Math.min(5, Math.max(3, config.epochs - 1))
                      : Math.min(15, Math.floor(config.epochs / 2))
                  } epochs without improvement
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="batchSize">
                  Batch Size
                  <InfoTooltip text="Number of samples processed before updating weights. Higher values use more memory." />
                </Label>
                <Input
                  id="batchSize"
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => handleChange('batchSize', parseInt(e.target.value))}
                  className={cn(errors.batchSize ? "border-destructive" : "")}
                />
                {errors.batchSize && (
                  <p className="text-xs text-destructive">{errors.batchSize}</p>
                )}
              </div>
            </div>
          </div>

          {/* Federated Learning Parameters */}
          {config.trainingMode === 'federated' && (
            <div className="bg-muted/50 p-4 rounded-lg animate-fade-in">
              <div className="mb-4">
                <h3 className="text-lg font-medium mb-2">Federated Learning Parameters</h3>
                <p className="text-sm text-muted-foreground">
                  Configure how the data is distributed across simulated clients and how often they communicate with the central server.
                </p>
              </div>
              
              {/* Federated Learning Info Box */}
              <div className="bg-slate-50 border border-slate-200 rounded-md p-3 mb-4 text-sm">
                <h4 className="font-semibold text-slate-700 mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4 text-medical" />
                  How Federated Learning Works
                </h4>
                <ol className="text-slate-600 space-y-1 ml-4 list-decimal">
                  <li><strong>Data Partitioning:</strong> Your dataset is split across the specified number of clients</li>
                  <li><strong>Local Training:</strong> Each client trains the model on their data for the specified local epochs</li>
                  <li><strong>Model Aggregation:</strong> Client model updates are sent to the server and aggregated (averaged)</li>
                  <li><strong>Distribution:</strong> The improved global model is sent back to all clients</li>
                  <li><strong>Repeat:</strong> Steps 2-4 repeat for the number of federated rounds specified</li>
                </ol>
                <p className="text-xs text-slate-500 mt-2 italic">
                  💡 This approach keeps data private at each client while still benefiting from collective learning.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="clients">
                    Number of Simulated Clients
                    <InfoTooltip text="Number of virtual clients to partition data across. Each client trains on its own subset of data independently." />
                  </Label>
                  <Input
                    id="clients"
                    type="number"
                    value={config.clients}
                    onChange={(e) => handleChange('clients', parseInt(e.target.value))}
                    className={cn(errors.clients ? "border-destructive" : "")}
                    placeholder="e.g., 5"
                  />
                  {errors.clients && (
                    <p className="text-xs text-destructive">{errors.clients}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Simulates multiple hospitals or institutions. More clients = more realistic but slower training.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="federatedRounds">
                    Number of Federated Rounds
                    <InfoTooltip text="How many times clients will train locally and share their model updates with the server. Each round involves: local training → model aggregation → distribution." />
                  </Label>
                  <Input
                    id="federatedRounds"
                    type="number"
                    value={config.federatedRounds}
                    onChange={(e) => handleChange('federatedRounds', parseInt(e.target.value))}
                    className={cn(errors.federatedRounds ? "border-destructive" : "")}
                    placeholder="e.g., 10"
                  />
                  {errors.federatedRounds && (
                    <p className="text-xs text-destructive">{errors.federatedRounds}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Communication cycles between clients and server. More rounds = better convergence but longer training.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="localEpochs">
                    Local Epochs per Client
                    <InfoTooltip text="Number of training epochs each client performs on their local data before sending updates to the server. Higher values mean less communication but may reduce model quality." />
                  </Label>
                  <Input
                    id="localEpochs"
                    type="number"
                    value={config.localEpochs}
                    onChange={(e) => handleChange('localEpochs', parseInt(e.target.value))}
                    className={cn(errors.localEpochs ? "border-destructive" : "")}
                    placeholder="e.g., 3"
                  />
                  {errors.localEpochs && (
                    <p className="text-xs text-destructive">{errors.localEpochs}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Epochs per round at each client. Balances communication cost vs. model performance.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Advanced Settings Toggle */}
          <div className="flex items-center space-x-2 pt-2">
            <Switch
              id="advanced-mode"
              checked={showAdvanced}
              onCheckedChange={setShowAdvanced}
            />
            <Label htmlFor="advanced-mode" className="cursor-pointer">
              Show Advanced Settings
            </Label>
          </div>

          {/* Advanced Settings Accordion */}
          {showAdvanced && (
            <Accordion type="multiple" className="w-full">
              {/* Model Settings */}
              <AccordionItem value="model">
                <AccordionTrigger>Advanced Model Settings</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="dropoutRate">
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
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="monitorMetric">
                        Monitor Metric
                        <InfoTooltip text="Metric to monitor for early stopping and model checkpointing." />
                      </Label>
                      <Select
                        value={config.monitorMetric}
                        onValueChange={(value) => handleChange('monitorMetric', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="val_loss">Validation Loss</SelectItem>
                          <SelectItem value="val_acc">Validation Accuracy</SelectItem>
                          <SelectItem value="val_f1">Validation F1 Score</SelectItem>
                          <SelectItem value="val_auroc">Validation AUROC</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center space-x-2 col-span-2">
                      <Switch
                        id="freezeBackbone"
                        checked={config.freezeBackbone}
                        onCheckedChange={(checked) => handleChange('freezeBackbone', checked)}
                      />
                      <Label htmlFor="freezeBackbone" className="cursor-pointer">
                        Freeze Backbone (Feature Extraction Mode)
                        <InfoTooltip text="When enabled, only train the classification head." />
                      </Label>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Training Settings */}
              <AccordionItem value="training">
                <AccordionTrigger>Advanced Training Settings</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="earlyStoppingPatience">
                        Early Stopping Patience
                        <InfoTooltip text="Number of epochs to wait without improvement before stopping training. Default: half of total epochs (capped at 15). Set higher to let training explore more, or lower to stop faster." />
                      </Label>
                      <Input
                        id="earlyStoppingPatience"
                        type="number"
                        value={config.earlyStoppingPatience}
                        onChange={(e) => handleChange('earlyStoppingPatience', parseInt(e.target.value))}
                        placeholder={`Auto: ${
                          config.epochs <= 10
                            ? Math.min(5, Math.max(3, config.epochs - 1))
                            : Math.min(15, Math.floor(config.epochs / 2))
                        }`}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="reduceLrPatience">
                        Reduce LR Patience
                        <InfoTooltip text="Epochs to wait before reducing learning rate." />
                      </Label>
                      <Input
                        id="reduceLrPatience"
                        type="number"
                        value={config.reduceLrPatience}
                        onChange={(e) => handleChange('reduceLrPatience', parseInt(e.target.value))}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="reduceLrFactor">
                        Reduce LR Factor
                        <InfoTooltip text="Factor to reduce learning rate by (new_lr = lr * factor)." />
                      </Label>
                      <Input
                        id="reduceLrFactor"
                        type="number"
                        step="0.1"
                        value={config.reduceLrFactor}
                        onChange={(e) => handleChange('reduceLrFactor', parseFloat(e.target.value))}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="minLr">
                        Minimum Learning Rate
                        <InfoTooltip text="Lower bound for learning rate reduction." />
                      </Label>
                      <Input
                        id="minLr"
                        type="number"
                        step="0.0000001"
                        value={config.minLr}
                        onChange={(e) => handleChange('minLr', parseFloat(e.target.value))}
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* System Settings */}
              <AccordionItem value="system">
                <AccordionTrigger>System Settings</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="device">
                        Compute Device
                        <InfoTooltip text="Hardware to use for training. Auto selects GPU if available." />
                      </Label>
                      <Select
                        value={config.device}
                        onValueChange={(value) => handleChange('device', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto (Recommended)</SelectItem>
                          <SelectItem value="cuda">CUDA (GPU)</SelectItem>
                          <SelectItem value="cpu">CPU</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="numWorkers">
                        Data Loader Workers
                        <InfoTooltip text="Parallel workers for data loading. 0 recommended for Windows." />
                      </Label>
                      <Input
                        id="numWorkers"
                        type="number"
                        value={config.numWorkers}
                        onChange={(e) => handleChange('numWorkers', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Image Processing */}
              <AccordionItem value="image">
                <AccordionTrigger>Image Processing</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="colorMode">
                        Color Mode
                        <InfoTooltip text="Image color format. RGB for color, L for grayscale." />
                      </Label>
                      <Select
                        value={config.colorMode}
                        onValueChange={(value) => handleChange('colorMode', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="RGB">RGB (Color)</SelectItem>
                          <SelectItem value="L">Grayscale</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="augmentationStrength">
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
                      />
                    </div>

                    <div className="flex items-center space-x-2 col-span-2">
                      <Switch
                        id="useImagenetNorm"
                        checked={config.useImagenetNorm}
                        onCheckedChange={(checked) => handleChange('useImagenetNorm', checked)}
                      />
                      <Label htmlFor="useImagenetNorm" className="cursor-pointer">
                        Use ImageNet Normalization
                        <InfoTooltip text="Apply ImageNet mean/std normalization for transfer learning." />
                      </Label>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Custom Preprocessing */}
              <AccordionItem value="preprocessing">
                <AccordionTrigger>Custom Preprocessing</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="useCustomPreprocessing"
                        checked={config.useCustomPreprocessing}
                        onCheckedChange={(checked) => handleChange('useCustomPreprocessing', checked)}
                      />
                      <Label htmlFor="useCustomPreprocessing" className="cursor-pointer">
                        Enable Custom Preprocessing
                        <InfoTooltip text="Apply custom image preprocessing techniques." />
                      </Label>
                    </div>

                    {config.useCustomPreprocessing && (
                      <div className="grid grid-cols-1 gap-3 pl-6 border-l-2 border-muted">
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="contrastStretch"
                            checked={config.contrastStretch}
                            onCheckedChange={(checked) => handleChange('contrastStretch', checked)}
                          />
                          <Label htmlFor="contrastStretch" className="cursor-pointer">
                            Contrast Stretching
                          </Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Switch
                            id="adaptiveHistogram"
                            checked={config.adaptiveHistogram}
                            onCheckedChange={(checked) => handleChange('adaptiveHistogram', checked)}
                          />
                          <Label htmlFor="adaptiveHistogram" className="cursor-pointer">
                            Adaptive Histogram Equalization
                          </Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Switch
                            id="edgeEnhancement"
                            checked={config.edgeEnhancement}
                            onCheckedChange={(checked) => handleChange('edgeEnhancement', checked)}
                          />
                          <Label htmlFor="edgeEnhancement" className="cursor-pointer">
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
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleSubmit}
            className="ml-auto bg-medical hover:bg-medical-dark"
          >
            Configure and Continue
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default ExperimentConfig;
