export interface SavedExperiment {
  id: string;
  name: string;
  date: string;
  datasetName: string;
  trainSplit: number;
  configuration: ExperimentConfiguration;
  results?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
  federatedResults?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
  final_epoch_stats?: FinalEpochStats | null;
}

/**
 * Final epoch confusion matrix statistics.
 * Pre-computed at training completion for instant rendering.
 */
export interface FinalEpochStats {
  sensitivity: number;    // TP / (TP + FN) - Recall
  specificity: number;    // TN / (TN + FP)
  precision_cm: number;   // TP / (TP + FP)
  accuracy_cm: number;    // (TP + TN) / Total
  f1_cm: number;          // 2 * (P * R) / (P + R)
}

export interface ExperimentConfiguration {
  // Basic required fields
  trainingMode: "centralized" | "federated" | "both";
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
  monitorMetric?: "val_loss" | "val_acc" | "val_f1" | "val_auroc";

  // Advanced training parameters
  earlyStoppingPatience?: number;
  reduceLrPatience?: number;
  reduceLrFactor?: number;
  minLr?: number;

  // System parameters
  device?: "auto" | "cpu" | "cuda";
  numWorkers?: number;

  // Image processing parameters
  colorMode?: "RGB" | "L";
  useImagenetNorm?: boolean;
  augmentationStrength?: number;

  // Advanced preprocessing
  useCustomPreprocessing?: boolean;
  contrastStretch?: boolean;
  adaptiveHistogram?: boolean;
  edgeEnhancement?: boolean;
}
