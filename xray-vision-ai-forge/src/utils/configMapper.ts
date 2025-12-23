/**
 * Configuration Mapper Utility
 *
 * Maps frontend simplified configuration to backend full configuration schema.
 * Provides sensible defaults for advanced settings that aren't exposed in the
 * basic UI while allowing full customization when needed.
 *
 * Dependencies:
 * - @/types/experiment: Frontend config types
 * - @/types/api: Backend config types
 */

import { ExperimentConfiguration } from '@/types/experiment';
import { ConfigurationUpdateRequest } from '@/types/api';

// ============================================================================
// Default Advanced Configuration
// ============================================================================

/**
 * Default values for advanced configuration options not shown in basic UI
 */
const ADVANCED_DEFAULTS = {
  // Model parameters
  dropout_rate: 0.3,
  freeze_backbone: true,
  num_classes: 1, // Binary classification
  monitor_metric: 'val_loss' as const,

  // Training parameters
  early_stopping_patience: 7,
  reduce_lr_patience: 3,
  reduce_lr_factor: 0.5,
  min_lr: 0.0000001,

  // System parameters
  device: 'auto' as const,
  num_workers: 0, // Windows compatibility

  // Image processing
  color_mode: 'RGB' as const,
  use_imagenet_norm: true,
  augmentation_strength: 1.0,
  use_custom_preprocessing: false,
  validate_images_on_init: true,
  pin_memory: true,
  persistent_workers: false,
  prefetch_factor: 2,

  // Custom preprocessing (disabled by default)
  contrast_stretch: true,
  adaptive_histogram: false,
  edge_enhancement: false,
  lower_percentile: 5.0,
  upper_percentile: 95.0,
  clip_limit: 2.0,
  edge_strength: 1.0,

  // System
  img_size: [256, 256] as [number, number],
  image_extension: '.png',
  sample_fraction: 0.05,
  validation_split: 0.20,
  seed: 42,
};

// ============================================================================
// Mapping Functions
// ============================================================================

/**
 * Map frontend simple configuration to backend full configuration
 *
 * @param frontendConfig - Simplified frontend configuration
 * @param overrides - Optional advanced configuration overrides
 * @returns Full backend configuration request
 */
export function mapToBackendConfig(
  frontendConfig: ExperimentConfiguration,
  overrides?: Partial<ConfigurationUpdateRequest>
): ConfigurationUpdateRequest {
  // Smart default: adjust early_stopping_patience based on epochs
  // For small epoch counts, use a minimum of 5 to allow exploration
  // For larger counts, use 50% of epochs (capped at 15)
  const calculatePatience = (epochs: number): number => {
    if (epochs <= 10) {
      // For small epoch counts, use at least 5 or epochs-1 (whichever is smaller)
      return Math.min(5, Math.max(3, epochs - 1));
    }
    // For larger epoch counts, use 50% (capped at 15)
    return Math.min(15, Math.floor(epochs / 2));
  };

  const smartEarlyStoppingPatience = frontendConfig.earlyStoppingPatience !== undefined
    ? frontendConfig.earlyStoppingPatience
    : calculatePatience(frontendConfig.epochs);

  console.log(`[ConfigMapper] Smart early stopping: epochs=${frontendConfig.epochs}, patience=${smartEarlyStoppingPatience}`);

  const backendConfig: ConfigurationUpdateRequest = {
    system: {
      img_size: overrides?.system?.img_size || ADVANCED_DEFAULTS.img_size,
      image_extension: overrides?.system?.image_extension || ADVANCED_DEFAULTS.image_extension,
      batch_size: frontendConfig.batchSize,
      sample_fraction: overrides?.system?.sample_fraction || ADVANCED_DEFAULTS.sample_fraction,
      validation_split: overrides?.system?.validation_split || ADVANCED_DEFAULTS.validation_split,
      seed: overrides?.system?.seed || ADVANCED_DEFAULTS.seed,
    },

    experiment: {
      // Basic parameters from frontend
      learning_rate: frontendConfig.learningRate,
      epochs: frontendConfig.epochs,
      batch_size: frontendConfig.batchSize,
      weight_decay: frontendConfig.weightDecay,
      fine_tune_layers_count: frontendConfig.fineTuneLayers,

      // Advanced parameters - use frontend values if provided, otherwise defaults
      dropout_rate: frontendConfig.dropoutRate ?? ADVANCED_DEFAULTS.dropout_rate,
      freeze_backbone: frontendConfig.freezeBackbone ?? ADVANCED_DEFAULTS.freeze_backbone,
      num_classes: ADVANCED_DEFAULTS.num_classes,
      monitor_metric: frontendConfig.monitorMetric || ADVANCED_DEFAULTS.monitor_metric,

      // Training parameters - use smart default for early stopping
      early_stopping_patience: smartEarlyStoppingPatience,
      reduce_lr_patience: frontendConfig.reduceLrPatience ?? ADVANCED_DEFAULTS.reduce_lr_patience,
      reduce_lr_factor: frontendConfig.reduceLrFactor ?? ADVANCED_DEFAULTS.reduce_lr_factor,
      min_lr: frontendConfig.minLr ?? ADVANCED_DEFAULTS.min_lr,
      validation_split: overrides?.experiment?.validation_split ?? overrides?.system?.validation_split ?? ADVANCED_DEFAULTS.validation_split,

      // Federated parameters (only if applicable)
      ...(frontendConfig.trainingMode === 'federated' || frontendConfig.trainingMode === 'both'
        ? {
            num_clients: frontendConfig.clients,
            num_rounds: frontendConfig.federatedRounds,
            local_epochs: frontendConfig.localEpochs,
            clients_per_round: frontendConfig.clients, // Use all clients
          }
        : {}),

      // System parameters
      device: frontendConfig.device || ADVANCED_DEFAULTS.device,
      num_workers: frontendConfig.numWorkers ?? ADVANCED_DEFAULTS.num_workers,

      // Image processing
      color_mode: frontendConfig.colorMode || ADVANCED_DEFAULTS.color_mode,
      use_imagenet_norm: frontendConfig.useImagenetNorm ?? ADVANCED_DEFAULTS.use_imagenet_norm,
      augmentation_strength: frontendConfig.augmentationStrength ?? ADVANCED_DEFAULTS.augmentation_strength,
      use_custom_preprocessing: frontendConfig.useCustomPreprocessing ?? ADVANCED_DEFAULTS.use_custom_preprocessing,
      validate_images_on_init: ADVANCED_DEFAULTS.validate_images_on_init,
      pin_memory: ADVANCED_DEFAULTS.pin_memory,
      persistent_workers: ADVANCED_DEFAULTS.persistent_workers,
      prefetch_factor: ADVANCED_DEFAULTS.prefetch_factor,

      // Custom preprocessing - use frontend values if custom preprocessing is enabled
      contrast_stretch: frontendConfig.useCustomPreprocessing
        ? (frontendConfig.contrastStretch ?? ADVANCED_DEFAULTS.contrast_stretch)
        : ADVANCED_DEFAULTS.contrast_stretch,
      adaptive_histogram: frontendConfig.useCustomPreprocessing
        ? (frontendConfig.adaptiveHistogram ?? ADVANCED_DEFAULTS.adaptive_histogram)
        : ADVANCED_DEFAULTS.adaptive_histogram,
      edge_enhancement: frontendConfig.useCustomPreprocessing
        ? (frontendConfig.edgeEnhancement ?? ADVANCED_DEFAULTS.edge_enhancement)
        : ADVANCED_DEFAULTS.edge_enhancement,
      lower_percentile: ADVANCED_DEFAULTS.lower_percentile,
      upper_percentile: ADVANCED_DEFAULTS.upper_percentile,
      clip_limit: ADVANCED_DEFAULTS.clip_limit,
      edge_strength: ADVANCED_DEFAULTS.edge_strength,
    },

    // Apply any additional overrides
    ...overrides,
  };

  return backendConfig;
}

/**
 * Generate experiment name based on configuration
 *
 * @param trainingMode - Training mode (centralized/federated/both)
 * @param timestamp - Optional timestamp (defaults to current time)
 * @returns Generated experiment name
 */
export function generateExperimentName(
  trainingMode: 'centralized' | 'federated' | 'both',
  timestamp?: Date
): string {
  const date = timestamp || new Date();
  const dateStr = date.toISOString().replace(/[:.]/g, '-').slice(0, 19);

  const modePrefix = {
    centralized: 'cent',
    federated: 'fed',
    both: 'comp',
  }[trainingMode];

  return `pneumonia_${modePrefix}_${dateStr}`;
}

/**
 * Validate frontend configuration before sending to backend
 *
 * @param config - Frontend configuration to validate
 * @returns Object with isValid flag and error messages
 */
export function validateConfiguration(config: ExperimentConfiguration): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // Validate basic parameters
  if (config.learningRate <= 0 || config.learningRate > 1) {
    errors.push('Learning rate must be between 0 and 1');
  }

  if (config.epochs < 1 || config.epochs > 1000) {
    errors.push('Epochs must be between 1 and 1000');
  }

  if (config.batchSize < 1 || config.batchSize > 512) {
    errors.push('Batch size must be between 1 and 512');
  }

  if (config.weightDecay < 0 || config.weightDecay > 1) {
    errors.push('Weight decay must be between 0 and 1');
  }

  if (config.fineTuneLayers < 0 || config.fineTuneLayers > 50) {
    errors.push('Fine-tune layers must be between 0 and 50');
  }

  // Validate federated parameters if applicable
  if (config.trainingMode === 'federated' || config.trainingMode === 'both') {
    if (!config.clients || config.clients < 2 || config.clients > 100) {
      errors.push('Number of clients must be between 2 and 100');
    }

    if (!config.federatedRounds || config.federatedRounds < 1 || config.federatedRounds > 100) {
      errors.push('Federated rounds must be between 1 and 100');
    }

    if (!config.localEpochs || config.localEpochs < 1 || config.localEpochs > 50) {
      errors.push('Local epochs must be between 1 and 50');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * Get recommended configuration based on training mode
 *
 * @param trainingMode - Training mode
 * @returns Recommended configuration
 */
export function getRecommendedConfig(
  trainingMode: 'centralized' | 'federated' | 'both'
): ExperimentConfiguration {
  const baseConfig: ExperimentConfiguration = {
    trainingMode,
    fineTuneLayers: 5,
    learningRate: 0.001,
    weightDecay: 0.0001,
    epochs: 10,
    batchSize: 128,
  };

  if (trainingMode === 'federated' || trainingMode === 'both') {
    return {
      ...baseConfig,
      clients: 3,
      federatedRounds: 3,
      localEpochs: 20,
    };
  }

  return baseConfig;
}

export default {
  mapToBackendConfig,
  generateExperimentName,
  validateConfiguration,
  getRecommendedConfig,
  ADVANCED_DEFAULTS,
};
