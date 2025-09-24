"""
Complete example showing the new streamlined data pipeline.
Demonstrates the flow from CSV → DataFrames → XRayDataModule → CustomImageDataset
"""

import logging
from pathlib import Path

# Configure logging to see the pipeline in action
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from federated_pneumonia_detection.src.entities import SystemConstants, ExperimentConfig
from federated_pneumonia_detection.src.utils.data_processing import load_and_split_data, get_image_directory_path
from federated_pneumonia_detection.src.control import XRayDataModule
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader


def main():
    """Demonstrate the complete data pipeline."""

    print("=== Federated Pneumonia Detection - Data Pipeline Example ===\n")

    # Step 1: Load configuration
    print("1. Loading configuration...")
    config_loader = ConfigLoader()

    # You can either load from config file or create manually
    try:
        # Load from YAML file
        constants = config_loader.create_system_constants()
        config = config_loader.create_experiment_config()
        print(f"   ✅ Loaded configuration from YAML")
    except FileNotFoundError:
        # Create manually if no config file exists
        constants = SystemConstants.create_custom(
            img_size=(224, 224),
            batch_size=32,
            sample_fraction=0.1,  # Use small sample for demo
            validation_split=0.2,
            seed=42
        )

        config = ExperimentConfig(
            batch_size=32,
            sample_fraction=0.1,
            validation_split=0.2,
            seed=42,
            use_custom_preprocessing=True,
            augmentation_strength=0.5
        )
        print(f"   ✅ Created default configuration")

    print(f"   📊 Sample fraction: {config.sample_fraction}")
    print(f"   🔄 Validation split: {config.validation_split}")
    print(f"   🖼️  Image size: {constants.IMG_SIZE}")
    print()

    # Step 2: Load and split data
    print("2. Loading and splitting metadata...")
    try:
        train_df, val_df = load_and_split_data(constants, config)

        print(f"   ✅ Data loaded successfully")
        print(f"   📈 Training samples: {len(train_df)}")
        print(f"   📊 Validation samples: {len(val_df)}")

        # Show class distribution
        train_dist = train_df[constants.TARGET_COLUMN].value_counts().to_dict()
        val_dist = val_df[constants.TARGET_COLUMN].value_counts().to_dict()
        print(f"   🎯 Train class distribution: {train_dist}")
        print(f"   🎯 Val class distribution: {val_dist}")

    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        print(f"   💡 Make sure your metadata file exists at: {constants.BASE_PATH}/{constants.METADATA_FILENAME}")
        return
    print()

    # Step 3: Get image directory
    print("3. Setting up image directory...")
    image_dir = get_image_directory_path(constants)
    print(f"   📁 Image directory: {image_dir}")

    if not Path(image_dir).exists():
        print(f"   ⚠️  Warning: Image directory doesn't exist")
        print(f"   💡 Create the directory structure or update your paths in configuration")
    else:
        print(f"   ✅ Image directory exists")
    print()

    # Step 4: Create DataModule
    print("4. Creating XRayDataModule...")

    # Get custom preprocessing config if enabled
    custom_preprocessing = config.get_custom_preprocessing_config() if config.use_custom_preprocessing else None

    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        constants=constants,
        config=config,
        image_dir=image_dir,
        color_mode=config.color_mode,
        custom_preprocessing_config=custom_preprocessing
    )

    print(f"   ✅ DataModule created")
    print(f"   🎨 Color mode: {config.color_mode}")
    print(f"   🔧 Custom preprocessing: {'Enabled' if config.use_custom_preprocessing else 'Disabled'}")
    print(f"   💪 Augmentation strength: {config.augmentation_strength}")
    print()

    # Step 5: Setup datasets
    print("5. Setting up datasets...")
    try:
        data_module.setup('fit')

        # Get dataset statistics
        stats = data_module.get_data_statistics()
        print(f"   ✅ Datasets created successfully")
        print(f"   📊 Train dataset: {stats['train_samples']} valid samples")
        print(f"   📊 Val dataset: {stats['val_samples']} valid samples")
        print(f"   💾 Estimated memory usage: {stats.get('train_memory_estimate_mb', 0):.1f} MB")

    except Exception as e:
        print(f"   ❌ Error setting up datasets: {e}")
        print(f"   💡 This is likely due to missing image files")
        print(f"   💡 The pipeline structure is working, but you need actual image data")
        return
    print()

    # Step 6: Create data loaders
    print("6. Testing data loaders...")
    try:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        print(f"   ✅ Data loaders created")
        print(f"   🔄 Train batches: {len(train_loader)}")
        print(f"   🔄 Val batches: {len(val_loader)}")
        print(f"   📦 Batch size: {config.batch_size}")

        # Try to get a sample batch (only if images exist)
        if Path(image_dir).exists() and any(Path(image_dir).iterdir()):
            try:
                sample_batch = data_module.get_sample_batch('train', batch_size=4)
                print(f"   ✅ Sample batch loaded")
                print(f"   🖼️  Image tensor shape: {sample_batch['image_shape']}")
                print(f"   🎯 Label distribution in batch: {sample_batch['label_distribution']}")
            except Exception as e:
                print(f"   ⚠️  Could not load sample batch: {e}")

    except Exception as e:
        print(f"   ❌ Error creating data loaders: {e}")
    print()

    # Step 7: Show the complete workflow summary
    print("7. 🎉 Pipeline Summary")
    print("   ┌─────────────────────────────────────────┐")
    print("   │              NEW WORKFLOW               │")
    print("   ├─────────────────────────────────────────┤")
    print("   │ 1. CSV → load_and_split_data()         │")
    print("   │ 2. train_df, val_df                    │")
    print("   │ 3. XRayDataModule(train_df, val_df)    │")
    print("   │ 4. CustomImageDataset (auto-created)   │")
    print("   │ 5. DataLoader with transforms           │")
    print("   └─────────────────────────────────────────┘")
    print()
    print("✨ Benefits of the new approach:")
    print("   • Cleaner separation of concerns")
    print("   • More configurable and extensible")
    print("   • Better error handling and validation")
    print("   • Integrated with PyTorch Lightning")
    print("   • Advanced image preprocessing options")
    print()

    return data_module


if __name__ == "__main__":
    data_module = main()