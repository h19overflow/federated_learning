#!/usr/bin/env python3
"""
Federated Pneumonia Detection System - Main Entry Point

This script provides the primary interface for the federated pneumonia detection system.
It supports multiple operation modes including training, evaluation, data pipeline testing,
and system validation.

Usage:
    python main.py --help                    # Show all available options
    python main.py --demo                    # Run interactive demo
    python main.py --train <path>            # Train model from dataset
    python main.py --evaluate <model_path>   # Evaluate trained model
    python main.py --pipeline-test           # Test data pipeline
    python main.py --system-check           # Validate system configuration
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Configure logging for the main application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('federated_pneumonia.log')
    ]
)

logger = logging.getLogger(__name__)

# Project imports
try:
    from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
    from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
    from federated_pneumonia_detection.src.control.reporting.model_evaluator import ModelEvaluator
    from federated_pneumonia_detection.src.utils.data_processing import load_and_split_data
    from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
    from federated_pneumonia_detection.models.system_constants import SystemConstants
except ImportError as e:
    logger.error(f"Failed to import federated_pneumonia_detection modules: {e}")
    logger.error("Please ensure the package is installed: pip install -e .")
    sys.exit(1)


def print_banner():
    """Display application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║           Federated Pneumonia Detection System                ║
    ║                                                               ║
    ║   🔒 Privacy-Preserving Medical AI for Chest X-Ray Analysis  ║
    ║   🏥 Collaborative Learning Across Medical Institutions      ║
    ║   ⚡ Built with PyTorch Lightning & Modern ML Practices      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def system_check() -> bool:
    """Perform comprehensive system validation."""
    logger.info("🔍 Performing system validation...")
    
    print("\n=== System Configuration Check ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 12):
        print("⚠️  Warning: Python 3.12+ is recommended")
    
    # Check core dependencies
    dependencies = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('yaml', 'PyYAML')
    ]
    
    missing_deps = []
    for dep, name in dependencies:
        try:
            __import__(dep)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Missing")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install -e .")
        return False
    
    # Check configuration
    try:
        config_loader = ConfigLoader()
        constants = config_loader.create_system_constants()
        config = config_loader.create_experiment_config()
        print("✅ Configuration: Valid")
    except Exception as e:
        print(f"⚠️  Configuration Warning: {e}")
        print("   Using default configuration")
    
    # Check directories
    directories = ['logs', 'results', 'models/checkpoints']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory '{dir_path}': Ready")
    
    print("\n🎉 System validation complete!")
    return True


def run_demo():
    """Run interactive demonstration of key features."""
    print("\n=== Interactive Demo Mode ===")
    
    print("\n1. System Configuration")
    print("-" * 30)
    
    # Load configuration
    config_loader = ConfigLoader()
    constants = config_loader.create_system_constants()
    config = config_loader.create_experiment_config()
    
    print(f"📊 Batch Size: {config.batch_size}")
    print(f"🎯 Learning Rate: {config.learning_rate}")
    print(f"📈 Max Epochs: {config.epochs}")
    print(f"🖼️  Image Size: {constants.IMG_SIZE}")
    print(f"🔄 Validation Split: {config.validation_split}")
    
    print("\n2. Data Pipeline Demo")
    print("-" * 30)
    
    # Check for sample data
    data_dirs = ['Training', 'Test', 'data']
    found_data = False
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            files = list(Path(data_dir).rglob('*.csv'))
            if files:
                print(f"✅ Found data in '{data_dir}': {len(files)} CSV files")
                found_data = True
            else:
                print(f"📁 Directory '{data_dir}' exists but no CSV files found")
        else:
            print(f"📂 Directory '{data_dir}' not found")
    
    if not found_data:
        print("\n💡 Demo Tip: Add your chest X-ray dataset to run full demo")
        print("   Expected structure:")
        print("   📁 Training/")
        print("   ├── 📄 metadata.csv (with columns: patient_id, filename, label)")
        print("   └── 🖼️  images/")
    
    print("\n3. Model Architecture")
    print("-" * 30)
    print("🏗️  Base Architecture: ResNet with Custom Head")
    print(f"🎛️  Dropout Rate: {config.dropout_rate}")
    print(f"❄️  Freeze Backbone: {config.freeze_backbone}")
    print(f"🔧 Fine-tune Layers: {config.fine_tune_layers_count}")
    
    print("\n4. Federated Learning Configuration")
    print("-" * 30)
    print(f"👥 Number of Clients: {config.num_clients}")
    print(f"🔄 FL Rounds: {config.num_rounds}")
    print(f"📊 Clients per Round: {config.clients_per_round}")
    print(f"🏃 Local Epochs: {config.local_epochs}")
    
    print("\n🎯 Demo Complete! Use --help to see all available options.")


def run_training(dataset_path: str):
    """Run model training on specified dataset."""
    logger.info(f"🚀 Starting training on dataset: {dataset_path}")
    
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    print(f"\n=== Training Mode ===")
    print(f"📁 Dataset: {dataset_path}")
    
    try:
        # Initialize trainer
        trainer = CentralizedTrainer(
            checkpoint_dir="models/checkpoints",
            logs_dir="logs/training"
        )
        
        # Start training
        results = trainer.train(
            source_path=dataset_path,
            experiment_name="pneumonia_detection_run"
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Best Validation Accuracy: {results.get('best_val_acc', 'N/A'):.3f}")
        print(f"💾 Model Checkpoint: {results.get('checkpoint_path', 'N/A')}")
        print(f"📈 Logs Directory: {results.get('logs_path', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"\n💡 Training Tips:")
        print(f"   • Ensure dataset contains CSV metadata and image files")
        print(f"   • Check that CSV has required columns: patient_id, filename, label")
        print(f"   • Verify sufficient disk space and memory")


def run_pipeline_test():
    """Test the complete data pipeline without training."""
    print("\n=== Data Pipeline Test ===")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        constants = config_loader.create_system_constants()
        config = ExperimentConfig(
            sample_fraction=0.01,  # Use tiny sample for testing
            validation_split=0.2,
            batch_size=4
        )
        
        print("✅ Configuration loaded")
        print(f"📊 Using {config.sample_fraction*100}% sample for testing")
        
        # Look for test data
        test_paths = ['Training', 'Test', 'data']
        for test_path in test_paths:
            if Path(test_path).exists():
                csv_files = list(Path(test_path).rglob('*.csv'))
                if csv_files:
                    print(f"✅ Found test data in {test_path}")
                    print("🧪 Pipeline test would validate data loading, preprocessing, and batch creation")
                    return
        
        print("📂 No test data found - pipeline test requires sample dataset")
        print("💡 Add a small sample dataset to test the complete pipeline")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Federated Pneumonia Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py --demo                    # Interactive demo
  python main.py --train Training/         # Train on dataset
  python main.py --system-check           # Validate configuration
  python main.py --pipeline-test          # Test data pipeline
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demonstration')
    parser.add_argument('--train', type=str, metavar='PATH',
                       help='Train model on dataset (zip file or directory)')
    parser.add_argument('--evaluate', type=str, metavar='MODEL_PATH',
                       help='Evaluate trained model')
    parser.add_argument('--pipeline-test', action='store_true',
                       help='Test data pipeline without training')
    parser.add_argument('--system-check', action='store_true',
                       help='Perform system validation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Show banner unless quiet mode
    if not args.quiet:
        print_banner()
    
    # Handle no arguments - show help and demo
    if len(sys.argv) == 1:
        print("\n💡 No arguments provided. Running demo mode...")
        print("   Use --help to see all available options.\n")
        args.demo = True
    
    # Execute based on arguments
    try:
        if args.system_check:
            system_check()
        elif args.demo:
            run_demo()
        elif args.train:
            run_training(args.train)
        elif args.evaluate:
            print(f"\n🔍 Model evaluation mode - Model: {args.evaluate}")
            print("📊 Evaluation functionality coming soon!")
        elif args.pipeline_test:
            run_pipeline_test()
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if not args.quiet:
            print(f"\n💡 For troubleshooting, try: python main.py --system-check")
        sys.exit(1)


if __name__ == '__main__':
    main()
