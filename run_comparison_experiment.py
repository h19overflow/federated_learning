"""
Script to run comparison experiment between centralized and federated learning.
Uses the ExperimentOrchestrator to run both approaches and generate comparison report.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from federated_pneumonia_detection.src.control.comparison.experiment_orchestrator import ExperimentOrchestrator

# TODO , Figure out how to extract logs from the trainer in order to show in the terminal on the frontend.
def main():
    """Run comparison experiment between centralized and federated learning."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("COMPARISON EXPERIMENT - Centralized vs Federated Learning")
    logger.info("="*80)

    # Define paths
    source_path = "Training"  # Directory containing Images/ and CSV files
    config_path = None  # Use default configuration
    partition_strategy = "stratified"  # For federated learning
    base_output_dir = "experiments"  # Base directory for all experiment outputs

    logger.info(f"Source path: {source_path}")
    logger.info(f"Partition strategy (FL): {partition_strategy}")
    logger.info(f"Base output directory: {base_output_dir}")

    # Create orchestrator
    logger.info("\nInitializing ExperimentOrchestrator...")
    orchestrator = ExperimentOrchestrator(
        config_path=config_path,
        base_output_dir=base_output_dir,
        partition_strategy=partition_strategy
    )

    # Display experiment info
    exp_info = orchestrator.get_experiment_info()
    logger.info("\nExperiment Configuration:")
    logger.info(f"  Timestamp: {exp_info['timestamp']}")
    logger.info(f"  Experiment Directory: {exp_info['experiment_dir']}")
    logger.info(f"  Partition Strategy: {exp_info['partition_strategy']}")

    # Run comparison
    try:
        logger.info("\n" + "="*80)
        logger.info("Starting comparison experiment...")
        logger.info("This will run both centralized and federated training")
        logger.info("="*80 + "\n")

        # Run the comparison
        comparison_results = orchestrator.run_comparison(
            source_path=source_path,
            centralized_name="pneumonia_centralized",
            federated_name="pneumonia_federated"
        )

        logger.info("\n" + "="*80)
        logger.info("COMPARISON EXPERIMENT COMPLETED!")
        logger.info("="*80)

        # Display detailed results
        logger.info("\nDetailed Results:")
        logger.info("-"*80)

        # Centralized results
        cent_results = comparison_results['centralized']
        logger.info("\nCentralized Training:")
        logger.info(f"  Status: {cent_results['status']}")
        logger.info(f"  Output Directory: {cent_results['output_dir']}")
        if cent_results['status'] == 'success':
            logger.info("  ✓ Training completed successfully")
        else:
            logger.error(f"  ✗ Training failed: {cent_results.get('error', 'Unknown error')}")

        # Federated results
        fed_results = comparison_results['federated']
        logger.info("\nFederated Learning:")
        logger.info(f"  Status: {fed_results['status']}")
        logger.info(f"  Output Directory: {fed_results['output_dir']}")
        if fed_results['status'] == 'success':
            logger.info("  ✓ Training completed successfully")
        else:
            logger.error(f"  ✗ Training failed: {fed_results.get('error', 'Unknown error')}")

        # Comparison metrics
        if 'comparison_metrics' in comparison_results:
            logger.info("\nComparison Metrics:")
            comp_metrics = comparison_results['comparison_metrics']

            if 'centralized_metrics' in comp_metrics:
                logger.info("\n  Centralized Metrics:")
                for key, value in comp_metrics['centralized_metrics'].items():
                    logger.info(f"    {key}: {value}")

            if 'federated_metrics' in comp_metrics:
                logger.info("\n  Federated Metrics:")
                for key, value in comp_metrics['federated_metrics'].items():
                    logger.info(f"    {key}: {value}")

        logger.info("\n" + "="*80)
        logger.info(f"All results saved to: {comparison_results['experiment_dir']}")
        logger.info("="*80)

        # Return success if both trainings succeeded
        if (cent_results['status'] == 'success' and fed_results['status'] == 'success'):
            return 0
        else:
            logger.warning("\nNote: Some training methods failed. Check logs for details.")
            return 1

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("COMPARISON EXPERIMENT FAILED!")
        logger.error("="*80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
