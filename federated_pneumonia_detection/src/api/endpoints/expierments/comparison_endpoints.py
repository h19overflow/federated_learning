"""
Endpoints for running comparison experiments between centralized and federated learning.

This module provides HTTP endpoints to trigger comparative analysis experiments that
run both centralized and federated training approaches on the same dataset and generate
comparison reports. The experiments are executed asynchronously in the background.

The comparison experiment process:
1. Initializes ExperimentOrchestrator with configured partition strategy
2. Runs centralized training on the full dataset
3. Runs federated learning with configured clients and rounds
4. Compares performance metrics between approaches
5. Generates comprehensive comparison reports and visualizations
6. Stores all results and comparison data in organized output directories

Configuration should be set prior to invoking experiments via the configuration endpoints.
Key comparison parameters: partition_strategy, num_clients, num_rounds, learning_rate
"""

import logging
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from typing import Dict, Any
import os
import shutil

from federated_pneumonia_detection.src.utils.logger import get_logger
from federated_pneumonia_detection.src.control.comparison.experiment_orchestrator import ExperimentOrchestrator

router = APIRouter(
    prefix="/experiments/comparison",
    tags=["experiments", "comparison"],
)

logger = get_logger(__name__)


def _run_comparison_task(
    source_path: str,
    partition_strategy: str,
    base_output_dir: str,
    centralized_name: str,
    federated_name: str,
) -> Dict[str, Any]:
    """
    Background task to execute comparison experiment.
    
    Args:
        source_path: Path to training data directory
        partition_strategy: Strategy for partitioning data (e.g., 'stratified')
        base_output_dir: Base directory for storing all experiment outputs
        centralized_name: Name identifier for centralized training run
        federated_name: Name identifier for federated training run
        
    Returns:
        Dictionary containing comparison results
    """
    task_logger = get_logger(f"{__name__}._task")
    
    task_logger.info("=" * 80)
    task_logger.info("COMPARISON EXPERIMENT - Centralized vs Federated Learning")
    task_logger.info("(Background Task)")
    task_logger.info("=" * 80)
    
    try:
        task_logger.info(f"\nInitializing ExperimentOrchestrator...")
        task_logger.info(f"  Source: {source_path}")
        task_logger.info(f"  Partition strategy: {partition_strategy}")
        task_logger.info(f"  Output directory: {base_output_dir}")
        
        orchestrator = ExperimentOrchestrator(
            config_path=None,
            base_output_dir=base_output_dir,
            partition_strategy=partition_strategy
        )
        
        exp_info = orchestrator.get_experiment_info()
        task_logger.info("\nExperiment Configuration:")
        task_logger.info(f"  Timestamp: {exp_info['timestamp']}")
        task_logger.info(f"  Experiment Directory: {exp_info['experiment_dir']}")
        task_logger.info(f"  Partition Strategy: {exp_info['partition_strategy']}")
        
        task_logger.info("\n" + "=" * 80)
        task_logger.info("Starting comparison experiment...")
        task_logger.info("This will run both centralized and federated training")
        task_logger.info("=" * 80 + "\n")
        
        comparison_results = orchestrator.run_comparison(
            source_path=source_path,
            centralized_name=centralized_name,
            federated_name=federated_name
        )
        
        task_logger.info("\n" + "=" * 80)
        task_logger.info("COMPARISON EXPERIMENT COMPLETED!")
        task_logger.info("=" * 80)
        
        task_logger.info("\nDetailed Results:")
        task_logger.info("-" * 80)
        
        cent_results = comparison_results['centralized']
        task_logger.info("\nCentralized Training:")
        task_logger.info(f"  Status: {cent_results['status']}")
        task_logger.info(f"  Output Directory: {cent_results['output_dir']}")
        if cent_results['status'] == 'success':
            task_logger.info("  ✓ Training completed successfully")
        else:
            task_logger.error(f"  ✗ Training failed: {cent_results.get('error', 'Unknown error')}")
        
        fed_results = comparison_results['federated']
        task_logger.info("\nFederated Learning:")
        task_logger.info(f"  Status: {fed_results['status']}")
        task_logger.info(f"  Output Directory: {fed_results['output_dir']}")
        if fed_results['status'] == 'success':
            task_logger.info("  ✓ Training completed successfully")
        else:
            task_logger.error(f"  ✗ Training failed: {fed_results.get('error', 'Unknown error')}")
        
        if 'comparison_metrics' in comparison_results:
            task_logger.info("\nComparison Metrics:")
            comp_metrics = comparison_results['comparison_metrics']
            
            if 'centralized_metrics' in comp_metrics:
                task_logger.info("\n  Centralized Metrics:")
                for key, value in comp_metrics['centralized_metrics'].items():
                    task_logger.info(f"    {key}: {value}")
            
            if 'federated_metrics' in comp_metrics:
                task_logger.info("\n  Federated Metrics:")
                for key, value in comp_metrics['federated_metrics'].items():
                    task_logger.info(f"    {key}: {value}")
        
        task_logger.info("\n" + "=" * 80)
        task_logger.info(f"All results saved to: {comparison_results['experiment_dir']}")
        task_logger.info("=" * 80)
        
        return comparison_results
        
    except Exception as e:
        task_logger.error("\n" + "=" * 80)
        task_logger.error("COMPARISON EXPERIMENT FAILED!")
        task_logger.error("=" * 80)
        task_logger.error(f"Error: {type(e).__name__}: {str(e)}")
        
        import traceback
        task_logger.error("\nFull traceback:")
        task_logger.error(traceback.format_exc())
        
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }


@router.post("/run")
async def start_comparison_experiment(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
    partition_strategy: str = Form("stratified"),
    base_output_dir: str = Form("experiments"),
    centralized_name: str = Form("pneumonia_centralized"),
    federated_name: str = Form("pneumonia_federated"),
) -> Dict[str, Any]:
    """
    Start comparison experiment between centralized and federated learning with uploaded data.
    
    Initiates a comprehensive comparison experiment that runs both centralized and
    federated training approaches on the same dataset using current configuration settings.
    The experiment runs asynchronously, allowing this endpoint to return immediately.
    
    **Experiment Overview:**
    This endpoint orchestrates a full experimental comparison including:
    - Running centralized supervised learning on the entire dataset
    - Running federated learning with data partitioned across virtual clients
    - Comparing performance metrics between both approaches
    - Generating detailed comparison reports and analysis
    
    **Centralized Approach:**
    - All training data processed by single trainer
    - Standard supervised learning with data augmentation
    - Single model trained end-to-end
    
    **Federated Approach:**
    - Data partitioned across num_clients virtual clients using partition_strategy
    - Each client trains locally with num_rounds communication rounds
    - Global model aggregated using federated averaging (FedAvg)
    - Models compared without centralizing training data
    
    **Output Structure:**
    ```
    {base_output_dir}/{timestamp}/
    ├── centralized/
    │   ├── checkpoints/
    │   ├── logs/
    │   └── results.json
    ├── federated/
    │   ├── checkpoints/
    │   ├── logs/
    │   └── results.json
    └── comparison/
        ├── metrics_comparison.json
        └── report.html
    ```
    
    **Prerequisites:**
    - Configuration should be set via `/configuration/set_configuration` endpoint
    - Upload a ZIP file containing Images/ directory and metadata CSV
    - Key federated parameters: num_clients, num_rounds, local_epochs
    
    **Parameters:**
    - `data_zip`: ZIP file containing Images/ directory and metadata CSV (required)
    - `partition_strategy`: Data partitioning strategy - 'stratified' (maintains class distribution) or 'random'
      (default: "stratified")
    - `base_output_dir`: Base directory for experiment outputs (default: "experiments")
    - `centralized_name`: Experiment name for centralized run (default: "pneumonia_centralized")
    - `federated_name`: Experiment name for federated run (default: "pneumonia_federated")
    
    **Response:**
    Returns immediately with confirmation that comparison has been queued. Check logs
    and results directories for progress and detailed metrics.
    
    **Configuration Recommendations:**
    
    For Centralized Training:
    - epochs: 15-50
    - batch_size: 256-512
    - learning_rate: 0.001-0.01
    
    For Federated Learning:
    - num_clients: 2-5 (for fair comparison with centralized)
    - num_rounds: 10-20
    - local_epochs: 1-5 (1 epoch per round for fair sample usage comparison)
    - learning_rate: 0.0005-0.005 (typically lower than centralized)
    
    **Partition Strategies:**
    - `stratified`: Maintains class distribution in each client's data (recommended)
    - `random`: Random sampling for each client (less realistic for heterogeneous data)
    
    **Status Tracking:**
    Monitor experiment progress through:
    - Individual training logs in centralized/ and federated/ directories
    - Comparison metrics in comparison/
    - Final HTML report with visualizations
    """
    import zipfile
    import tempfile
    
    temp_dir = None
    try:
        # Create temp directory for extraction
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, data_zip.filename)
        
        # Save uploaded file
        with open(zip_path, "wb") as f:
            content = await data_zip.read()
            f.write(content)
        
        # Extract archive
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        source_path = extract_path
        
        logger.info(f"Received request to start comparison experiment")
        logger.info(f"  Partition strategy: {partition_strategy}")
        logger.info(f"  Output directory: {base_output_dir}")
        logger.info(f"  Extracted data to: {source_path}")
        
        background_tasks.add_task(
            _run_comparison_task,
            source_path=source_path,
            partition_strategy=partition_strategy,
            base_output_dir=base_output_dir,
            centralized_name=centralized_name,
            federated_name=federated_name,
        )
        
        return {
            "message": "Comparison experiment started successfully",
            "base_output_dir": base_output_dir,
            "partition_strategy": partition_strategy,
            "centralized_name": centralized_name,
            "federated_name": federated_name,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
