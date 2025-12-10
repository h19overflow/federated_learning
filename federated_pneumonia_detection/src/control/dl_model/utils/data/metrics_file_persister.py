import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import json


class MetricsFilePersister:
    """Handles file persistence (JSON/CSV) for training metrics."""

    def __init__(self, save_dir: str, experiment_name: str):
        """
        Initialize the metrics file persister.

        Args:
            save_dir: Directory to save metrics files
            experiment_name: Name of the experiment for file naming
        """
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        epoch_metrics: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Save metrics to JSON and CSV files.

        Args:
            epoch_metrics: List of epoch metric dictionaries
            metadata: Experiment metadata dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save epoch metrics as CSV for easy plotting
        if epoch_metrics:
            df = pd.DataFrame(epoch_metrics)
            csv_path = self.save_dir / f"{self.experiment_name}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV metrics to: {csv_path}")

        # Save metadata separately
        metadata_path = self.save_dir / f"{self.experiment_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to: {metadata_path}")
