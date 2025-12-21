"""
Configuration schema for comparative analysis experiments.

Defines Pydantic models for experiment configuration, statistical parameters,
and output settings used throughout the analysis pipeline.
"""

from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data source and preprocessing."""

    source_path: Path = Field(..., description="Path to ZIP file or directory with dataset")
    csv_filename: str = Field(
        default="stage2_train_metadata.csv",
        description="Metadata CSV filename inside archive",
    )
    image_dir_name: str = Field(default="Images", description="Images directory name")

    @field_validator("source_path", mode="before")
    @classmethod
    def validate_source_path(cls, v: str | Path) -> Path:
        return Path(v)


class ExperimentConfig(BaseModel):
    """Configuration for training experiments."""

    n_runs: int = Field(default=5, ge=1, le=20, description="Number of runs per approach")
    master_seed: int = Field(default=42, description="Master seed for reproducibility")
    epochs: int = Field(default=10, ge=1, description="Training epochs for centralized")
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")


class FederatedConfig(BaseModel):
    """Configuration specific to federated learning experiments."""

    num_clients: int = Field(default=2, ge=2, description="Number of federated clients")
    num_rounds: int = Field(default=5, ge=1, description="Number of federated rounds")
    local_epochs: int = Field(default=2, ge=1, description="Local epochs per round")


class StatisticalConfig(BaseModel):
    """Configuration for statistical analysis."""

    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level")
    confidence_level: float = Field(default=0.95, gt=0, lt=1, description="CI confidence level")
    n_bootstrap: int = Field(default=10000, ge=1000, description="Bootstrap iterations")
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1", "auroc"],
        description="Metrics to compare",
    )


class OutputConfig(BaseModel):
    """Configuration for output files and figures."""

    output_dir: Path = Field(default=Path("analysis_output"), description="Output directory")
    figure_dpi: int = Field(default=300, ge=72, description="Figure DPI for export")
    figure_format: Literal["png", "pdf", "eps"] = Field(default="png", description="Figure format")
    latex_output: bool = Field(default=True, description="Generate LaTeX tables")
    markdown_output: bool = Field(default=True, description="Generate Markdown report")

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v: str | Path) -> Path:
        return Path(v)


class AnalysisConfig(BaseModel):
    """Main configuration container for comparative analysis."""

    data: DataConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_source_path(cls, source_path: str | Path, **kwargs) -> "AnalysisConfig":
        """Create config with minimal required parameters."""
        data_config = DataConfig(source_path=source_path)
        return cls(data=data_config, **kwargs)

    def get_output_subdir(self, name: str) -> Path:
        """Get a subdirectory path within the output directory."""
        subdir = self.output.output_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir
