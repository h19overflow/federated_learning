"""
Seed management for reproducible experiments.

Generates deterministic seed sequences from a master seed to ensure
consistent random states across centralized and federated experiments.
"""

import hashlib
import logging
from typing import List, Optional


class SeedManager:
    """
    Manages reproducible seed sequences for experiments.

    Generates the same sequence of seeds from a master seed,
    ensuring both centralized and federated experiments use
    identical random states for fair comparison.
    """

    def __init__(
        self,
        master_seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize seed manager.

        Args:
            master_seed: Master seed for sequence generation
            logger: Optional logger instance
        """
        self.master_seed = master_seed
        self.logger = logger or logging.getLogger(__name__)
        self._sequence: Optional[List[int]] = None

    def generate_sequence(self, n_seeds: int) -> List[int]:
        """
        Generate deterministic seed sequence.

        Uses hash-based generation to create reproducible seeds
        that are well-distributed and avoid correlation.

        Args:
            n_seeds: Number of seeds to generate

        Returns:
            List of integer seeds
        """
        seeds = []
        for i in range(n_seeds):
            seed_input = f"{self.master_seed}_{i}"
            hash_value = hashlib.sha256(seed_input.encode()).hexdigest()
            seed = int(hash_value[:8], 16) % (2**31)
            seeds.append(seed)

        self._sequence = seeds
        self.logger.info(f"Generated {n_seeds} seeds from master seed {self.master_seed}")
        return seeds

    def get_seed(self, run_index: int) -> int:
        """
        Get seed for a specific run.

        Args:
            run_index: Zero-indexed run number

        Returns:
            Seed value for the run
        """
        if self._sequence is None:
            raise ValueError("Seed sequence not generated. Call generate_sequence first.")

        if run_index >= len(self._sequence):
            raise IndexError(f"Run index {run_index} exceeds sequence length {len(self._sequence)}")

        return self._sequence[run_index]

    @property
    def sequence(self) -> Optional[List[int]]:
        """Get generated seed sequence."""
        return self._sequence

    def to_dict(self) -> dict:
        """Export seed configuration for reproducibility documentation."""
        return {
            "master_seed": self.master_seed,
            "sequence": self._sequence,
            "n_seeds": len(self._sequence) if self._sequence else 0,
        }
