import numpy as np
import pandas as pd
from pandas import DataFrame
from flwr_datasets.partitioner import Partitioner
class CustomPartitioner(Partitioner):
    def __init__(self, base_dataset, num_partitions: int):
        super().__init__()
        self._base_dataset = base_dataset  # Your full PyTorch Dataset instance
        self._num_partitions = num_partitions

        total_size = len(base_dataset)
        indices = np.random.permutation(total_size)  # Shuffle
        self.partition_indices = np.array_split(indices, num_partitions)  # Evenly split indices

    @property
    def num_partitions(self):
        return self._num_partitions

    def load_partition(self, partition_id: int)->DataFrame:
        
        assert 0 <= partition_id < self._num_partitions
        partition_idx = self.partition_indices[partition_id]
        # Return a Subset of your base Dataset
        return self._base_dataset.iloc[partition_idx].reset_index(drop=True)
