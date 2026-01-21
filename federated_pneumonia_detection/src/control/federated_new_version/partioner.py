import numpy as np
from flwr_datasets.partitioner import Partitioner
from pandas import DataFrame


class CustomPartitioner(Partitioner):
    def __init__(self, base_dataset, num_partitions: int):
        super().__init__()
        self._base_dataset = base_dataset  # Your full PyTorch Dataset instance
        self._num_partitions = num_partitions

        total_size = len(base_dataset)
        indices = np.random.permutation(total_size)  # Shuffle
        self.partition_indices = np.array_split(
            indices,
            num_partitions,
        )  # Evenly split indices

    @property
    def num_partitions(self):
        return self._num_partitions

    def load_partition(self, partition_id: int) -> DataFrame:
        if not (0 <= partition_id < self._num_partitions):
            raise ValueError(
                f"partition_id must be between 0 and {self._num_partitions - 1}, got {partition_id}",
            )
        partition_idx = self.partition_indices[partition_id]
        # Return a Subset of your base Dataset
        sliced_df = self._base_dataset.iloc[partition_idx].reset_index(drop=True)
        if "filename" not in sliced_df.columns and "patientId" in sliced_df.columns:
            sliced_df["filename"] = sliced_df.apply(
                lambda x: str(x["patientId"]) + ".png",
                axis=1,
            )
        return sliced_df
