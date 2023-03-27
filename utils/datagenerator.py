import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):

    def __init__(self,
                 dataframes: list[pd.DataFrame],
                 input_shape: tuple,
                 batch_size: int,
                 target_col: str
                 ) -> None:
        self.dataframes = dataframes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.target_col = target_col

        self.num_samples = len(dataframes)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __getitem__(self, index: int) -> dict[np.array, np.array]:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch = self.dataframes[start_idx: end_idx]
        features, target = self.__getdata(batch)

        return {"st_maps": torch.tensor(features, dtype=torch.float),
                "target": torch.tensor(target, dtype=torch.float)}

    def __getdata(self, batch: list[pd.DataFrame]
                  ) -> tuple[np.array, np.array]:
        features = []
        for data in batch:
            feature_columns = [col for col in list(data.columns) if self.is_yuv_column(col)]

            features.append(data[feature_columns]
                            .to_numpy().reshape(self.input_shape))

        targets = [np.average(data[self.target_col].astype(float))
                   for data in batch]

        features = np.array(features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return features, targets

    @staticmethod
    def is_yuv_column(col):
        return col.startswith('Y') or col.startswith('U') or col.startswith('V')

    def on_epoch_end(self) -> None:
        random.shuffle(self.dataframes)


def collate_fn(batch):
    return batch


def get_data(st_maps, window_size, batch_size, predicted_column, num_workers=0):
    """A function that returns a DataLoader for the given data."""
    generator = DataGenerator(st_maps,
                              input_shape=(window_size, 25, 3),
                              batch_size=batch_size,
                              target_col=predicted_column)

    return torch.utils.data.DataLoader(dataset=generator,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=False,
                                       collate_fn=collate_fn)
