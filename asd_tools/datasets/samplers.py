import random
import numpy as np
from torch.utils.data.sampler import BatchSampler
import logging


class OutlierBalancedBatchSampler(BatchSampler):
    """BatchSampler = positive:negative+outlier."""

    def __init__(
        self,
        dataset,
        n_pos=32,
        n_neg=32,
        shuffle=False,
        drop_last=False,
        anomaly_as_neg=True,
        n_anomaly=0,
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            n_pos (int, optional): The number of positive sample in the mini-batch. Defaults to 32.
            n_neg (int, optional): The number of negative sample in the mini-batch. Defaults to 32.
            shuffle (bool, optional): Shuffle. Defaults to False.
            drop_last (bool, optional): Drop last. Defaults to False.
            anomaly_as_neg (bool, optional): Anomaly as negative. Defaults to True.
            n_anomaly (int, optional): The number of anomaly sample in the mini-batch. Defaults to 1
        """
        self.n_pos_file = len(dataset.pos_files)
        self.n_neg_file = len(dataset.neg_files) + len(dataset.outlier_files)
        self.n_anomaly_file = len(dataset.pos_anomaly_files)
        self.anomaly_as_neg = anomaly_as_neg
        if anomaly_as_neg:
            self.anomaly_idx = np.arange(
                self.n_pos_file,
                self.n_pos_file + self.n_anomaly_file,
            )
        else:
            self.n_pos_file += len(dataset.pos_anomaly_files)
            self.n_anomaly_file = 0
        self.pos_idx = np.arange(self.n_pos_file)
        self.neg_idx = np.arange(
            self.n_pos_file + self.n_anomaly_file,
            self.n_pos_file + self.n_anomaly_file + self.n_neg_file,
        )
        self.used_idx_cnt = {"pos": 0, "neg": 0}
        self.count = 0
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_anomaly = n_anomaly
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            np.random.shuffle(self.pos_idx)
            np.random.shuffle(self.neg_idx)
        while self.count < self.n_pos_file:
            indices = []
            indices.extend(
                self.pos_idx[
                    self.used_idx_cnt["pos"] : self.used_idx_cnt["pos"] + self.n_pos
                ]
            )
            self.used_idx_cnt["pos"] += self.n_pos
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_neg
                ]
            )
            self.used_idx_cnt["neg"] += self.n_neg
            if (
                self.anomaly_as_neg
                and (self.n_anomaly > 0)
                and (self.n_anomaly_file > 0)
            ):
                indices.extend(np.random.choice(self.anomaly_idx, self.n_anomaly))
            if self.shuffle:
                random.shuffle(indices)
            yield indices
            self.count += self.n_pos
        if not self.drop_last:
            indices = []
            indices.extend(self.pos_idx[self.used_idx_cnt["pos"] :])
            indices.extend(
                self.neg_idx[
                    self.used_idx_cnt["neg"] : self.used_idx_cnt["neg"] + self.n_neg
                ]
            )
            if (
                self.anomaly_as_neg
                and (self.n_anomaly > 0)
                and (self.n_anomaly_file > 0)
            ):
                indices.extend(np.random.choice(self.anomaly_idx, self.n_anomaly))
            yield indices
        if self.used_idx_cnt["pos"] + self.n_pos > self.n_pos_file:
            self.used_idx_cnt["pos"] = 0
            self.used_idx_cnt["neg"] = 0

    def __len__(self):
        return self.n_pos_file // self.n_pos
