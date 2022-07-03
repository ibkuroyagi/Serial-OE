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
    ):
        """Batch Sampler.

        Args:
            dataset (dataset): dataset for ASD
            n_pos (int, optional): The number of positive sample in the mini-batch. Defaults to 32.
            n_neg (int, optional): The number of negative sample in the mini-batch. Defaults to 32.
            shuffle (bool, optional): shuffle. Defaults to False.
            drop_last (bool, optional): drop last. Defaults to False.
        """
        self.n_pos_file = len(dataset.pos_files)
        self.pos_idx = np.arange(self.n_pos_file)

        self.n_neg_file = len(dataset.neg_files) + len(dataset.outlier_files)
        self.neg_idx = np.arange(
            self.n_pos_file,
            self.n_pos_file + self.n_neg_file,
        )
        self.used_idx_cnt = {"pos": 0, "neg": 0}
        self.count = 0
        self.n_pos = n_pos
        self.n_neg = n_neg
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
            yield indices
        if self.used_idx_cnt["pos"] + self.n_pos > self.n_pos_file:
            self.used_idx_cnt["pos"] = 0
            self.used_idx_cnt["neg"] = 0

    def __len__(self):
        return self.n_pos_file // self.n_pos
