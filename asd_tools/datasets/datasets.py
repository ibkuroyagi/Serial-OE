import json
import logging
import numpy as np
from multiprocessing import Manager
from torch.utils.data import Dataset
from asd_tools import datasets
from asd_tools.utils import read_hdf5


class OutlierWaveASDDataset(Dataset):
    """Outlier Wave dataset."""

    def __init__(
        self,
        pos_machine_scp="",
        outlier_scp="",
        neg_machine_scps=[],
        allow_cache=False,
        augmentation_params={},
        statistic_path="",
        in_sample_norm=False,
    ):
        """Initialize dataset."""
        self.pos_files, self.neg_files = [], []
        with open(pos_machine_scp, "r") as f:
            self.pos_files = [s.strip() for s in f.readlines()]
        for neg_machine_scp in neg_machine_scps:
            with open(neg_machine_scp, "r") as f:
                neg_files = [s.strip() for s in f.readlines()]
            self.neg_files += neg_files
        if len(outlier_scp) == 0:
            self.outlier_files = []
        else:
            with open(outlier_scp, "r") as f:
                self.outlier_files = [s.strip() for s in f.readlines()]
        self.neg_files.sort()
        self.wav_files = self.pos_files + self.neg_files + self.outlier_files
        self.outlier_size = len(self.pos_files) + len(self.neg_files)
        self.caches_size = (
            len(self.pos_files)
            # + len(self.neg_files)
        )
        self.augmentation_params = augmentation_params
        self.transform = None
        if len(augmentation_params) != 0:
            compose_list = []
            for key in self.augmentation_params.keys():
                aug_class = getattr(datasets, key)
                compose_list.append(aug_class(**self.augmentation_params[key]))
                logging.debug(f"{key}")
            self.transform = datasets.Compose(compose_list)
        # statistic
        self.statistic = None
        self.in_sample_norm = in_sample_norm
        if in_sample_norm:
            logging.info("Data is normalized in sample. Not using statistic feature.")
        else:
            with open(statistic_path, "r") as f:
                self.statistic = json.load(f)
            logging.info(
                f"{statistic_path} mean: {self.statistic['mean']:.4f},"
                f" std: {self.statistic['std']:.4f}"
            )
        # for cache
        self.allow_cache = allow_cache
        if allow_cache:
            # Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(self.caches_size)]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave: (ndarray) Wave (T, ).
                machine: (str) Name of machine.
                section: (int) Number of machine id.
        """
        if self.allow_cache and (self.caches_size > idx):
            # logging.info(f"self.caches[{idx}]:{self.caches[idx]}")
            if len(self.caches[idx]) != 0:
                if self.transform is None:
                    return self.caches[idx]
                else:
                    self.caches[idx]["wave"] = self.transform(
                        self.caches[idx]["origin_wave"]
                    )
                    return self.caches[idx]
        path = self.wav_files[idx]
        items = {"path": path}
        if self.outlier_size > idx:
            items["wave"] = read_hdf5(path, "wave")
            if self.statistic is not None:
                items["wave"] -= self.statistic["mean"]
                items["wave"] /= self.statistic["std"]
            if self.in_sample_norm:
                items["wave"] -= items["wave"].mean()
                items["wave"] /= items["wave"].std()
            items["machine"] = path.split("/")[-3]
            items["section"] = int(path.split("/")[-1].split("_")[2])
            if items["machine"] in ["ToyCar", "ToyConveyor"]:
                items["section"] -= 1
            items["is_normal"] = int(path.split("/")[-1].split("_")[0] == "normal")
            if self.transform is not None:
                items["origin_wave"] = items["wave"].copy()
                items["wave"] = self.transform(items["origin_wave"])
            if self.allow_cache and (self.caches_size > idx):
                self.caches[idx] = items
        else:
            items["wave"] = read_hdf5(path, read_random=True)
            if self.statistic is not None:
                items["wave"] -= self.statistic["mean"]
                items["wave"] /= self.statistic["std"]
            if self.in_sample_norm:
                items["wave"] -= items["wave"].mean()
                if items["wave"].std() == 0:
                    items["wave"] += np.random.randn(len(items["wave"]))
                items["wave"] /= items["wave"].std()
            items["machine"] = "outlier"
            items["section"] = 0
            items["is_normal"] = 0
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.wav_files)
