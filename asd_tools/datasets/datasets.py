import json
import logging
from multiprocessing import Manager
from torch.utils.data import Dataset
from serial_oe.utils import read_hdf5


class ASDDataset(Dataset):
    """Pytorch dataset."""

    def __init__(
        self,
        pos_machine_scp="",
        pos_anomaly_machine_scp="",
        neg_machine_scps=[],
        allow_cache=False,
        statistic_path="",
    ):
        """Initialize dataset."""
        with open(pos_machine_scp, "r") as f:
            self.pos_files = [s.strip() for s in f.readlines()]
        self.pos_anomaly_files, self.neg_files, self.outlier_files = [], [], []
        if len(pos_anomaly_machine_scp) != 0:
            with open(pos_anomaly_machine_scp, "r") as f:
                self.pos_anomaly_files = [s.strip() for s in f.readlines()]
        for neg_machine_scp in neg_machine_scps:
            with open(neg_machine_scp, "r") as f:
                neg_files = [s.strip() for s in f.readlines()]
            self.neg_files += neg_files
        self.neg_files.sort()
        self.wav_files = self.pos_files + self.pos_anomaly_files + self.neg_files
        self.caches_size = (
            len(self.pos_files)
            + len(self.pos_anomaly_files)
            # + len(self.neg_files)
        )
        # statistic
        self.statistic = None
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
            if len(self.caches[idx]) != 0:
                return self.caches[idx]
        path = self.wav_files[idx]
        items = {"path": path}
        items["wave"] = read_hdf5(path, "wave")
        if self.statistic is not None:
            items["wave"] -= self.statistic["mean"]
            items["wave"] /= self.statistic["std"]
        items["machine"] = path.split("/")[-3]
        items["section"] = int(path.split("/")[-1].split("_")[2])
        if items["machine"] in ["ToyCar", "ToyConveyor"]:
            items["section"] -= 1
        items["is_normal"] = int(path.split("/")[-1].split("_")[0] == "normal")
        if self.allow_cache and (self.caches_size > idx):
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.wav_files)
