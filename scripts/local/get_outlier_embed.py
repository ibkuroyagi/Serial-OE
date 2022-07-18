#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Extract embedding vectors."""

import argparse
import logging
import os
import sys
import numpy as np
import matplotlib
import pandas as pd
import torch
import yaml
import h5py
from torch.utils.data import DataLoader
import json
import logging
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import asd_tools
import asd_tools.losses
import asd_tools.models
from asd_tools.utils import seed_everything

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class OutlierDataset(Dataset):
    """Outlier Wave dataset."""

    def __init__(
        self,
        outlier_scps=[],
        statistic_path="",
        in_sample_norm=False,
    ):
        """Initialize dataset."""
        self.outlier_files = []
        for outlier_scp in outlier_scps:
            with open(outlier_scp, "r") as f:
                self.outlier_files += [s.strip() for s in f.readlines()]
        self.outlier_files.sort()
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

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave{i}: (ndarray) Wave (T, ).
        """
        path = self.outlier_files[idx]
        items = {"path": path}
        hdf5_file = h5py.File(path, "r")
        for i in range(len(hdf5_file)):
            items[f"wave{i}"] = hdf5_file[f"wave{i}"][()]
            if self.statistic is not None:
                items[f"wave{i}"] -= self.statistic["mean"]
                items[f"wave{i}"] /= self.statistic["std"]
            if self.in_sample_norm:
                items[f"wave{i}"] -= items[f"wave{i}"].mean()
                if items[f"wave{i}"].std() == 0:
                    items[f"wave{i}"] += np.random.randn(len(items[f"wave{i}"]))
                items[f"wave{i}"] /= items[f"wave{i}"].std()
        hdf5_file.close()
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.outlier_files)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
    )
    parser.add_argument(
        "--pos_machine",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--outlier_scps",
        default=[],
        type=str,
        nargs="*",
        help="outlier file path. (default=[])",
    )
    parser.add_argument(
        "--audioset_scps",
        default=[],
        type=str,
        nargs="*",
        help="audioset file path. (default=[])",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="",
        help="Statistic info of positive data in json.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="checkpoint file path to trained models.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    if "ToyConveyor" in args.pos_machine:
        config["model_params"]["out_dim"] = 6
    seed_everything(seed=config["seed"])
    config["batch_size"] = (
        config["n_pos"] + config["n_neg"] + config.get("n_anomaly", 0)
    )
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    # for valid and eval
    for mode in ["valid", "eval"]:
        df = pd.read_csv(args.checkpoint.replace(".pkl", f"_{mode}.csv"))
        agg_df = df.groupby(by="path").mean().reset_index()
        agg_df.to_csv(args.checkpoint.replace(".pkl", f"_{mode}_mean.csv"), index=False)
    outlier_dataset = OutlierDataset(
        outlier_scps=args.outlier_scps,
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    audioset_dataset = OutlierDataset(
        outlier_scps=args.audioset_scps,
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    logging.info(f"The number of outlier files = {len(outlier_dataset)}.")
    logging.info(f"The number of audioset files = {len(audioset_dataset)}.")
    outlier_loader = DataLoader(
        outlier_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    audioset_loader = DataLoader(
        audioset_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    model_class = getattr(asd_tools.models, config["model_type"])
    model = model_class(**config["model_params"])
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    logging.info(f"Successfully loaded {args.checkpoint}.")
    logging.info(
        f"Steps:{state_dict['steps']}, "
        f"Epochs:{state_dict['epochs']}, "
        f"BEST loss:{state_dict['best_loss']}"
    )
    for mode, loader in {
        "outlier": outlier_loader,
        "audioset": audioset_loader,
    }.items():
        pred_machine = np.empty((0, 1))
        pred_section = np.empty((0, config["model_params"]["out_dim"]))
        embed = np.empty((0, config["model_params"]["embedding_size"]))
        path_list = np.empty((0, 1))
        split_list = np.empty((0, 1))
        for batch in tqdm(loader):
            if mode == "outlier":
                with torch.no_grad():
                    for i in range(len(batch.keys()) - 1):
                        y_ = model(batch[f"wave{i}"].to(device))
                        pred_machine = np.concatenate(
                            [pred_machine, y_["machine"].cpu().numpy()], axis=0
                        )
                        pred_section = np.concatenate(
                            [pred_section, y_["section"].cpu().numpy()], axis=0
                        )
                        embed = np.concatenate(
                            [embed, y_["embedding"].cpu().numpy()], axis=0
                        )
                        path_list = np.concatenate(
                            [path_list, np.array([batch["path"]])]
                        )
                        split_list = np.concatenate(
                            [split_list, np.ones((len(y_["embedding"]), 1)) * i]
                        )
            elif mode == "audioset":
                with torch.no_grad():
                    for i in range(5):
                        y_ = model(batch[f"wave{i}"].to(device))
                        pred_machine = np.concatenate(
                            [pred_machine, y_["machine"].cpu().numpy()], axis=0
                        )
                        pred_section = np.concatenate(
                            [pred_section, y_["section"].cpu().numpy()], axis=0
                        )
                        embed = np.concatenate(
                            [embed, y_["embedding"].cpu().numpy()], axis=0
                        )
                        path_list = np.concatenate([path_list, batch["path"][:, None]])
                        split_list = np.concatenate(
                            [split_list, np.ones((len(y_["embedding"]), 1)) * i]
                        )
        embed_cols = [f"e{i}" for i in range(config["model_params"]["embedding_size"])]
        pred_section_cols = [
            f"pred_section{i}" for i in range(config["model_params"]["out_dim"])
        ]
        columns = (
            [
                "path",
                "split",
                "pred_machine",
                "embed_norm",
            ]
            + pred_section_cols
            + embed_cols
        )

        logging.info(
            f"path_list:{path_list.shape}, split_list:{split_list.shape}, "
            f"pred_section:{pred_section.shape}, embed:{embed.shape}"
        )
        df = pd.DataFrame(embed, columns=embed_cols)
        df["path"] = path_list
        df["split"] = split_list.astype(int)
        df["pred_machine"] = pred_machine
        df["embed_norm"] = (
            np.sqrt(np.power(embed, 2).sum(1))
            / config["model_params"]["embedding_size"]
        )
        df[pred_section_cols] = pred_section
        df = df[columns]
        agg_df = df.groupby(by="path").mean().reset_index()
        csv_path = args.checkpoint.replace(".pkl", f"_{mode}_mean.csv")
        agg_df.to_csv(csv_path, index=False)
        logging.info(f"Saved at {csv_path}")


if __name__ == "__main__":
    main()
