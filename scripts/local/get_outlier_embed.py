#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Extract embedding vectors from outliers."""

import argparse
import csv
import json
import logging
import sys
import numpy as np
import matplotlib
import pandas as pd
import torch
import yaml
import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import asd_tools
import asd_tools.losses
import asd_tools.models
from asd_tools.datasets import OutlierWaveASDDataset
from asd_tools.datasets import WaveEvalCollator
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
        description="Extract embedding vectors from outliers."
    )
    parser.add_argument(
        "--pos_machine",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--train_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--train_neg_machine_scps",
        default=[],
        type=str,
        nargs="*",
        help="list of root directories of positive train datasets.",
    )
    parser.add_argument(
        "--valid_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive valid dataset.",
    )
    parser.add_argument(
        "--valid_neg_machine_scps",
        default=[],
        type=str,
        nargs="*",
        help="list of root directories of positive train datasets.",
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
    # get dataset
    train_dataset = OutlierWaveASDDataset(
        pos_machine_scp=args.train_pos_machine_scp,
        neg_machine_scps=args.train_neg_machine_scps,
        outlier_scps=[],
        allow_cache=True,
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    logging.info(f"The number of train files = {len(train_dataset)}.")
    logging.info(f"pos = {len(train_dataset.pos_files)}.")
    valid_dataset = OutlierWaveASDDataset(
        pos_machine_scp=args.valid_pos_machine_scp,
        neg_machine_scps=args.valid_neg_machine_scps,
        outlier_scps=[],
        allow_cache=True,
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    logging.info(f"pos = {len(valid_dataset.pos_files)}.")
    outlier_dataset = OutlierDataset(
        outlier_scps=args.outlier_scps,
        statistic_path=args.statistic_path,
        in_sample_norm=config.get("in_sample_norm", False),
    )
    logging.info(f"The number of outlier files = {len(outlier_dataset)}.")

    collator = WaveEvalCollator(
        sf=config["sf"],
        sec=config["sec"],
        n_split=config["n_split"],
        is_label=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collator,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        collate_fn=collator,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    outlier_loader = DataLoader(
        outlier_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    if len(args.audioset_scps) > 0:
        audioset_dataset = OutlierDataset(
            outlier_scps=args.audioset_scps,
            statistic_path=args.statistic_path,
            in_sample_norm=config.get("in_sample_norm", False),
        )
        logging.info(f"The number of audioset files = {len(audioset_dataset)}.")
        audioset_loader = DataLoader(
            audioset_dataset,
            batch_size=config["batch_size"] * 8,
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
    split_dict = {
        "dcase_train": config["n_split"],
        "dcase_valid": config["n_split"],
        "audioset": 5,
    }
    loader_dict = {
        "dcase_train": train_loader,
        "dcase_valid": valid_loader,
        "outlier": outlier_loader,
    }
    # if len(args.audioset_scps) > 0:
    #     loader_dict["audioset"] = audioset_loader
    for mode, loader in loader_dict.items():
        csv_path = args.checkpoint.replace(".pkl", f"_{mode}_mean.csv")
        with open(csv_path, "w", newline="") as g:
            writer = csv.writer(g)
            writer.writerow(columns)
            for batch in tqdm(loader):
                b_pred_machine = np.empty((0, 1))
                b_pred_section = np.empty((0, config["model_params"]["out_dim"]))
                b_embed = np.empty((0, config["model_params"]["embedding_size"]))
                b_path_list = np.empty((0, 1))
                b_split_list = np.empty((0, 1))
                if mode == "outlier":
                    with torch.no_grad():
                        for i in range(len(batch.keys()) - 1):
                            y_ = model(batch[f"wave{i}"].to(device))
                            b_pred_machine = np.concatenate(
                                [b_pred_machine, y_["machine"].cpu().numpy()], axis=0
                            )
                            b_pred_section = np.concatenate(
                                [b_pred_section, y_["section"].cpu().numpy()], axis=0
                            )
                            b_embed = np.concatenate(
                                [b_embed, y_["embedding"].cpu().numpy()], axis=0
                            )
                            b_path_list = np.concatenate(
                                [b_path_list, np.array([batch["path"]])]
                            )
                            b_split_list = np.concatenate(
                                [b_split_list, np.ones((len(y_["embedding"]), 1)) * i]
                            )
                elif mode in ["dcase_train", "dcase_valid", "audioset"]:
                    with torch.no_grad():
                        for i in range(split_dict[mode]):
                            if mode == "audioset":
                                y_ = model(batch[f"wave{i}"].to(device))
                            else:
                                y_ = model(batch[f"X{i}"].to(device))
                            b_pred_machine = np.concatenate(
                                [b_pred_machine, y_["machine"].cpu().numpy()], axis=0
                            )
                            b_pred_section = np.concatenate(
                                [b_pred_section, y_["section"].cpu().numpy()], axis=0
                            )
                            b_embed = np.concatenate(
                                [b_embed, y_["embedding"].cpu().numpy()], axis=0
                            )
                            b_path_list = np.concatenate(
                                [b_path_list, np.array(batch["path"]).reshape(-1, 1)]
                            )
                            b_split_list = np.concatenate(
                                [b_split_list, np.ones((len(y_["embedding"]), 1)) * i]
                            )
                b_df = pd.DataFrame(b_embed, columns=embed_cols)
                b_df["path"] = b_path_list
                b_df["split"] = b_split_list.astype(int)
                b_df["pred_machine"] = b_pred_machine
                b_df["embed_norm"] = (
                    np.sqrt(np.power(b_embed, 2).sum(1))
                    / config["model_params"]["embedding_size"]
                )
                b_df[pred_section_cols] = b_pred_section
                b_df = b_df[columns]
                b_agg_df = b_df.groupby(by="path").mean().reset_index()
                writer.writerows(b_agg_df.values)


if __name__ == "__main__":
    main()
