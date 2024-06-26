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

from torch.utils.data import DataLoader

import serial_oe
import serial_oe.losses
import serial_oe.models
from serial_oe.datasets import ASDDataset
from serial_oe.datasets import WaveEvalCollator
from serial_oe.utils import seed_everything

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Extract features from trained model.")
    parser.add_argument(
        "--valid_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--eval_pos_machine_scp",
        type=str,
        required=True,
        help="directory to evaluation machine.",
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
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="checkpoint file path to resume training. (default=[])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        help="Seed of the model. (default=2022)",
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
    if "ToyConveyor" in args.valid_pos_machine_scp:
        config["model_params"]["out_dim"] = 6
    seed_everything(seed=config["seed"])
    config["batch_size"] = (
        config["n_pos"] + config["n_neg"] + config.get("n_anomaly", 0)
    )
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    valid_dataset = ASDDataset(
        pos_machine_scp=args.valid_pos_machine_scp,
        neg_machine_scps=[],
        allow_cache=True,
        statistic_path=args.statistic_path,
    )
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    logging.info(f"pos = {len(valid_dataset.pos_files)}.")

    collator = WaveEvalCollator(
        sf=config["sf"],
        sec=config["sec"],
        n_split=config["n_split"],
        is_label=True,
    )
    loader_dict = {
        "valid": DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            collate_fn=collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
    }
    if os.path.isfile(args.eval_pos_machine_scp):
        eval_dataset = ASDDataset(
            pos_machine_scp=args.eval_pos_machine_scp,
            neg_machine_scps=[],
            allow_cache=True,
            statistic_path=args.statistic_path,
        )
        logging.info(f"The number of evaluation files = {len(eval_dataset)}.")
        loader_dict["eval"] = DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            collate_fn=collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
    for checkpoint in args.checkpoints:
        model_class = getattr(serial_oe.models, config["model_type"])
        model = model_class(**config["model_params"])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        logging.info(f"Successfully loaded {checkpoint}.")
        logging.info(
            f"Steps:{state_dict['steps']}, "
            f"Epochs:{state_dict['epochs']}, "
            f"BEST loss:{state_dict['best_loss']}"
        )
        for mode, loader in loader_dict.items():
            pred_machine = np.empty((0, 1))
            pred_product = np.empty((0, config["model_params"]["out_dim"]))
            embed = np.empty((0, config["model_params"]["embedding_size"]))
            path_list = np.empty((0, 1))
            split_list = np.empty((0, 1))
            product_list = np.empty((0, 1))
            is_normal_list = np.empty((0, 1))

            for batch in loader:
                for i in range(config["n_split"]):
                    with torch.no_grad():
                        y_ = model(batch[f"X{i}"].to(device))
                    pred_machine = np.concatenate(
                        [pred_machine, y_["machine"].cpu().numpy()], axis=0
                    )
                    pred_product = np.concatenate(
                        [pred_product, y_["product"].cpu().numpy()], axis=0
                    )
                    embed = np.concatenate(
                        [embed, y_["embedding"].cpu().numpy()], axis=0
                    )
                    path_list = np.concatenate([path_list, batch["path"][:, None]])
                    split_list = np.concatenate(
                        [split_list, np.ones((len(y_["embedding"]), 1)) * i]
                    )
                    product_list = np.concatenate(
                        [product_list, batch["product"][:, None]]
                    )
                    is_normal_list = np.concatenate(
                        [is_normal_list, batch["is_normal"][:, None]]
                    )

            embed_cols = [
                f"e{i}" for i in range(config["model_params"]["embedding_size"])
            ]
            pred_product_cols = [
                f"pred_product{i}" for i in range(config["model_params"]["out_dim"])
            ]
            columns = (
                [
                    "path",
                    "split",
                    "product",
                    "is_normal",
                    "pred_machine",
                    "embed_norm",
                ]
                + pred_product_cols
                + embed_cols
            )

            logging.info(
                f"path_list:{path_list.shape}, split_list:{split_list.shape}, "
                f"product_list:{product_list.shape}, "
                f"is_normal_list:{is_normal_list.shape}, pred_machine:{pred_machine.shape}, "
                f"pred_product:{pred_product.shape}, embed:{embed.shape}"
            )
            df = pd.DataFrame(embed, columns=embed_cols)
            df["path"] = path_list
            df["split"] = split_list.astype(int)
            df["product"] = product_list.astype(int)
            df["is_normal"] = is_normal_list.astype(int)
            df["pred_machine"] = pred_machine
            df["embed_norm"] = (
                np.sqrt(np.power(embed, 2).sum(1))
                / config["model_params"]["embedding_size"]
            )
            df[pred_product_cols] = pred_product

            df = df[columns]
            csv_path = checkpoint.replace(".pkl", f"_{mode}.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved at {csv_path}")


if __name__ == "__main__":
    main()
