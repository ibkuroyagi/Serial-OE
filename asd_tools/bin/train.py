#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Train Anomalous Sound Detection model."""

import argparse
import logging
import os
import sys
import matplotlib
import torch
import yaml

from torch.utils.data import DataLoader

import serial_oe
import serial_oe.models
import serial_oe.losses
import serial_oe.optimizers
import serial_oe.schedulers
from serial_oe.datasets import WaveCollator
from serial_oe.trainer import OECTrainer
from serial_oe.utils import count_params
from serial_oe.utils import seed_everything
from serial_oe.datasets import BalancedBatchSampler
from serial_oe.datasets import ASDDataset

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in serial_oe/bin/train.py)."
    )
    parser.add_argument(
        "--pos_machine", type=str, required=True, help="Name of positive machine."
    )
    parser.add_argument(
        "--train_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--train_pos_anomaly_machine_scp",
        default="",
        type=str,
        help="root directory of positive train anomaly dataset.",
    )
    parser.add_argument(
        "--train_neg_machine_scps",
        default=[],
        type=str,
        nargs="+",
        help="list of root directories of positive train datasets.",
    )
    parser.add_argument(
        "--valid_pos_machine_scp",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--valid_neg_machine_scps",
        default=[],
        type=str,
        nargs="+",
        help="list of root directories of positive train datasets.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--valid_outlier_scps",
        default=[],
        type=str,
        nargs="*",
        help="list of scp file of validation outlier dataset.",
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="",
        help="Statistic info of positive data in json.",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
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
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

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

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    seed_everything(seed=config["seed"])
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    config["batch_size"] = (
        config["n_pos"] + config["n_neg"] + config.get("n_anomaly", 0)
    )
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    # get dataset
    train_dataset = ASDDataset(
        pos_machine_scp=args.train_pos_machine_scp,
        pos_anomaly_machine_scp=args.train_pos_anomaly_machine_scp,
        neg_machine_scps=args.train_neg_machine_scps,
        outlier_scps=args.outlier_scps,
        allow_cache=config.get("allow_cache", False),
        statistic_path=args.statistic_path,
    )
    valid_dataset = ASDDataset(
        pos_machine_scp=args.valid_pos_machine_scp,
        pos_anomaly_machine_scp="",
        neg_machine_scps=args.valid_neg_machine_scps,
        outlier_scps=args.valid_outlier_scps,
        allow_cache=True,
        statistic_path=args.statistic_path,
    )
    train_balanced_batch_sampler = BalancedBatchSampler(
        train_dataset,
        n_pos=config["n_pos"],
        n_neg=config["n_neg"],
        shuffle=True,
        drop_last=True,
        anomaly_as_neg=config.get("anomaly_as_neg", True),
        n_anomaly=config.get("n_anomaly_in_mini_batch", 0),
    )
    valid_balanced_batch_sampler = BalancedBatchSampler(
        valid_dataset,
        n_pos=config["n_pos"],
        n_neg=config["n_pos"],
        shuffle=False,
        drop_last=True,
        anomaly_as_neg=config.get("anomaly_as_neg", True),
        n_anomaly=config.get("n_anomaly_in_mini_batch", 0),
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"train pos = {len(train_dataset.pos_files)}.")
    logging.info(f"train pos anomaly = {len(train_dataset.pos_anomaly_files)}.")
    logging.info(f"train neg = {len(train_dataset.neg_files)}.")
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    logging.info(f"valid pos = {len(valid_dataset.pos_files)}.")
    logging.info(f"valid pos anomaly = {len(valid_dataset.pos_anomaly_files)}.")
    logging.info(f"valid neg = {len(valid_dataset.neg_files)}.")
    train_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        pos_machine=args.pos_machine,
        shuffle=True,
        use_is_normal=False,
        anomaly_as_neg=config.get("anomaly_as_neg", True),
    )
    valid_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        pos_machine=args.pos_machine,
        shuffle=False,
        use_is_normal=False,
        anomaly_as_neg=config.get("anomaly_as_neg", True),
    )
    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_sampler=train_balanced_batch_sampler,
            collate_fn=train_collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_sampler=valid_balanced_batch_sampler,
            collate_fn=valid_collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    model_class = getattr(serial_oe.models, config["model_type"])
    if config["pos_machine"] == "ToyConveyor":
        config["model_params"]["out_dim"] = 6
    model = model_class(**config["model_params"]).to(device)
    logging.info(model)
    params_cnt = count_params(model)
    logging.info(f"Size of model is {params_cnt}.")
    criterion = {
        "machine_loss": getattr(serial_oe.losses, config["machine_loss_type"],)(
            **config["machine_loss_params"]
        ).to(device),
        "section_loss": getattr(serial_oe.losses, config["section_loss_type"],)(
            **config["section_loss_params"]
        ).to(device),
    }
    optimizer_class = getattr(
        serial_oe.optimizers,
        config["optimizer_type"],
    )
    params_list = [{"params": model.parameters()}]
    metric_fc = None
    if config.get("metric_fc_type", None) is not None:
        metric_fc_class = getattr(
            serial_oe.losses,
            config["metric_fc_type"],
        )
        metric_fc = metric_fc_class(**config["metric_fc_params"]).to(device)
        params_list.append({"params": metric_fc.parameters()})
    optimizer = optimizer_class(params_list, **config["optimizer_params"])
    scheduler = None
    if config.get("scheduler_type", None) is not None:
        scheduler_class = getattr(
            serial_oe.schedulers,
            config["scheduler_type"],
        )
        if config["scheduler_type"] == "OneCycleLR":
            config["scheduler_params"]["epochs"] = config["train_max_epochs"]
            if config.get("anomaly_as_neg", True):
                steps_per_epoch = len(train_dataset.pos_files) // config["n_pos"] + 1
            else:
                steps_per_epoch = (
                    len(train_dataset.pos_files) + len(train_dataset.pos_anomaly_files)
                ) // config["n_pos"] + 1
            config["scheduler_params"]["steps_per_epoch"] = steps_per_epoch
        scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])

    # define trainer
    trainer = OECTrainer(
        steps=1,
        epochs=1,
        data_loader=data_loader,
        model=model.to(device),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train=True,
        metric_fc=metric_fc,
    )

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
