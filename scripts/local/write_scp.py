#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Split training data and write path."""
import argparse
import codecs
import glob
import random
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_args():
    parser = argparse.ArgumentParser(
        description="Split training data and write the path."
    )
    parser.add_argument(
        "--dumpdir",
        default=[],
        type=str,
        nargs="+",
        help="Path of dump dir.",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation dataset.",
    )
    parser.add_argument(
        "--max_anomaly_pow",
        type=int,
        default=7,
    )
    return parser.parse_args()


def write_eval(args):
    """Run write eval scp."""
    h5_list = [
        path for path in sorted(glob.glob(os.path.join(args.dumpdir[0], "*.h5")))
    ]
    with codecs.open(
        os.path.join(args.dumpdir[0], "eval.scp"), "w", encoding="utf-8"
    ) as g:
        for h5_file in sorted(h5_list):
            g.write(f"{h5_file}\n")
    # random sampling anomaly data for each ID.
    machine = args.dumpdir[0].split("/")[-2]
    for seed in range(5):
        random.seed(seed)
        with codecs.open(
            os.path.join(
                args.dumpdir[0],
                f"train_anomaly0_max{args.max_anomaly_pow}_seed{seed}.scp",
            ),
            "w",
            encoding="utf-8",
        ) as g:
            g.write("")
        for pow in range(args.max_anomaly_pow):
            use_anomaly_list = []
            n_use = 2**pow
            for id_ in range(7):
                anomaly_id_list = []
                if machine == "ToyCar":
                    id_ += 1
                elif machine == "ToyConveyor":
                    id_ += 1
                    if id_ == 7:
                        continue
                for h5 in h5_list:
                    if ("anomaly" in h5) and (int(h5.split("_")[-2]) == id_):
                        anomaly_id_list.append(h5)
                # Select max_anomaly_pow sample
                use_anomaly_list += random.sample(anomaly_id_list, n_use)
            with codecs.open(
                os.path.join(
                    args.dumpdir[0],
                    f"train_anomaly{n_use}_max{args.max_anomaly_pow}_seed{seed}.scp",
                ),
                "w",
                encoding="utf-8",
            ) as g:
                for h5_file in sorted(use_anomaly_list):
                    g.write(f"{h5_file}\n")

        with codecs.open(
            os.path.join(
                args.dumpdir[0], f"eval_max{args.max_anomaly_pow}_seed{seed}.scp"
            ),
            "w",
            encoding="utf-8",
        ) as g:
            for h5 in h5_list:
                if h5 not in use_anomaly_list:
                    g.write(f"{h5}\n")


def write_dev(args):
    """Run write dev scp."""
    all_h5_list = []
    for dumpdir in args.dumpdir:
        all_h5_list += [
            path for path in sorted(glob.glob(os.path.join(dumpdir, "*.h5")))
        ]
    with codecs.open(
        os.path.join(args.dumpdir[0], "dev.scp"), "w", encoding="utf-8"
    ) as g:
        for path in all_h5_list:
            g.write(f"{path}\n")
    label_list = [int(label.split("/")[-1].split("_")[2]) for label in all_h5_list]
    skf = StratifiedKFold(
        n_splits=int(1 / args.valid_ratio), shuffle=True, random_state=42
    )
    for train_idx, valid_idx in skf.split(all_h5_list, label_list):
        break
    with codecs.open(
        os.path.join(args.dumpdir[0], f"train_{args.valid_ratio}.scp"),
        "w",
        encoding="utf-8",
    ) as g:
        for idx in np.sort(train_idx):
            g.write(f"{all_h5_list[idx]}\n")
    with codecs.open(
        os.path.join(args.dumpdir[0], f"valid_{args.valid_ratio}.scp"),
        "w",
        encoding="utf-8",
    ) as g:
        for idx in np.sort(valid_idx):
            g.write(f"{all_h5_list[idx]}\n")


if __name__ == "__main__":
    args = get_args()
    if args.valid_ratio == 0:
        write_eval(args)
    elif args.valid_ratio > 0:
        write_dev(args)
