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
        default=0.1,
        help="Ratio of validation dataset.",
    )
    parser.add_argument(
        "--max_audioset_size_2_pow",
        type=int,
        default=0,
    )
    return parser.parse_args()


def write_audioset(args):
    """Run write audioset scp."""
    h5_list = [
        path for path in sorted(glob.glob(os.path.join(args.dumpdir[0], "**", "*.h5")))
    ]
    print(f"Audioset has {len(h5_list)} files.")
    for audioset_size_2_pow in range(args.max_audioset_size_2_pow):
        scp_size = int(min(2**audioset_size_2_pow, len(h5_list)))
        scp_path = os.path.join(
            args.dumpdir[0], f"audioset_2__{audioset_size_2_pow}.scp"
        )
        sampled_list = random.sample(h5_list, scp_size)
        with codecs.open(
            scp_path,
            "w",
            encoding="utf-8",
        ) as g:
            for h5_file in sampled_list:
                g.write(f"{h5_file}\n")
        print(f"Wrote {scp_path}, it contain {len(sampled_list)} files.")


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
    elif args.max_audioset_size_2_pow > 0:
        write_audioset(args)
    else:
        write_dev(args)
