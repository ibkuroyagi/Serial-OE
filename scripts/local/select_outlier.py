#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Select outliers from the embedding."""
import argparse
import codecs
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from asd_tools.utils import sigmoid


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Select outliers from the embedding.")
    parser.add_argument(
        "--pos_machine",
        default=None,
        type=str,
        help="root directory of positive train dataset.",
    )
    parser.add_argument(
        "--outlier_mean_csv",
        default=None,
        type=str,
        help="outlier mean csv file path.",
    )
    parser.add_argument(
        "--audioset_mean_csv",
        default=None,
        type=str,
        help="audioset mean csv file path.",
    )
    parser.add_argument(
        "--outlier_scp_save_dir",
        default=None,
        type=str,
        help="Directory of outlier scp files..",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    cols = ["pred_machine", "pred_section_max"]
    names = ["outlier", "section"]
    if args.pos_machine in ["fan", "pump", "slider", "valve", "ToyCar"]:
        dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif args.pos_machine == "ToyConveyor":
        dev_section = [0, 1, 2, 3, 4, 5]
    load_cols = ["path", "pred_machine"] + [f"pred_section{i}" for i in dev_section]
    agg_df = pd.concat(
        [
            pd.read_csv(args.outlier_mean_csv, usecols=load_cols),
            pd.read_csv(args.audioset_mean_csv, usecols=load_cols),
        ]
    )
    agg_df["pred_section_max"] = agg_df[[f"pred_section{i}" for i in dev_section]].max(
        1
    )
    for ii, col in enumerate(cols):
        plt.figure(figsize=(5, 4))
        sig = sigmoid(agg_df[col])
        plt.hist(sig, bins=100, alpha=0.5, label=names[ii], density=False)
        plt.hist(
            sig, bins=100, cumulative=True, density=False, label=names[ii], alpha=0.5
        )
        plt.legend(loc="lower right")
        use_cnt = (sig > 0.5).sum()
        N = len(agg_df)
        title = f"{args.pos_machine}, N={N}, cnt={use_cnt}, rate > 0.5 ={use_cnt/N*100:.2f}[%]"
        plt.title(title)
        plt.tight_layout()
        fig_path = f"exp/fig/select_{names[ii]}_{args.pos_machine}.png"
        plt.savefig(fig_path)
        logging.info(f"Saved at {fig_path}.")
        for threshold in [
            0,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9995,
            0.9999,
            1,
        ]:
            use_idx = sig > threshold
            use_cnt = use_idx.sum()
            N = len(agg_df)
            title = f"{args.pos_machine}:{names[ii]}, N={N}, col={col}, cnt={use_cnt}, rate={use_cnt/N*100:.2f}[%], {threshold}"
            logging.info(title)
            path_list = list(agg_df.loc[use_idx, "path"])
            outlier_scp_path = os.path.join(
                args.outlier_scp_save_dir, f"{names[ii]}_{threshold}.scp"
            )
            with codecs.open(
                outlier_scp_path,
                "w",
                encoding="utf-8",
            ) as g:
                for path in path_list:
                    g.write(f"{path}\n")
            logging.info(f"Outlier scp file is saved at {outlier_scp_path}.")

    for threshold in [
        0,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]:
        use_idx = sig < np.percentile(sig, threshold * 100)
        use_cnt = use_idx.sum()
        N = len(agg_df)
        title = f"{args.pos_machine}:{names[ii]}, N={N}, col={col}, cnt={use_cnt}, rate={use_cnt/N*100:.2f}[%], {threshold}"
        logging.info(title)
        path_list = list(agg_df.loc[use_idx, "path"])
        outlier_scp_path = os.path.join(
            args.outlier_scp_save_dir, f"machine_{threshold}.scp"
        )
        with codecs.open(
            outlier_scp_path,
            "w",
            encoding="utf-8",
        ) as g:
            for path in path_list:
                g.write(f"{path}\n")
        logging.info(f"Outlier scp file is saved at {outlier_scp_path}.")


if __name__ == "__main__":
    main()
