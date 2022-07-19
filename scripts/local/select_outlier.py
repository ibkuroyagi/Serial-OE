#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Select outliers from the embedding."""
import argparse
import codecs
import logging
import sys
import os
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
    col = "pred_machine"
    agg_df = pd.concat(
        [
            pd.read_csv(args.outlier_mean_csv, usecols=["path", col]),
            pd.read_csv(args.audioset_mean_csv, usecols=["path", col]),
        ]
    )
    plt.figure(figsize=(5, 4))
    sig = sigmoid(agg_df[col])
    plt.hist(sig, bins=100, alpha=0.5, label="outlier", density=False)
    plt.hist(sig, bins=100, cumulative=True, density=False, label="outlier", alpha=0.5)
    plt.legend(loc="lower right")
    use_cnt = (sig > 0.5).sum()
    N = len(agg_df)
    title = (
        f"{args.pos_machine}, N={N}, cnt={use_cnt}, rate > 0.5 ={use_cnt/N*100:.2f}[%]"
    )
    plt.title(title)
    plt.tight_layout()
    fig_path = f"exp/fig/select_outlier_{args.pos_machine}.png"
    plt.savefig(fig_path)
    logging.info(f"Saved at {fig_path}.")
    for threshold in [0, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]:
        use_idx = sig > threshold
        use_cnt = use_idx.sum()
        N = len(agg_df)
        title = f"{args.pos_machine}, N={N}, col={col}, cnt={use_cnt}, rate={use_cnt/N*100:.2f}[%], {threshold}"
        logging.info(title)
        path_list = list(agg_df.loc[use_idx, "path"])
        outlier_scp_path = os.path.join(
            args.outlier_scp_save_dir, f"outlier_{threshold}.scp"
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
