#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Scoring."""

import argparse
import logging
import os
import sys
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
    )
    parser.add_argument("--feature", type=str, default="", help="Type of feature.")
    parser.add_argument(
        "--agg_checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="Aggregated checkpoints files.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()
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
    return args


def main(args):
    """Run scoring process."""
    dev_hauc_cols = []
    dev_columns = []
    eval_columns = []
    use_eval = True
    modes = ["dev", "eval"] if use_eval else ["dev"]
    sections = {"dev": [0, 1, 2], "eval": [3, 4, 5]}
    for machine in ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]:
        dev_columns += [f"dev_{machine}_hauc"]
        dev_hauc_cols += [f"dev_{machine}_hauc"]
        if use_eval:
            eval_columns += [f"eval_{machine}_hauc"]

    agg_df = pd.read_csv(args.agg_checkpoints[0])
    post_processes = list(agg_df.columns)
    for rm in ["path", "is_normal", "section", "mode"]:
        post_processes.remove(rm)
    columns = ["path", "dev_hauc"]
    if use_eval:
        columns += [
            "eval_hauc",
            "hauc",
        ]
    columns += eval_columns + dev_columns
    score_df = pd.DataFrame(index=post_processes, columns=columns)

    save_path = os.path.join(
        "/".join(
            ["exp", "all"] + os.path.dirname(args.agg_checkpoints[0]).split("/")[2:]
        ),
        f"score{args.feature}.csv",
    )
    score_df.loc[:, "path"] = save_path
    for agg_path in args.agg_checkpoints:
        logging.info(f"Loaded {agg_path}.")
        agg_df = pd.read_csv(agg_path)
        machine = agg_path.split("/")[1]
        for post_process in post_processes:
            for mode in modes:
                auc_list = []
                pauc_list = []
                for section in sections[mode]:
                    target_idx = (agg_df["mode"] == mode) & (
                        agg_df["section"] == section
                    )
                    auc_list.append(
                        roc_auc_score(
                            1 - agg_df.loc[target_idx, "is_normal"],
                            -agg_df.loc[target_idx, post_process],
                        )
                    )
                    # score_df.loc[post_process, f"{mode}_{machine}_auc"] = auc
                    pauc_list.append(
                        roc_auc_score(
                            1 - agg_df.loc[target_idx, "is_normal"],
                            -agg_df.loc[target_idx, post_process],
                            max_fpr=0.1,
                        )
                    )
                    # score_df.loc[post_process, f"{mode}_{machine}_pauc"] = pauc
                score_list = auc_list + pauc_list
                score_df.loc[post_process, f"{mode}_{machine}_hauc"] = hmean(score_list)
                score_df.loc[post_process, f"{mode}_{machine}_hauc_std"] = np.array(
                    score_list
                ).std()
    for post_process in post_processes:
        score_df.loc[post_process, "dev_hauc"] = hmean(
            score_df.loc[post_process, dev_hauc_cols].values.flatten()
        )
        if use_eval:
            score_df.loc[post_process, "eval_hauc"] = hmean(
                score_df.loc[post_process, eval_columns].values.flatten()
            )
            score_df.loc[post_process, "hauc"] = hmean(
                score_df.loc[post_process, dev_columns + eval_columns].values.flatten()
            )
    score_df = score_df.reset_index().rename(columns={"index": "post_process"})
    score_df.to_csv(save_path, index=False)
    logging.info(f"Successfully saved at {save_path}")


def concat_scores(args):
    df_list = []
    for agg_checkpoint in args.agg_checkpoints:
        logging.info(f"Loaded file is {agg_checkpoint}.")
        df_list.append(pd.read_csv(agg_checkpoint))
    score_df = pd.concat(df_list, axis=0)
    save_path = "/".join(agg_checkpoint.split("/")[:-2] + [f"score{args.feature}.csv"])
    score_df.to_csv(save_path, index=False)
    logging.info(f"Concatenated file is saved at {save_path}.")


if __name__ == "__main__":
    args = get_args()
    if args.concat:
        concat_scores(args)
    else:
        main(args)
