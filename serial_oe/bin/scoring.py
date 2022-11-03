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
from statistics import mean
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def get_args():
    parser = argparse.ArgumentParser(description="Scoring predicted scores.")
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
    hdev_columns = []
    heval_columns = []
    dev_columns = []
    eval_columns = []
    pdev_columns = []
    peval_columns = []
    modes = ["dev", "eval"]
    for machine in ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]:
        hdev_columns += [f"dev_{machine}_hauc"]
        heval_columns += [f"eval_{machine}_hauc"]
        dev_columns += [f"dev_{machine}_auc"]
        eval_columns += [f"eval_{machine}_auc"]
        pdev_columns += [f"dev_{machine}_pauc"]
        peval_columns += [f"eval_{machine}_pauc"]

    agg_df = pd.read_csv(args.agg_checkpoints[0])
    post_processes = list(agg_df.columns)
    for rm in ["path", "is_normal", "product", "mode"]:
        post_processes.remove(rm)
    columns = [
        "path",
        "dev_hauc",
        "eval_hauc",
        "hauc",
        "dev_auc",
        "eval_auc",
        "auc",
        "dev_pauc",
        "eval_pauc",
    ]
    columns += (
        heval_columns
        + hdev_columns
        + eval_columns
        + dev_columns
        + peval_columns
        + pdev_columns
    )
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
        if machine == "ToyCar":
            dev_products = [0, 1, 2, 3]
            eval_products = [4, 5, 6]
        elif machine == "ToyConveyor":
            dev_products = [0, 1, 2]
            eval_products = [3, 4, 5]
        else:
            dev_products = [0, 2, 4, 6]
            eval_products = [1, 3, 5]
        products = {"dev": dev_products, "eval": eval_products}
        for post_process in post_processes:
            for mode in modes:
                auc_list = []
                pauc_list = []
                mauc = 1.1
                for product in products[mode]:
                    target_idx = agg_df["product"] == product
                    auc = roc_auc_score(
                        1 - agg_df.loc[target_idx, "is_normal"],
                        -agg_df.loc[target_idx, post_process],
                    )
                    auc_list.append(auc)
                    score_df.loc[post_process, f"{machine}_{product}_auc"] = auc
                    if mauc > auc:
                        mauc = auc
                    pauc = roc_auc_score(
                        1 - agg_df.loc[target_idx, "is_normal"],
                        -agg_df.loc[target_idx, post_process],
                        max_fpr=0.1,
                    )
                    pauc_list.append(pauc)
                    score_df.loc[post_process, f"{machine}_{product}_pauc"] = pauc
                score_df.loc[post_process, f"{mode}_{machine}_mauc"] = mauc
                score_df.loc[post_process, f"{mode}_{machine}_auc"] = mean(auc_list)
                score_df.loc[post_process, f"{mode}_{machine}_pauc"] = mean(pauc_list)
                score_list = auc_list + pauc_list
                score_df.loc[post_process, f"{mode}_{machine}_hauc"] = hmean(score_list)
                score_df.loc[post_process, f"{mode}_{machine}_hauc_std"] = np.array(
                    score_list
                ).std()
    for post_process in post_processes:
        score_df.loc[post_process, "dev_hauc"] = hmean(
            score_df.loc[post_process, hdev_columns].values.flatten()
        )
        score_df.loc[post_process, "eval_hauc"] = hmean(
            score_df.loc[post_process, heval_columns].values.flatten()
        )
        score_df.loc[post_process, "hauc"] = hmean(
            score_df.loc[post_process, hdev_columns + heval_columns].values.flatten()
        )
        score_df.loc[post_process, "dev_auc"] = mean(
            score_df.loc[post_process, dev_columns].values.flatten()
        )
        score_df.loc[post_process, "eval_auc"] = mean(
            score_df.loc[post_process, eval_columns].values.flatten()
        )
        score_df.loc[post_process, "auc"] = mean(
            score_df.loc[post_process, dev_columns + eval_columns].values.flatten()
        )
        score_df.loc[post_process, "dev_pauc"] = mean(
            score_df.loc[post_process, pdev_columns].values.flatten()
        )
        score_df.loc[post_process, "eval_pauc"] = mean(
            score_df.loc[post_process, peval_columns].values.flatten()
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
