#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanagi

"""Inference models."""

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml
from serial_oe.utils import seed_everything, zscore
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def reset_columns(df):
    df_col = df.columns
    df = df.T.reset_index(drop=False).T
    for i in range(df.shape[1]):
        rename_col = {i: "_".join(df_col[i])}
        df = df.rename(columns=rename_col)
    df = df.drop(["level_0", "level_1"], axis=0).reset_index()
    return df


def upper_mean(series):
    """Average of values from maximum to median"""
    sorted_series = sorted(series)
    upper_elements = sorted_series[len(series) // 2 :]
    return np.mean(upper_elements)


def main():
    """Run inference process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in serial_oe/bin/train.py)."
    )
    parser.add_argument(
        "--pos_machine", type=str, required=True, help="Name of positive machine."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--use_norm",
        type=str,
        default="",
        help="If '_norm', normalize all input features.",
    )
    parser.add_argument("--feature", type=str, default="", help="Type of feature.")
    parser.add_argument(
        "--tail_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="Extracted embedding file for validation.",
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
    seed_everything(seed=config["seed"])
    n_section = 6 if args.pos_machine == "ToyConveyor" else 7
    if args.pos_machine in ["fan", "pump", "slider", "valve"]:
        dev_section = [0, 2, 4, 6]
    elif args.pos_machine == "ToyCar":
        dev_section = [0, 1, 2, 3]
    elif args.pos_machine == "ToyConveyor":
        dev_section = [0, 1, 2]
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    if args.feature == "":
        feature_cols = (
            ["pred_machine", "embed_norm"]
            + [f"pred_section{j}" for j in range(n_section)]
            + [f"e{i}" for i in range(config["model_params"]["embedding_size"])]
        )
    elif args.feature == "_embed":
        feature_cols = [
            f"e{i}" for i in range(config["model_params"]["embedding_size"])
        ]
    elif args.feature == "_prediction":
        feature_cols = ["pred_machine"] + [f"pred_section{j}" for j in range(n_section)]
    hyper_params = [2**i for i in range(6)]
    checkpoint_dirs = [os.path.dirname(checkpoint) for checkpoint in args.checkpoints]
    for checkpoint_dir in checkpoint_dirs:
        post_cols = []
        checkpoint = checkpoint_dir.split("/")[-1]
        checkpoint += args.tail_name
        logging.info(f"checkpoint_dir:{checkpoint_dir}")
        valid_df = pd.read_csv(os.path.join(checkpoint_dir, checkpoint + "_valid.csv"))
        eval_df = pd.read_csv(os.path.join(checkpoint_dir, checkpoint + "_eval.csv"))
        if args.feature == "_none":
            post_cols += ["pred_section"]
            for section_id in range(n_section):
                eval_df.loc[
                    eval_df["section"] == section_id, "pred_section"
                ] = eval_df.loc[
                    eval_df["section"] == section_id, f"pred_section{section_id}"
                ]
        else:
            for section_id in range(n_section):
                for used_set in ["all"]:
                    if used_set == "all":
                        input_valid = valid_df[(valid_df["section"] == section_id)][
                            feature_cols
                        ]
                    input_eval = eval_df[eval_df["section"] == section_id][feature_cols]
                    if args.use_norm == "_norm":
                        input_valid_mean = input_valid.mean()
                        input_valid_std = input_valid.std(ddof=0)
                        input_valid = (input_valid - input_valid_mean) / input_valid_std
                        input_eval = (input_eval - input_valid_mean) / input_valid_std
                    for hp in hyper_params:
                        lof = LocalOutlierFactor(n_neighbors=hp, novelty=True)
                        lof.fit(input_valid)
                        lof_score = lof.score_samples(input_eval)
                        eval_df.loc[
                            eval_df["section"] == section_id, f"LOF_{hp}_{used_set}"
                        ] = zscore(lof_score)
                        gmm = GaussianMixture(
                            n_components=hp, random_state=config["seed"]
                        )
                        gmm.fit(input_valid)
                        gmm_score = gmm.score_samples(input_eval) / 1000
                        eval_df.loc[
                            eval_df["section"] == section_id, f"GMM_{hp}_{used_set}"
                        ] = zscore(gmm_score)
                        ocsvm = OneClassSVM(nu=1 / (2 * hp), kernel="rbf")
                        ocsvm.fit(input_valid)
                        ocsvm_score = ocsvm.score_samples(input_eval)
                        eval_df.loc[
                            eval_df["section"] == section_id, f"OCSVM_{hp}_{used_set}"
                        ] = zscore(ocsvm_score).astype(float)
                        kde = KernelDensity(kernel="gaussian", bandwidth=hp)
                        kde.fit(input_valid)
                        kde_score = kde.score_samples(input_eval)
                        eval_df.loc[
                            eval_df["section"] == section_id, f"KDE_{hp}_{used_set}"
                        ] = zscore(kde_score).astype(float)
                        knn = NearestNeighbors(n_neighbors=hp, metric="euclidean")
                        knn.fit(input_valid)
                        knn_score = knn.kneighbors(input_eval)[0].mean(1)
                        eval_df.loc[
                            eval_df["section"] == section_id, f"KNN_{hp}_{used_set}"
                        ] = zscore(-knn_score)
                        if section_id == 0:
                            post_cols += [
                                f"LOF_{hp}_{used_set}",
                                f"GMM_{hp}_{used_set}",
                                f"OCSVM_{hp}_{used_set}",
                                f"KDE_{hp}_{used_set}",
                                f"KNN_{hp}_{used_set}",
                            ]
            post_cols.sort()
        columns = ["path", "section", "is_normal"] + post_cols
        logging.info(f"columns:{columns}")
        agg_df = (
            eval_df[columns].groupby("path").agg(["max", "mean", "median", upper_mean])
        )
        agg_df = reset_columns(agg_df)
        del (
            agg_df["is_normal_max"],
            agg_df["is_normal_mean"],
            agg_df["is_normal_upper_mean"],
            agg_df["section_max"],
            agg_df["section_mean"],
            agg_df["section_upper_mean"],
        )
        agg_df.rename(
            columns={"is_normal_median": "is_normal", "section_median": "section"},
            inplace=True,
        )
        dev_idx = np.zeros(len(agg_df)).astype(bool)
        for sec in dev_section:
            dev_idx |= agg_df["section"] == sec
        agg_df.loc[dev_idx, "mode"] = "dev"
        agg_df.loc[~dev_idx, "mode"] = "eval"
        agg_path = os.path.join(
            checkpoint_dir, checkpoint + f"{args.feature}{args.use_norm}_agg.csv"
        )
        agg_df.to_csv(agg_path, index=False)
        logging.info(f"Successfully saved {agg_path}.")


if __name__ == "__main__":
    main()
