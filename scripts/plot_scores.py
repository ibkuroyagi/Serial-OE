# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from IPython.core.display import display


machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

# %%
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
no = 200
seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
anomaly_score_list = np.zeros((len(machines), len(n_anomaly_list), len(seed_list)))
for k, seed in enumerate(seed_list):
    for j, n_anomaly in enumerate(n_anomaly_list):
        score_path = f"exp/all/audioset_v{no}_0.15_anomaly{n_anomaly}_max6_seed{seed}/checkpoint-100epochs/score_embed.csv"
        df = pd.read_csv(score_path)
        df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
        df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
        df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
        df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
        df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
        for i, machine in enumerate(machines):
            sorted_df = df[
                (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 2)
            ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
            sorted_df.reset_index(drop=True, inplace=True)
            auc_pauc = (
                sorted_df.loc[
                    0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
                ].values.mean()
                * 100
            )
            # print(seed, n_anomaly, machine, auc_pauc)
            anomaly_score_list[i, j, k] = auc_pauc
# %%
use_one_fig = False
if use_one_fig:
    fig, ax = plt.subplots()
for i, machine in enumerate(machines):
    yerr_se = anomaly_score_list[i].std(1) / np.sqrt(len(seed_list))
    x = np.arange(len(n_anomaly_list))
    y_mean = anomaly_score_list[i].mean(1)
    if not use_one_fig:
        fig, ax = plt.subplots()
    ax.plot(x, y_mean, marker="o", label=machine)
    ax.errorbar(
        x,
        y_mean,
        yerr=yerr_se,
        capsize=3,
        fmt="o",
        ecolor="k",
        ms=7,
        mfc="None",
        mec="k",
    )
    if not use_one_fig:
        ax.set_xticklabels([0] + n_anomaly_list)
        ax.set_ylim()
        ax.set_xlabel("The number of anomaly samples")
        ax.set_ylabel("ave-AUC [%]")
        ax.set_title(f"The number of anomalous data vs. {machine} ASD performance")
        os.makedirs(f"exp/fig", exist_ok=True)
        plt.savefig(f"exp/fig/{no}_{machine}.png")

yerr_se = anomaly_score_list.mean(0).std(1) / np.sqrt(len(seed_list))
x = np.arange(len(n_anomaly_list))
y_mean = anomaly_score_list.mean(0).mean(1)
if not use_one_fig:
    fig, ax = plt.subplots()
ax.plot(x, y_mean, marker="o", label="average")
ax.errorbar(
    x,
    y_mean,
    yerr=yerr_se,
    capsize=3,
    fmt="o",
    ecolor="k",
    ms=7,
    mfc="None",
    mec="k",
)
ax.set_xticklabels([0] + n_anomaly_list)
ax.set_ylim()
ax.set_xlabel("The number of anomaly samples")
ax.set_ylabel("ave-AUC [%]")
ax.set_title("The number of anomalous data vs. all ASD performance")
plt.legend()
plt.tight_layout()
plt.savefig(f"exp/fig/{no}_all.png")

# %%
# 外れ値データの使用割合と性能の関係
no = "000"
col_name = "section"
# col_name = "outlier"
threshold_list = [
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
]

anomaly_score_list = np.zeros((len(machines), len(threshold_list)))
outliers_list = np.zeros((len(machines), len(threshold_list)))
for j, threshold in enumerate(threshold_list):
    score_path = f"exp/all/audioset_v{no}_{col_name}{threshold}_0.15_seed0/checkpoint-100epochs/score_embed.csv"

    df = pd.read_csv(score_path)
    df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
    df["valid"] = df["path"].map(lambda x: float(x.split("_")[3].split("/")[0]))
    df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
    df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
    df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
    for i, machine in enumerate(machines):
        outliers_list[i, j] = sum(
            1
            for line in open(
                f"exp/{machine}/audioset_v{no}_0.15/checkpoint-100epochs/{col_name}_{threshold}.scp"
            )
        )
        sorted_df = df[
            (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 2)
        ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
        sorted_df.reset_index(drop=True, inplace=True)
        auc_pauc = (
            sorted_df.loc[
                0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
            ].values.mean()
            * 100
        )
        # print(threshold, machine, auc_pauc)
        anomaly_score_list[i, j] = auc_pauc
# %%
use_one_fig = False
if use_one_fig:
    fig, ax = plt.subplots(figsize=(10, 4))
for i, machine in enumerate(machines):
    x = np.arange(len(threshold_list))
    y_mean = anomaly_score_list[i]
    if not use_one_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    ln1 = ax.bar(x, outliers_list[i], tick_label=threshold_list, log=True, alpha=0.5)
    ax2 = ax.twinx()
    ln2 = ax2.plot(x, y_mean, marker="o", label=machine, c="r")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    ax.set_xlabel("Thresholds")
    ax.set_ylabel("The number of outliers")
    ax.grid(True)
    ax2.set_ylabel("ave-ACU [%]")

    if not use_one_fig:
        ax.set_xticklabels(threshold_list)
        ax.set_title(f"The number of outlier data vs. {machine}'s performance")
        os.makedirs("exp/fig", exist_ok=True)
        plt.savefig(f"exp/fig/{no}_{machine}_{col_name}.png")

yerr_se = anomaly_score_list.std(0) / np.sqrt(len(threshold_list))
x = np.arange(len(threshold_list))
y_mean = anomaly_score_list.mean(0)
if not use_one_fig:
    fig, ax = plt.subplots()
ax.plot(x, y_mean, marker="o", label="average")
ax.errorbar(
    x,
    y_mean,
    yerr=yerr_se,
    capsize=3,
    fmt="o",
    ecolor="k",
    ms=7,
    mfc="None",
    mec="k",
)
ax.set_xticklabels(threshold_list)
ax.set_ylim()
ax.set_xlabel("The number of anomaly samples")
ax.set_ylabel("ave-AUC [%]")
ax.set_title("The number of anomalous data vs. all ASD performance")
plt.legend()
plt.tight_layout()
plt.savefig(f"exp/fig/{no}_all.png")
# %%
# seed with outlier or section
no = "000"
col_name = "section"
# col_name = "outlier"
threshold_list = [
    0,
    # 0.05,
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
    # 0.9995,
    # 0.9999,
    1,
]
seed_list = [0, 1, 2, 3, 4]
anomaly_score_list = np.zeros((len(machines), len(threshold_list), len(seed_list)))
outliers_list = np.zeros((len(machines), len(threshold_list), len(seed_list)))
for k, seed in enumerate(seed_list):
    for j, threshold in enumerate(threshold_list):
        score_path = f"exp/all/audioset_v{no}_{col_name}{threshold}_0.15_seed{seed}/checkpoint-100epochs/score_embed.csv"
        df = pd.read_csv(score_path)
        df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
        df["valid"] = df["path"].map(lambda x: float(x.split("_")[3].split("/")[0]))
        df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
        df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
        df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
        for i, machine in enumerate(machines):
            outliers_list[i, j, k] = sum(
                1
                for line in open(
                    f"exp/{machine}/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/{col_name}_{threshold}.scp"
                )
            )
            sorted_df = df[
                (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 2)
            ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
            sorted_df.reset_index(drop=True, inplace=True)
            auc_pauc = (
                sorted_df.loc[
                    0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
                ].values.mean()
                * 100
            )
            # print(threshold, machine, auc_pauc)
            anomaly_score_list[i, j, k] = auc_pauc
# %%
use_one_fig = False
if use_one_fig:
    fig, ax = plt.subplots(figsize=(10, 4))
for i, machine in enumerate(machines):
    x = np.arange(len(threshold_list))
    y_mean = anomaly_score_list[i].mean(1)
    yerr_se = anomaly_score_list[i].std(1) / np.sqrt(len(seed_list))
    if not use_one_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    ln1 = ax.bar(
        x,
        outliers_list[i].mean(1),
        yerr=outliers_list[i].std(1),
        tick_label=threshold_list,
        log=True,
        alpha=0.5,
    )
    ax2 = ax.twinx()
    ln2 = ax2.errorbar(
        x,
        y_mean,
        yerr=yerr_se,
        capsize=3,
        fmt="o",
        ecolor="k",
        ms=7,
        mfc="r",
        mec="k",
        label=machine,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_xlabel("Thresholds")
    ax.set_ylabel("The number of outliers")
    ax.grid(True)
    ax2.set_ylabel("ave-AUC [%]")
    if not use_one_fig:
        ax.set_xticklabels(threshold_list)
        ax.set_title(f"The number of outlier data vs. {machine}'s performance")
        os.makedirs(f"exp/fig", exist_ok=True)
        plt.savefig(f"exp/fig/{col_name}_{no}_{machine}.png")

score = []
score_se = []
outlier = []
outlier_se = []
for i in range(len(threshold_list)):
    score.append(anomaly_score_list[:, i, :].mean())
    score_se.append(
        anomaly_score_list[:, i, :].std() / np.sqrt(len(seed_list) * len(machines))
    )
    outlier.append(outliers_list[:, i, :].mean())
    outlier_se.append(
        outliers_list[:, i, :].std() / np.sqrt(len(seed_list) * len(machines))
    )
x = np.arange(len(threshold_list))
if not use_one_fig:
    fig, ax = plt.subplots(figsize=(10, 4))
ln1 = ax.bar(
    x,
    outlier,
    yerr=outlier_se,
    tick_label=threshold_list,
    log=True,
    alpha=0.5,
)
ax2 = ax.twinx()
ln2 = ax2.errorbar(
    x,
    score,
    yerr=score_se,
    capsize=3,
    fmt="o",
    ecolor="k",
    ms=7,
    mfc="r",
    mec="k",
)
ax.set_xticklabels(threshold_list)
ax.set_ylim()
ax.set_xlabel("The number of anomaly samples")
ax2.set_ylabel("ave-AUC [%]")
ax.set_title("The number of anomalous data vs. all ASD performance")
plt.legend()
plt.tight_layout()
plt.savefig(f"exp/fig/{col_name}_{no}_all.png")

# %%
