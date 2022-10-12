# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from IPython.core.display import display
import subprocess
from statistics import mean

machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

# %%
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
no = 208
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
# DDCSAD のスコアと一緒にプロット
# %%
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
no = 120
# no_list = ["120", "126", "127", "128", "130"]
no_list = ["120", "108", "129"]
no_list = ["220", "208", "229"]
seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
ddcsad_score_list = np.zeros((len(machines), len(n_anomaly_list), len(seed_list)))
anomaly_score_list = np.zeros(
    (len(no_list), len(machines), len(n_anomaly_list), len(seed_list))
)
for l, no in enumerate(no_list):
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
                anomaly_score_list[l, i, j, k] = auc_pauc
                if int(no) >= 200:
                    step = "4000"
                else:
                    step = "5000"
                ddcsad_no = no[0]
                ddcsad_path = f"/home/i_kuroyanagi/workspace2/laboratory/ParallelWaveGAN/egs/dcase2020-task2/greedy/exp2/{machine}/bin/ResNet38Double/v{ddcsad_no}80-anomaly_seed{seed}/scratch/anomaly{n_anomaly}/dev/checkpoint-{step}steps/split10frame256_0.5/agg.csv"
                ddcsad_df = pd.read_csv(ddcsad_path)
                ddcsad_score_list[i, j, k] = (
                    ddcsad_df.loc[
                        0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
                    ].values.mean()
                    * 100
                )
use_one_fig = True
for i, machine in enumerate(machines):
    title_no = ""
    if use_one_fig:
        fig, ax = plt.subplots()
    for l, no in enumerate(no_list):
        title_no += f"{no}_"
        if not use_one_fig:
            fig, ax = plt.subplots()
        yerr_se = anomaly_score_list[l, i].std(1) / np.sqrt(len(seed_list))
        ddcsad_se = ddcsad_score_list[i].std(1) / np.sqrt(len(seed_list))
        x = np.arange(len(n_anomaly_list))
        y_mean = anomaly_score_list[l, i].mean(1)
        ddcsad_mean = ddcsad_score_list[i].mean(1)
        ax.plot(x, y_mean, marker="o", label=f"{no} {machine}")
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
    ax.plot(x, ddcsad_mean, marker="o", label=f"DDCSAD {machine}")
    ax.errorbar(
        x,
        ddcsad_mean,
        yerr=ddcsad_se,
        capsize=3,
        fmt="o",
        ecolor="k",
        ms=7,
        mfc="None",
        mec="k",
    )
    if use_one_fig:
        ax.set_xticklabels([0] + n_anomaly_list)
        ax.set_ylim()
        ax.set_xlabel("The number of anomaly samples")
        ax.set_ylabel("ave-AUC [%]")
        ax.set_title(f"The number of anomalous data vs. {machine} ASD performance")
        os.makedirs("exp/fig", exist_ok=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"exp/fig/{title_no}w_ddcsad_{machine}.png")
if use_one_fig:
    fig, ax = plt.subplots()
for l, no in enumerate(no_list):
    yerr_se = anomaly_score_list[l].mean(0).std(1) / np.sqrt(len(seed_list))
    x = np.arange(len(n_anomaly_list))
    y_mean = anomaly_score_list[l].mean(0).mean(1)
    ax.plot(x, y_mean, marker="o", label=f"{no} average")
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
ddcsad_se = ddcsad_score_list.mean(0).std(1) / np.sqrt(len(seed_list))
ddcsad_mean = ddcsad_score_list.mean(0).mean(1)
ax.plot(x, ddcsad_mean, marker="o", label="DDCSAD average")
ax.errorbar(
    x,
    ddcsad_mean,
    yerr=ddcsad_se,
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
plt.savefig(f"exp/fig/{title_no}w_ddcsad_all.png")
# %%
# mAUCの性能をプロット
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
no = 120
# no_list = ["120", "126", "127", "128", "130"]
no_list = ["120", "108", "129"]
no_list = ["220", "208", "229"]
seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
ddcsad_score_list = np.zeros((len(machines), len(n_anomaly_list), len(seed_list)))
anomaly_score_list = np.zeros(
    (len(no_list), len(machines), len(n_anomaly_list), len(seed_list))
)
for l, no in enumerate(no_list):
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
                anomaly_score_list[l, i, j, k] = (
                    sorted_df.loc[0, [f"dev_{machine}_mauc"]] * 100
                )
                if int(no) >= 200:
                    step = "4000"
                else:
                    step = "5000"
                ddcsad_no = no[0]
                ddcsad_path = f"/home/i_kuroyanagi/workspace2/laboratory/ParallelWaveGAN/egs/dcase2020-task2/greedy/exp2/{machine}/bin/ResNet38Double/v{ddcsad_no}80-anomaly_seed{seed}/scratch/anomaly{n_anomaly}/dev/checkpoint-{step}steps/split10frame256_0.5/agg.csv"
                ddcsad_df = pd.read_csv(ddcsad_path)
                ddcsad_score_list[i, j, k] = (
                    ddcsad_df.loc[0, [f"dev_{machine}_mauc"]] * 100
                )
use_one_fig = True

for i, machine in enumerate(machines):
    title_no = ""
    if use_one_fig:
        fig, ax = plt.subplots()
    for l, no in enumerate(no_list):
        title_no += f"{no}_"
        if not use_one_fig:
            fig, ax = plt.subplots()
        yerr_se = anomaly_score_list[l, i].std(1) / np.sqrt(len(seed_list))
        ddcsad_se = ddcsad_score_list[i].std(1) / np.sqrt(len(seed_list))
        x = np.arange(len(n_anomaly_list))
        y_mean = anomaly_score_list[l, i].mean(1)
        ddcsad_mean = ddcsad_score_list[i].mean(1)
        ax.plot(x, y_mean, marker="o", label=f"{no} {machine}")
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
    ax.plot(x, ddcsad_mean, marker="o", label=f"DDCSAD {machine}")
    ax.errorbar(
        x,
        ddcsad_mean,
        yerr=ddcsad_se,
        capsize=3,
        fmt="o",
        ecolor="k",
        ms=7,
        mfc="None",
        mec="k",
    )
    if use_one_fig:
        ax.set_xticklabels([0] + n_anomaly_list)
        ax.set_ylim()
        ax.set_xlabel("The number of anomaly samples")
        ax.set_ylabel("ave-AUC [%]")
        ax.set_title(f"The number of anomalous data vs. {machine} mAUC")
        os.makedirs("exp/fig", exist_ok=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"exp/fig/{title_no}w_ddcsad_{machine}_mAUC.png")
if use_one_fig:
    fig, ax = plt.subplots()
for l, no in enumerate(no_list):
    yerr_se = anomaly_score_list[l].mean(0).std(1) / np.sqrt(len(seed_list))
    x = np.arange(len(n_anomaly_list))
    y_mean = anomaly_score_list[l].mean(0).mean(1)
    ax.plot(x, y_mean, marker="o", label=f"{no} average")
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
ddcsad_se = ddcsad_score_list.mean(0).std(1) / np.sqrt(len(seed_list))
ddcsad_mean = ddcsad_score_list.mean(0).mean(1)
ax.plot(x, ddcsad_mean, marker="o", label="DDCSAD average")
ax.errorbar(
    x,
    ddcsad_mean,
    yerr=ddcsad_se,
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
ax.set_title("The number of anomalous data vs. mAUC")
plt.legend()
plt.tight_layout()
plt.savefig(f"exp/fig/{title_no}w_ddcsad_all_mAUC.png")

# %%
# 外れ値データの使用割合と性能の関係
no = "029"
col_name = "machine"
# col_name = "outlier"
threshold_list = [
    # 0,
    # 0.05,
    0.1,
    # 0.2,
    0.3,
    # 0.4,
    0.5,
    # 0.6,
    0.7,
    # 0.8,
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
feature = "_embed"
# feature = "_prediction"
no = "029"
col_name = "machine2"
# col_name = "outlier2"
if col_name == "machine":
    threshold_list = [
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
    ]
elif col_name in ["section2", "outlier2"]:
    threshold_list = [
        0,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        # 0.3,
        0.5,
        # 0.7,
        1,
    ]
else:
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
        0.995,
        0.999,
        # 0.9995,
        # 0.9999,
        1,
    ]
# seed_list = [4]
seed_list = [0, 1, 2, 3, 4]
anomaly_score_list = np.zeros((len(machines), len(threshold_list), len(seed_list)))
outliers_list = np.zeros((len(machines), len(threshold_list), len(seed_list)))
for k, seed in enumerate(seed_list):
    for j, threshold in enumerate(threshold_list):
        score_path = f"exp/all/audioset_v{no}_{col_name}{threshold}_0.15_seed{seed}/checkpoint-100epochs/score{feature}.csv"
        if ((threshold == 0) and (col_name == ["outlier2", "section2", "machine"])) or (
            (threshold == 1) and (col_name in ["outlier", "section", "machine2"])
        ):
            score_path = f"exp/all/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/score{feature}.csv"
        # if (
        #     (threshold == 1)
        #     and (col_name == "machine")
        #     or ((threshold == 0) and (col_name in ["outlier", "section"]))
        # ):
        #     score_path = f"exp/all/audioset_v{no}_outlier0_0.15_seed{seed}/checkpoint-100epochs/score{feature}.csv"
        df = pd.read_csv(score_path)
        df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
        # df["valid"] = df["path"].map(lambda x: float(x.split("_")[3].split("/")[0]))
        df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
        df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
        df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
        for i, machine in enumerate(machines):
            outliers_list[i, j, k] = int(
                subprocess.check_output(
                    [
                        "wc",
                        "-l",
                        f"exp/{machine}/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/{col_name}_{threshold}.scp",
                    ]
                )
                .decode()
                .split(" ")[0]
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
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    ax.set_xlabel(
        f"The number of outlier samples, sampled by {col_name} feature:{feature}"
    )
    ax.set_ylabel("The number of outliers")
    ax.grid(True)
    ax2.set_ylabel("ave-AUC [%]")
    if not use_one_fig:
        ax.set_xticklabels(threshold_list)
        ax.set_title(f"The number of outlier data vs. {machine}'s performance")
        os.makedirs("exp/fig", exist_ok=True)
        plt.savefig(f"exp/fig/{col_name}_{no}{feature}_{machine}.png")

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
ax.set_xlabel(f"The number of outlier samples, sampled by {col_name} feature:{feature}")
ax2.set_ylabel("ave-AUC [%]")
ax.set_title("The number of anomalous data vs. all ASD performance")
plt.legend()
plt.tight_layout()
plt.savefig(f"exp/fig/{col_name}_{no}{feature}_all.png")

# %%
# %%

no = "020"
col_name = "section2"
if col_name in ["outlier", "section"]:
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
elif col_name in ["outlier2", "section2"]:
    threshold_list = [
        0,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        # 0.3,
        0.5,
        # 0.7,
        1,
    ]
seed_list = [0, 1, 2, 3, 4]
# seed_list = [0]
outliers_list = np.zeros((len(machines), len(threshold_list), len(seed_list)))
for k, seed in enumerate(seed_list):
    for j, threshold in enumerate(threshold_list):
        for i, machine in enumerate(machines):
            outliers_list[i, j, k] = int(
                subprocess.check_output(
                    [
                        "wc",
                        "-l",
                        f"exp/{machine}/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/{col_name}_{threshold}.scp",
                    ]
                )
                .decode()
                .split(" ")[0]
            )
use_one_fig = False
if use_one_fig:
    fig, ax = plt.subplots(figsize=(10, 4))
for i, machine in enumerate(machines):
    x = np.arange(len(threshold_list))
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
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc="lower left")
    ax.set_xlabel(f"The number of outlier samples, sampled by {col_name}")
    ax.set_ylabel("The number of outliers")
    ax.grid(True)
    if not use_one_fig:
        ax.set_xticklabels(threshold_list)
        ax.set_title(f"The number of outlier data vs. {machine}'s performance")
        os.makedirs("exp/fig", exist_ok=True)
        plt.savefig(f"exp/fig/hist_{col_name}_{no}_{machine}.png")

outlier = []
outlier_se = []
for i in range(len(threshold_list)):
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

ax.set_xticklabels(threshold_list)
ax.set_ylim()
ax.set_xlabel(f"The number of outlier samples, sampled by {col_name}")
ax.set_title("The number of anomalous data vs. all ASD performance")
plt.legend()
plt.tight_layout()

# %%
n_train = []
eval_list = np.zeros((len(machines), 2))
for k, machine in enumerate(machines):
    cmd = f"wc -l dump/dev/{machine}/train/dev.scp"
    c = subprocess.check_output(cmd.split()).decode().split()[0]
    n_train.append(int(c))
    if machine in ["fan", "pump", "slider", "valve"]:
        dev_section = [0, 2, 4, 6]
        # dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyCar":
        dev_section = [0, 1, 2, 3]
        # dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyConveyor":
        dev_section = [0, 1, 2]
        # dev_section = [0, 1, 2, 3, 4, 5]
    for i, s in enumerate(dev_section):
        for j, state in enumerate(["normal", "anomaly"]):
            eval_list[k, j] += len(
                glob.glob(f"dump/dev/{machine}/test/{state}_id_{s:02}_*")
            ) / len(dev_section)
# %%
