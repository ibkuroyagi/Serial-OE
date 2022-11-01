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
method_list = ["Serial-OE", "Serial Method", r"w/o $L_{\rm product}$"]
n_normal = 1000

# %%
# DDCSAD のスコアと一緒にプロット
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
ex_name = "use"
# ex_name = "contaminate"

if ex_name == "use":
    no_list = ["120", "108", "129"]
elif ex_name == "contaminate":
    no_list = ["220", "208", "229"]

seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
# anomaly_percent = [f"{per/(n_normal)*100:.1f}" for per in n_anomaly_list]
anomaly_percent = [per / (n_normal) * 100 for per in n_anomaly_list]

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
fontsize = 19
figsize = (8, 9)
title_no = ""
if use_one_fig:
    fig, ax = plt.subplots(figsize=figsize)
for l, method in enumerate(method_list):
    yerr_se = anomaly_score_list[l].mean(0).std(1) / np.sqrt(len(seed_list))
    x = np.arange(len(n_anomaly_list))
    y_mean = anomaly_score_list[l].mean(0).mean(1)
    ax.plot(anomaly_percent, y_mean, marker="o", label=f"{method}")
    ax.errorbar(
        anomaly_percent,
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
ax.plot(anomaly_percent, ddcsad_mean, marker="o", label="DDCSAD")
ax.errorbar(
    anomaly_percent,
    ddcsad_mean,
    yerr=ddcsad_se,
    capsize=3,
    fmt="o",
    ecolor="k",
    ms=7,
    mfc="None",
    mec="k",
)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.set_xlabel("The ratio of anomalous data to normal data [%]", fontsize=fontsize)
ax.set_ylabel("The average AUC [%]", fontsize=fontsize)
ax.grid()
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"exp/fig/paper_{ex_name}_aAUC_all.png")
# %%
# mAUCの性能をプロット
# 異常スコアの変遷をエラーバー(標準誤差,SE)をプロットする
ex_name = "use"
# ex_name = "contaminate"

if ex_name == "use":
    no_list = ["120", "108", "129"]
elif ex_name == "contaminate":
    no_list = ["220", "208", "229"]
seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
anomaly_percent = [per / (n_normal) * 100 for per in n_anomaly_list]

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
if use_one_fig:
    fig, ax = plt.subplots(figsize=figsize)
for l, no in enumerate(method_list):
    yerr_se = anomaly_score_list[l].mean(0).std(1) / np.sqrt(len(seed_list))
    x = np.arange(len(n_anomaly_list))
    y_mean = anomaly_score_list[l].mean(0).mean(1)
    ax.plot(anomaly_percent, y_mean, marker="o", label=f"{no}")
    ax.errorbar(
        anomaly_percent,
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
ax.plot(anomaly_percent, ddcsad_mean, marker="o", label="DDCSAD")
ax.errorbar(
    anomaly_percent,
    ddcsad_mean,
    yerr=ddcsad_se,
    capsize=3,
    fmt="o",
    ecolor="k",
    ms=7,
    mfc="None",
    mec="k",
)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.set_xlabel("The ratio of anomalous data to normal data [%]", fontsize=fontsize)
ax.set_ylabel("The minimum AUC [%]", fontsize=fontsize)
ax.grid()
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"exp/fig/paper_{ex_name}_mAUC_all.png")

# %%
# スコアの比を計算
ex_name = "use"
ex_name = "contaminate"

if ex_name == "use":
    no_list = ["120", "108", "129"]
elif ex_name == "contaminate":
    no_list = ["220", "208", "229"]

seed_list = [0, 1, 2, 3, 4]
n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
# anomaly_percent = [f"{per/(n_normal)*100:.1f}" for per in n_anomaly_list]
anomaly_percent = [per / (n_normal) * 100 for per in n_anomaly_list]

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
serial_oe_mean = anomaly_score_list[0].mean(0).mean(1)
for l, method in enumerate(method_list):
    y_mean = anomaly_score_list[l].mean(0).mean(1)
    print((serial_oe_mean / y_mean - 1) * 100)
ddcsad_mean = ddcsad_score_list.mean(0).mean(1)
print((serial_oe_mean / ddcsad_mean - 1) * 100)
# %%
