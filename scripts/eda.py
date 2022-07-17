# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio
import h5py
import librosa
import soundfile as sf
from statistics import mean
from IPython.core.display import display

# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
# machine = machines[0]
for machine in machines:
    with open(f"dump/dev/{machine}/test/eval.scp", "r") as f:
        eval_list = [s.strip() for s in f.readlines()]
    df = pd.DataFrame(eval_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    df["is_anomaly"] = df["path"].map(lambda x: int("anomaly" in x.split("_")[0]))
    df["is_normal"] = ~df["is_anomaly"] + 2
    df["is_dev"] = df["path"].map(lambda x: int("dev" in x.split("_")[0]))
    print(machine)
    display(
        df[["section", "is_dev", "is_anomaly", "is_normal"]].groupby(by="section").sum()
    )
# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
for machine in machines:
    dir_list = os.listdir(f"downloads/dev/{machine}/train")
    df = pd.DataFrame(dir_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    print("train", machine, sorted(df["section"].unique()))

    dir_list = os.listdir(f"downloads/dev/{machine}/test")
    df = pd.DataFrame(dir_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    print("dev/test", machine, sorted(df["section"].unique()))
    dir_list = os.listdir(f"downloads/eval/{machine}/test")
    df = pd.DataFrame(dir_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    print("eval/test", machine, sorted(df["section"].unique()))
# %%
a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
dev_sec = [1, 2, 3]
dev_idx = np.zeros(len(df)).astype(bool)
for sec in dev_sec:
    dev_idx |= df["section"] == sec
# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

n_top = -1
df_list = []
score_list = sorted(glob.glob("exp/all/**/score_embed.csv"))
for score_path in score_list:
    if n_top > 0:
        df = pd.read_csv(score_path)
        df.sort_values(by="eval_hauc", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_list.append(df.iloc[:n_top])
    else:
        df = pd.read_csv(score_path)
        df_list.append(df)
df = pd.concat(df_list)
df.sort_values(by="eval_hauc", ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)
df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
df["valid"] = df["path"].map(lambda x: float(x.split("_")[2]))
df["pow"] = df["path"].map(lambda x: int(int(x.split("/")[2].split("_")[3][1:])))
# %%
machine = machines[0]
columns = [
    "post_process",
    "path",
    f"eval_{machine}_hauc",
    f"dev_{machine}_hauc",
    f"dev_{machine}_mauc",
]
sorted_df = df.sort_values(by=f"eval_{machine}_hauc", ascending=False)[columns]
# %%
df0 = pd.read_csv("exp/all/audioset_v000_0.1_p0/score_embed.csv")
df0.sort_values(by="eval_hauc", ascending=False, inplace=True)
df0.reset_index(drop=True, inplace=True)
# %%
df4 = pd.read_csv("exp/all/audioset_v000_0.15_p0/score_embed.csv")
df4.sort_values(by="eval_hauc", ascending=False, inplace=True)
df4.reset_index(drop=True, inplace=True)
df4.head(10)
# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
df_list = []
for rate in [0.1, 0.15]:
    score_list = [
        f"exp/all/audioset_v000_{rate}/checkpoint-100epochs/score_embed.csv",
        # "exp/all/audioset_v001_0.15_p0/checkpoint-100epochs/score_embed.csv",
    ]
    for score_path in score_list:
        df = pd.read_csv(score_path)
        df_list.append(df)
    df = pd.concat(df_list)
    df.sort_values(by="eval_hauc", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
    df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
    # df["pow"] = df["path"].map(lambda x: int(int(x.split("/")[2].split("_")[3][1:])))
    df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
    df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
    df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])

    auc_list = []  # AUC,pAUC,mAUC
    auc_cols = []
    mauc_cols = []
    for machine in machines:
        sorted_df = df[
            (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 1)
        ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
        sorted_df.reset_index(drop=True, inplace=True)
        dev_cols = [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
        dev_cols = [f"dev_{machine}_mauc"]
        auc_cols += [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
        mauc_cols.append(f"dev_{machine}_mauc")
        auc_list += list(sorted_df.loc[0, dev_cols].values * 100)
        print(sorted_df.loc[0, "post_process"])
    print(auc_list)
    print(mean(auc_list))
# %%


fname = "/home/i_kuroyanagi/workspace2/laboratory/asd_recipe/scripts/downloads/dev/fan/test/anomaly_id_00_00000000.wav"
# fname = "/home/i_kuroyanagi/workspace2/laboratory/asd_recipe/scripts/downloads/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good/pure_0.wav"
# fname - "/home/i_kuroyanagi/workspace2/laboratory/asd_recipe/scripts/downloads/UAV-Propeller-Anomaly-Audio-Dataset/Dataset/1_broken/0001/ccw_f8/1/ch1.wav"
x, orig_sr = sf.read(fname)
x = librosa.resample(x, orig_sr=orig_sr, target_sr=16000)
# %%


fname = "/home/i_kuroyanagi/workspace2/laboratory/asd_recipe/scripts/dump/uav/Dataset-1_broken-0001-ccw_f8-1-ch1.h5"
hdf5_file = h5py.File(fname, "r")
# %%
wave = torch.tensor([hdf5_file["wave63"][()], hdf5_file["wave64"][()]])
n_fft = 2048
hop_length = 256
power = 1.0
# メルスペクトログラム
melspectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    hop_length=256,
    f_min=0.0,
    f_max=8000.0,
    pad=0,
    n_mels=224,
    power=1.0,
    normalized=True,
    center=True,
    pad_mode="reflect",
    onesided=True,
)
# スペクトログラム
melspectrogram1 = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    f_min=0.0,
    f_max=8000.0,
    pad=0,
    n_mels=224,
    power=1.0,
    normalized=True,
    center=True,
    pad_mode="reflect",
    onesided=True,
)

# ログメルスペクトログラム
amplitude2db = T.AmplitudeToDB(stype=1)


w0 = melspectrogram(wave)
w1 = amplitude2db(w0)
w2 = amplitude2db(melspectrogram1(wave))
# %%
plt.figure()
plt.imshow(w0[0], aspect="auto")
plt.colorbar()
plt.figure()
plt.imshow(w1[0], aspect="auto")
plt.colorbar()
plt.figure()
plt.imshow(w2[0], aspect="auto")
plt.colorbar()
plt.figure()
plt.imshow(w1[0] - w2[0], aspect="auto")
plt.colorbar()
