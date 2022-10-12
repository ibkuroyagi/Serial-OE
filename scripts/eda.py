# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture
import h5py
import librosa
import soundfile as sf
from statistics import mean
from IPython.core.display import display
from asd_tools.utils import sigmoid
from asd_tools.utils import seed_everything
from asd_tools.utils import zscore
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import KernelDensity
from statistics import mean
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score


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


machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
# %%
hp = 1
pos_machine = machines[0]
checkpoint_dir = (
    f"exp/{pos_machine}/audioset_v000_machine0_0.15_seed0/checkpoint-100epochs"
)
checkpoint = checkpoint_dir.split("/")[-1]
feature_cols = [f"e{i}" for i in range(128)]
valid_df = pd.read_csv(os.path.join(checkpoint_dir, checkpoint + "_valid.csv"))
eval_df = pd.read_csv(os.path.join(checkpoint_dir, checkpoint + "_eval.csv"))
if pos_machine in ["fan", "pump", "slider", "valve"]:
    dev_section = [0, 2, 4, 6]
elif pos_machine == "ToyCar":
    dev_section = [0, 1, 2, 3]
elif pos_machine == "ToyConveyor":
    dev_section = [0, 1, 2]
for section_id in dev_section:
    input_valid = valid_df[(valid_df["section"] == section_id)][feature_cols]
    input_eval = eval_df[eval_df["section"] == section_id][feature_cols]
    kde = KernelDensity(kernel="gaussian", bandwidth=1 / hp)
    kde.fit(input_valid)
    kde_score = kde.score_samples(input_eval)
    eval_df.loc[eval_df["section"] == section_id, f"KDE_{hp}_all"] = zscore(kde_score)
agg_fc = ["max", "mean", "median", upper_mean]
columns = ["path", "section", "is_normal", f"KDE_{hp}_all"]
agg_df = eval_df[columns].groupby("path").agg(agg_fc)
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

auc_list = []
col = f"KDE_{hp}_all_upper_mean"
for section_id in dev_section:
    target_idx = agg_df["section"] == section_id
    print(section_id)
    auc = roc_auc_score(
        1 - agg_df.loc[target_idx, "is_normal"].values.astype(int),
        -agg_df.loc[target_idx, col].values.astype(float),
    )
    auc_list.append(auc)
print(f"{pos_machine}:{mean(auc_list)}")
# %%
# 推論専用スクリプトの生成
# 100epoch目のpklがあればそれをcheckpointにする
# confはcheckpointから生成
# featureは"_prediction"と"_embed"を実行
checkpoint_list = sorted(
    glob.glob("exp/fan/**/checkpoint-100epochs/checkpoint-100epochs.pkl")
)
for checkpoints in checkpoint_list:
    no = checkpoints.split("_")[1]
    tag = checkpoints.split("/")[2]
    conf = f"conf/tuning/asd_model.audioset_{no}.yaml"
    job = f"sbatch ./ex1.sh --conf {conf} --tag {tag}"
    print("sleep 12")
    print(job)
# %%
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
# noneの出力
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
seed_list = [0, 1, 2, 3, 4]
seed_auc_list = np.zeros((len(seed_list), len(machines) * 2))
seed_hauc_mauc_list = np.zeros((len(seed_list), len(machines) * 2))

no = "020"
for seed in seed_list:
    score_path = (
        f"exp/all/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/score_none.csv"
    )
    df = pd.read_csv(score_path)
    df.sort_values(by="eval_hauc", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
    df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
    df["agg"] = df["post_process"].map(lambda x: x.split("_")[2])
    auc_list = []  # AUC,pAUC,mAUC
    hauc_mauc_list = []
    for i, machine in enumerate(machines):
        sorted_df = df[(df["agg"] == "upper")].sort_values(
            by=f"eval_{machine}_hauc", ascending=False
        )
        sorted_df.reset_index(drop=True, inplace=True)
        auc_pauc = list(
            sorted_df.loc[0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]].values * 100
        )
        auc_list += auc_pauc
        hauc_mauc_list += [
            mean(auc_pauc),
            sorted_df.loc[0, f"dev_{machine}_mauc"] * 100,
        ]
    seed_auc_list[seed] = auc_list
    seed_hauc_mauc_list[seed] = hauc_mauc_list
# print(f"\n{no} hauc_mauc mean")
# for i in range(12):
#     print(seed_hauc_mauc_list.mean(0)[i], end=", ")
# print(f"\n{no} hauc_mauc SE")
# for i in range(12):
#     print(seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list)), end=", ")
print(f"\n{no} hauc_mauc")
for i in range(12):
    ave = seed_hauc_mauc_list.mean(0)[i]
    se = seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list))
    print(f"{ave:.2f}\pm{se:.2f}", end=", ")
ave = seed_hauc_mauc_list.mean(0)[::2].mean()
se = seed_hauc_mauc_list[:, ::2].std() / np.sqrt(len(seed_list) * len(machine))
print(f"{ave:.2f}\pm{se:.2f}", end=", ")
ave = seed_hauc_mauc_list.mean(0)[1::2].mean()
se = seed_hauc_mauc_list[:, 1::2].std() / np.sqrt(len(seed_list) * len(machine))
print(f"{ave:.2f}\pm{se:.2f}")

# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
seed_list = [0, 1, 2, 3, 4]
h_list = ["GMM", "KNN", "KDE", "LOF", "OCSVM"]
hp_list = [2, 32, 1, 32, 1]
seed_auc_list = np.zeros((len(seed_list), len(machines) * 2))
seed_hauc_mauc_list = np.zeros((len(seed_list), len(machines) * 2))

no = "008"
for hp_idx, h in enumerate(h_list):
    for seed in seed_list:
        score_path = f"exp/all/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/score_embed.csv"
        df = pd.read_csv(score_path)
        df.sort_values(by="eval_hauc", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
        df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
        df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
        df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
        df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
        # sorted_df = df[
        #     (df["h"] == h) & (df["agg"] == "upper")
        # ].sort_values(by=f"eval_hauc", ascending=False)
        # sorted_df.reset_index(drop=True, inplace=True)
        # hp_list.append(sorted_df.loc[0, ["hp"]])
        auc_list = []  # AUC,pAUC,mAUC
        hauc_mauc_list = []
        for i, machine in enumerate(machines):
            sorted_df = df[
                (df["h"] == h) & (df["agg"] == "upper") & (df["hp"] == hp_list[hp_idx])
            ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
            sorted_df.reset_index(drop=True, inplace=True)
            auc_pauc = list(
                sorted_df.loc[0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]].values
                * 100
            )
            auc_list += auc_pauc
            hauc_mauc_list += [
                mean(auc_pauc),
                sorted_df.loc[0, f"dev_{machine}_mauc"] * 100,
            ]
        seed_auc_list[seed] = auc_list
        seed_hauc_mauc_list[seed] = hauc_mauc_list
    # print(f"\n{no} hauc_mauc mean")
    # for i in range(12):
    #     print(seed_hauc_mauc_list.mean(0)[i], end=", ")
    # print(f"\n{no} hauc_mauc SE")
    # for i in range(12):
    #     print(seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list)), end=", ")
    print(f"\n{no} {h} hauc_mauc")
    for i in range(12):
        ave = seed_hauc_mauc_list.mean(0)[i]
        se = seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list))
        print(f"{ave:.2f}\pm{se:.2f}", end=", ")
    ave = seed_hauc_mauc_list.mean(0)[::2].mean()
    se = seed_hauc_mauc_list[:, ::2].std() / np.sqrt(len(seed_list) * len(machine))
    print(f"{ave:.2f}\pm{se:.2f}", end=", ")
    ave = seed_hauc_mauc_list.mean(0)[1::2].mean()
    se = seed_hauc_mauc_list[:, 1::2].std() / np.sqrt(len(seed_list) * len(machine))
    print(f"{ave:.2f}\pm{se:.2f}")
# %%
machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
seed_list = [0, 1, 2, 3, 4]
no_list = ["008", "019", "020", "021", "026", "027", "028", "029", "030"]
seed_auc_list = np.zeros((len(seed_list), len(machines) * 2))
seed_hauc_mauc_list = np.zeros((len(seed_list), len(machines) * 2))
for no in no_list:
    for seed in seed_list:
        score_path = f"exp/all/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/score_embed.csv"
        df = pd.read_csv(score_path)
        df.sort_values(by="eval_hauc", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
        df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
        # df["pow"] = df["path"].map(lambda x: int(int(x.split("/")[2].split("_")[3][1:])))
        df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
        df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
        df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])

        auc_list = []  # AUC,pAUC,mAUC
        hauc_mauc_list = []
        for i, machine in enumerate(machines):
            sorted_df = df[
                (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 2)
            ].sort_values(by=f"eval_{machine}_hauc", ascending=False)
            sorted_df.reset_index(drop=True, inplace=True)
            auc_pauc = list(
                sorted_df.loc[0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]].values
                * 100
            )
            auc_list += auc_pauc
            hauc_mauc_list += [
                mean(auc_pauc),
                sorted_df.loc[0, f"dev_{machine}_mauc"] * 100,
            ]
        seed_auc_list[seed] = auc_list
        seed_hauc_mauc_list[seed] = hauc_mauc_list
    print(f"\n{no} hauc_mauc mean")
    for i in range(12):
        print(seed_hauc_mauc_list.mean(0)[i], end=", ")
    print(f"\n{no} hauc_mauc SE")
    for i in range(12):
        print(seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list)), end=", ")
    # print(f"\n{no} hauc_mauc")
    # for i in range(12):
    #     ave = seed_hauc_mauc_list.mean(0)[i]
    #     se = seed_hauc_mauc_list.std(0)[i] / np.sqrt(len(seed_list))
    #     print(f"{ave:.2f}\pm{se:.2f}", end=", ")
    # ave = seed_hauc_mauc_list.mean(0)[::2].mean()
    # se = seed_hauc_mauc_list[:,::2].std() / np.sqrt(len(seed_list)*len(machine))
    # print(f"{ave:.2f}\pm{se:.2f}", end=", ")
    # ave = seed_hauc_mauc_list.mean(0)[1::2].mean()
    # se = seed_hauc_mauc_list[:,1::2].std() / np.sqrt(len(seed_list)*len(machine))
    # print(f"{ave:.2f}\pm{se:.2f}")
# %%
no = "020"
machine = "fan"
if machine in ["fan", "pump", "slider", "valve"]:
    dev_section = [0, 2, 4, 6]
elif machine == "ToyCar":
    dev_section = [0, 1, 2, 3]
elif machine == "ToyConveyor":
    dev_section = [0, 1, 2]
dev_cols = [f"{machine}_{i}_auc" for i in dev_section]
score_path = f"exp/all/audioset_v{no}_0.15_seed4/checkpoint-100epochs/score_embed.csv"
df = pd.read_csv(score_path)
df.sort_values(by="eval_hauc", ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)
df["no"] = df["path"].map(lambda x: int(x.split("_")[1][1:]))
df["valid"] = df["path"].map(lambda x: float(x.split("_")[2].split("/")[0]))
df["h"] = df["post_process"].map(lambda x: x.split("_")[0])
df["hp"] = df["post_process"].map(lambda x: int(x.split("_")[1]))
df["agg"] = df["post_process"].map(lambda x: x.split("_")[3])
sorted_df = df[
    (df["h"] == "GMM") & (df["agg"] == "upper") & (df["hp"] == 4)
].sort_values(by=f"eval_{machine}_hauc", ascending=False)
sorted_df.reset_index(drop=True, inplace=True)
display(sorted_df[dev_cols])
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
# %%
a = np.arange(10, 20)
b = np.random.choice(a, 20)
print(b)
# %%
path = "exp/fan/audioset_v000_0.1/checkpoint-100epochs/checkpoint-100epochs_valid.csv"
valid_df = pd.read_csv(
    "exp/fan/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_valid.csv"
)
# eval_df = pd.read_csv("exp/fan/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_eval.csv")
eval_df = pd.read_csv(
    "exp/fan/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_outlier_mean.csv"
)

# df_cols = pd.read_csv(path, header=0, nrows=1).columns
# df = pd.read_csv(path, header=0, skiprows=5, nrows=1500, names=df_cols)
df = pd.read_csv(path)

# %%
col = "pred_machine"
col = "pred_section_max"

for machine in machines:
    # agg_df = pd.read_csv(
    #     f"exp/{machine}/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_outlier_mean.csv"
    # )
    path = f"exp/{machine}/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_audioset_mean.csv"
    df_cols = pd.read_csv(path, header=0, nrows=1).columns
    if machine in ["fan", "pump", "slider", "valve"]:
        dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyCar":
        dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyConveyor":
        dev_section = [0, 1, 2, 3, 4, 5]
    agg_df = pd.read_csv(path, header=0, skiprows=5, nrows=100000, names=df_cols)
    agg_df["pred_section_max"] = agg_df[[f"pred_section{i}" for i in dev_section]].max(
        1
    )
    plt.figure(figsize=(6, 6))
    # plt.hist(agg_df[col], bins=50, alpha=0.5, label="outlier")
    plt.hist(sigmoid(agg_df[col]), bins=50, alpha=0.5, label="outlier")
    plt.legend()
    use_cnt = (sigmoid(agg_df[col]) > 0.5).sum()
    N = len(agg_df)
    plt.title(f"{machine}, N={N}, col={col}, cnt={use_cnt}, rate={use_cnt/N*100:.1f}")
    plt.tight_layout()
# %%
# machine = machines[1]
for machine in machines:
    checkpoint_dir = f"exp/{machine}/audioset_v000_0.15/checkpoint-100epochs"
    col = "pred_machine"
    # agg_df = pd.read_csv(
    #     os.path.join(checkpoint_dir, "checkpoint-100epochs_outlier_mean.csv"),
    # )
    agg_df = pd.concat(
        [
            pd.read_csv(
                os.path.join(checkpoint_dir, "checkpoint-100epochs_outlier_mean.csv"),
                usecols=["path", col],
            ),
            pd.read_csv(
                os.path.join(checkpoint_dir, "checkpoint-100epochs_audioset_mean.csv"),
                usecols=["path", col],
            ),
        ]
    )

    # sig = sigmoid(agg_df[col])
    sig = agg_df[col].values
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
        # 0.95,
        # 0.99,
        # 0.999,
        # 0.9995,
        # 0.9999,
        # 0.99995,
        # 0.99999,
        # 0.999995,
        1,
    ]:
        # use_idx = sig < threshold
        use_idx = sig < np.percentile(sig, threshold * 100)
        use_cnt = use_idx.sum()
        N = len(agg_df)
        title = f"{machine}, N={N}, col={col}, cnt={use_cnt}, rate={use_cnt/N*100:.2f}[%], {threshold}"
        print(title)
# %%
machine = machines[1]
use_train = False
checkpoint_dir = f"exp/{machine}/audioset_v000_0.15/checkpoint-100epochs"
col = "pred_machine"
agg_df = pd.read_csv(
    os.path.join(checkpoint_dir, "checkpoint-100epochs_outlier_mean.csv"),
)
n_plot = 100
h_agg_df = agg_df[sigmoid(agg_df[col]) > 0.999].iloc[:n_plot]
h_agg_df["label"] = "high"
m_agg_df = agg_df[(sigmoid(agg_df[col]) > 0.4) & (sigmoid(agg_df[col]) < 0.6)].iloc[
    :n_plot
]
m_agg_df["label"] = "mid"
l_agg_df = agg_df[sigmoid(agg_df[col]) < 0.1].iloc[:n_plot]
l_agg_df["label"] = "low"

# 埋め込みベクトルの可視化
# 1. IDごとに色を変えて正常〇と異常✕を各50サンプルをプロット（4色）
# 2. 異なるマシンタイプも色を変える（1色）
# 3. 外れ値も色を変える（閾値ごとに色を変える 0, 0.5, 0.999）

other_machines = machines.copy()
other_machines.remove(machine)
if machine in ["fan", "pump", "slider", "valve"]:
    dev_section = [0, 2, 4, 6]
    dev_section = [0, 1, 2, 3, 4, 5, 6]
elif machine == "ToyCar":
    dev_section = [0, 1, 2, 3]
    dev_section = [1, 2, 3, 4, 5, 6, 7]
elif machine == "ToyConveyor":
    dev_section = [0, 1, 2]
    dev_section = [1, 2, 3, 4, 5, 6]

eval_df = pd.read_csv(
    os.path.join(checkpoint_dir, "checkpoint-100epochs_eval_mean.csv")
)
eval_df["fid"] = eval_df["path"].map(lambda x: int(x.split("_")[-1].split(".")[0]))
eval_df = eval_df[eval_df["fid"] < n_plot]

dcase_train_df = pd.read_csv(
    os.path.join(checkpoint_dir, "checkpoint-100epochs_dcase_train_mean.csv")
)
dcase_train_df["fid"] = dcase_train_df["path"].map(
    lambda x: int(x.split("_")[-1].split(".")[0])
)
dcase_train_df["section"] = dcase_train_df["path"].map(lambda x: int(x.split("_")[-2]))
dcase_train_df["machine"] = dcase_train_df["path"].map(lambda x: x.split("/")[2])
dcase_train_df = dcase_train_df[dcase_train_df["fid"] < n_plot]
dcase_train_df["label"] = None
dcase_valid_df = pd.read_csv(
    os.path.join(checkpoint_dir, "checkpoint-100epochs_dcase_valid_mean.csv")
)
dcase_valid_df["fid"] = dcase_valid_df["path"].map(
    lambda x: int(x.split("_")[-1].split(".")[0])
)
dcase_valid_df["section"] = dcase_valid_df["path"].map(lambda x: int(x.split("_")[-2]))
dcase_valid_df["machine"] = dcase_valid_df["path"].map(lambda x: x.split("/")[2])
dcase_valid_df = dcase_valid_df[dcase_valid_df["fid"] < n_plot]

for sec in dev_section:
    print(sec)
    eval_df.loc[
        (eval_df["section"] == sec) & (eval_df["is_normal"] == 1),
        "label",
    ] = f"normal_{sec}"
    eval_df.loc[
        (eval_df["section"] == sec) & (eval_df["is_normal"] == 0),
        "label",
    ] = f"anomaly_{sec}"
    if use_train:
        idx = (dcase_train_df["section"] == sec) & (
            dcase_train_df["machine"] == machine
        )
        dcase_train_df.loc[
            idx,
            "label",
        ] = f"train_normal_{sec}"
    for om in other_machines:
        dcase_valid_df.loc[
            (dcase_valid_df["section"] == sec)
            & (dcase_valid_df["machine"] == om)
            & (dcase_valid_df["fid"] < n_plot // 15),
            "label",
        ] = "pseudo-anomaly"
        if use_train:
            dcase_train_df.loc[
                (dcase_train_df["section"] == sec)
                & (dcase_train_df["machine"] == om)
                & (dcase_train_df["fid"] < n_plot // 15),
                "label",
            ] = "train_pseudo-anomaly"
embed_cols = ["label"] + [f"e{i}" for i in range(128)]
# %%
algorithm = "umap"
use_df = pd.concat(
    [
        eval_df[embed_cols],
        dcase_train_df[embed_cols],
        dcase_valid_df[embed_cols],
        h_agg_df[embed_cols],
        m_agg_df[embed_cols],
        l_agg_df[embed_cols],
    ]
)
use_df = use_df[~use_df["label"].isna()]
embed = use_df[embed_cols[1:]].values
if algorithm == "tsne":
    tsne = TSNE(n_components=2, random_state=2022, perplexity=10, n_iter=1000)
    X_embedded = tsne.fit_transform(embed)
elif algorithm == "umap":
    mapper = umap.UMAP(densmap=True, n_neighbors=5, random_state=2022)
    X_embedded = mapper.fit_transform(embed)

# %%
label_list = list(use_df["label"].unique())
plt.figure(figsize=(20, 20))
for label in label_list:
    idx = use_df["label"] == label
    if "normal" in label:
        marker = "o"
    elif "anomaly_" in label:
        marker = "x"
    elif "train_normal" in label:
        marker = "o"
    else:
        marker = ","
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, marker=marker)
plt.legend()

# %%
# 実行スクリプトの生成
col_names = ["section", "outlier"]
# thresholds = [
#     0,
#     0.05,
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
#     0.95,
#     0.99,
#     0.999,
#     0.9995,
#     0.9999,
#     1,
# ]
col_names = ["machine", "machine2"]
# col_names = ["machine2"]
# thresholds = [0]
seeds = [0, 1, 2, 3, 4]
no = "029"
# seeds = [0, 1, 2, 3]
for col_name in col_names:
    for seed in seeds:
        if col_name == "machine":
            thresholds = [
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
            thresholds = [
                0,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.3,
                0.5,
                0.7,
                1,
            ]
        else:
            thresholds = [
                0,
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
                1,
            ]
        for threshold in thresholds:
            for machine in machines:
                if not os.path.isfile(
                    f"exp/{machine}/audioset_v{no}_{col_name}{threshold}_0.15_seed{seed}/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv"
                ):
                    run_stage = 3
                    if os.path.isfile(
                        f"exp/{machine}/audioset_v{no}_{col_name}{threshold}_0.15_seed{seed}/checkpoint-100epochs/checkpoint-100epochs.pkl"
                    ):
                        run_stage = 4
                    job = f"./sub_use_outliers_job.sh --start_stage 3 --run_stage {run_stage} --stop_stage 3 --threshold {threshold} --col_name {col_name} --seed {seed} --machine {machine} --no audioset_v{no}"
                    print(job)

# %%
seeds = [0, 1, 2, 3, 4]

n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
no = "226"
# nos = [127]
nos = [126, 127, 128, 129, 130, 226, 227, 228, 229, 230]
for no in nos:
    for seed in seeds:
        for n_anomaly in n_anomaly_list:
            for machine in machines:
                if not os.path.isfile(
                    f"exp/{machine}/audioset_v{no}_0.15_anomaly{n_anomaly}_max6_seed{seed}/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv"
                ):
                    run_stage = 3
                    if os.path.isfile(
                        f"exp/{machine}/audioset_v{no}_0.15_anomaly{n_anomaly}_max6_seed{seed}/checkpoint-100epochs/checkpoint-100epochs.pkl"
                    ):
                        run_stage = 4
                    job = f"./sub_job.sh --no audioset_v{no} --stage 1 --start_stage {run_stage} --n_anomaly {n_anomaly} --seed {seed} --machine {machine}"
                    print(job)

# %%
seeds = [0, 1, 2, 3, 4]

nos = ["026", "027", "028", "029", "030"]
for no in nos:
    for seed in seeds:
        for n_anomaly in n_anomaly_list:
            for machine in machines:
                if not os.path.isfile(
                    f"exp/{machine}/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv"
                ):
                    run_stage = 3
                    if os.path.isfile(
                        f"exp/{machine}/audioset_v{no}_0.15_seed{seed}/checkpoint-100epochs/checkpoint-100epochs.pkl"
                    ):
                        run_stage = 4
                    job = f"./sub_job.sh --no audioset_v{no} --stage 1 --start_stage {run_stage} --seed {seed} --machine {machine}"
                    print(job)
# %%
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
agg_checkpoints = [
    "exp/fan/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
    "exp/pump/audioset_v000_0.15/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
    "exp/slider/audioset_v000_0.1/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
    "exp/ToyCar/audioset_v000_0.1/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
    "exp/ToyConveyor/audioset_v000_0.1/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
    "exp/valve/audioset_v000_0.1/checkpoint-100epochs/checkpoint-100epochs_embed_agg.csv",
]
agg_df = pd.read_csv(agg_checkpoints[0])
post_processes = list(agg_df.columns)
for rm in ["path", "is_normal", "section", "mode"]:
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
    "/".join(["exp", "all"] + os.path.dirname(agg_checkpoints[0]).split("/")[2:]),
    f"score_embed.csv",
)
score_df.loc[:, "path"] = save_path
for agg_path in agg_checkpoints:
    agg_df = pd.read_csv(agg_path)
    machine = agg_path.split("/")[1]
    if machine == "ToyCar":
        dev_sections = [0, 1, 2, 3]
        eval_sections = [4, 5, 6]
    elif machine == "ToyConveyor":
        dev_sections = [0, 1, 2]
        eval_sections = [3, 4, 5]
    else:
        dev_sections = [0, 2, 4, 6]
        eval_sections = [1, 3, 5]
    sections = {"dev": dev_sections, "eval": eval_sections}
    for post_process in post_processes:
        for mode in modes:
            auc_list = []
            pauc_list = []
            mauc = 1.1
            for section in sections[mode]:
                target_idx = agg_df["section"] == section
                auc = roc_auc_score(
                    1 - agg_df.loc[target_idx, "is_normal"],
                    -agg_df.loc[target_idx, post_process],
                )
                auc_list.append(auc)
                score_df.loc[post_process, f"{machine}_{section}_auc"] = auc
                if mauc > auc:
                    mauc = auc
                pauc = roc_auc_score(
                    1 - agg_df.loc[target_idx, "is_normal"],
                    -agg_df.loc[target_idx, post_process],
                    max_fpr=0.1,
                )
                pauc_list.append(pauc)
                score_df.loc[post_process, f"{machine}_{section}_pauc"] = pauc
            score_df.loc[post_process, f"{mode}_{machine}_mauc"] = mauc
            score_df.loc[post_process, f"{mode}_{machine}_auc"] = mean(auc_list)
            score_df.loc[post_process, f"{mode}_{machine}_pauc"] = mean(pauc_list)
            score_list = auc_list + pauc_list
            score_df.loc[post_process, f"{mode}_{machine}_hauc"] = hmean(score_list)
            score_df.loc[post_process, f"{mode}_{machine}_hauc_std"] = np.array(
                score_list
            ).std()
