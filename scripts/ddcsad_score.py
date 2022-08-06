# %%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.metrics import roc_auc_score
from statistics import mean
from scipy.stats import hmean


machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
expdir = "/home/i_kuroyanagi/workspace2/laboratory/ParallelWaveGAN/egs/dcase2020-task2/greedy/exp2"

# %%
no = "v080"
no = "v180-anomaly"
if no == "v080":
    n_anomaly_list = [0]
    norm = "bin"
    step = "4000"
elif no == "v180-anomaly":
    n_anomaly_list = [0, 1, 2, 4, 8, 16, 32]
    norm = "bin"
    step = "5000"
seed = 0
n_anomaly = 1
machine = machines[0]
for n_anomaly in n_anomaly_list:
    for seed in [0, 1, 2, 3]:
        for machine in machines:
            columns = [
                f"dev_{machine}_hauc",
                f"dev_{machine}_aveauc",
                f"dev_{machine}_mauc",
                f"dev_{machine}_auc",
                f"dev_{machine}_pauc",
            ]
            if machine in ["fan", "pump", "slider", "valve"]:
                dev_section = ["00", "02", "04", "06"]
            elif machine == "ToyCar":
                dev_section = ["04", "01", "02", "03"]
            elif machine == "ToyConveyor":
                dev_section = ["03", "01", "02"]
            for _id in dev_section:
                columns += [f"{machine}_{int(_id)}_auc", f"{machine}_{int(_id)}_pauc"]
            score_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
            mauc = 1.1
            auc_list = []
            pauc_list = []
            for _id in dev_section:
                result_path = f"{expdir}/{machine}/{norm}/ResNet38Double/{no}_seed{seed}/scratch/anomaly{n_anomaly}/id{_id}/checkpoint-{step}steps/split10frame256_0.5/scores.csv"
                is_normal_path = f"{expdir}/{machine}/{norm}/ResNet38Double/{no}_seed{seed}/scratch/anomaly{n_anomaly}/id{_id}/checkpoint-{step}steps/split10frame256_0.5/is_normal.csv"
                result_df = pd.read_csv(result_path)
                is_normal_df = pd.read_csv(is_normal_path)
                auc = roc_auc_score(1 - is_normal_df["is_normal"], -result_df["mean"])
                score_df.loc[0, f"{machine}_{int(_id)}_auc"] = auc
                pauc = roc_auc_score(
                    1 - is_normal_df["is_normal"], -result_df["mean"], max_fpr=0.1
                )
                score_df.loc[0, f"{machine}_{int(_id)}_pauc"] = pauc
                auc_list.append(auc)
                pauc_list.append(pauc)
                if mauc > auc:
                    mauc = auc
            auc_apuc_list = auc_list + pauc_list
            score_df.loc[0, f"dev_{machine}_hauc"] = hmean(auc_apuc_list)
            score_df.loc[0, f"dev_{machine}_aveauc"] = mean(auc_apuc_list)
            score_df.loc[0, f"dev_{machine}_auc"] = mean(auc_list)
            score_df.loc[0, f"dev_{machine}_pauc"] = mean(pauc_list)
            score_df.loc[0, f"dev_{machine}_mauc"] = mauc
            save_path = (
                result_path
            ) = f"{expdir}/{machine}/{norm}/ResNet38Double/{no}_seed{seed}/scratch/anomaly{n_anomaly}/dev/checkpoint-{step}steps/split10frame256_0.5/agg.csv"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            score_df.to_csv(save_path, index=False)

# %%
no = 100
seed_list = [0, 1, 2, 3]
n_anomaly = 0
ddcsad_score_list = np.zeros((len(machines), len(seed_list)))
for k, seed in enumerate(seed_list):
    for i, machine in enumerate(machines):
        ddcsad_path = f"/home/i_kuroyanagi/workspace2/laboratory/ParallelWaveGAN/egs/dcase2020-task2/greedy/exp2/{machine}/bin/ResNet38Double/v080_seed{seed}/scratch/anomaly{n_anomaly}/dev/checkpoint-4000steps/split10frame256_0.5/agg.csv"
        ddcsad_df = pd.read_csv(ddcsad_path)
        ddcsad_score_list[i, k] = (
            ddcsad_df.loc[
                0, [f"dev_{machine}_auc", f"dev_{machine}_pauc"]
            ].values.mean()
            * 100
        )

# %%
ddcsad_score = ddcsad_score_list.mean(1)
ddcsad_score
# %%
