# %%
import pandas as pd
import numpy as np
import os
import glob

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
