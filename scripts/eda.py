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
    print("train", machine, df["section"].unique())

    dir_list = os.listdir(f"downloads/dev/{machine}/test")
    df = pd.DataFrame(dir_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    print("dev/test", machine, df["section"].unique())
    dir_list = os.listdir(f"downloads/eval/{machine}/test")
    df = pd.DataFrame(dir_list, columns=["path"])
    df["section"] = df["path"].map(lambda x: int(x.split("_")[2]))
    print("eval/test", machine, df["section"].unique())
# %%
