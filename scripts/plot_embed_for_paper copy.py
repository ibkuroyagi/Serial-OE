# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.core.display import display
from matplotlib.legend_handler import HandlerTuple
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors

machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
# machines = ["pump"]

# %%
seed = 0
for no in ["v030", "v029", "v028", "v027", "v026", "v008"]:
    for machine in machines:
        if machine != "pump":
            continue
        other_machines = machines.copy()
        other_machines.remove(machine)
        if machine in ["fan", "pump", "slider", "valve"]:
            dev_section = [0, 2, 4, 6]
        elif machine == "ToyCar":
            dev_section = [0, 1, 2, 3]
        elif machine == "ToyConveyor":
            dev_section = [0, 1, 2]
        # for machine in machines:
        use_train = False
        col = "pred_machine"
        if no in [
            "v008",
            "v019",
            "v020",
            "v021",
            "v026",
            "v027",
            "v028",
            "v029",
            "v030",
        ]:
            # for simple ASD model
            col_name = ""
            checkpoint_dir = (
                f"exp/{machine}/audioset_{no}_0.15_seed{seed}/checkpoint-100epochs"
            )
        elif no in ["v119", "v219", "v108", "v208"]:
            # for using anomaly data model
            checkpoint_dir = f"exp/{machine}/audioset_{no}_0.15_anomaly32_max6_seed0/checkpoint-100epochs"
            col = "pred_machine"
            # col = "pred_section_max"
        agg_df = pd.read_csv(
            os.path.join(checkpoint_dir, "checkpoint-100epochs_outlier_mean.csv"),
        )
        agg_df["pred_section_max"] = agg_df[
            [f"pred_section{i}" for i in dev_section]
        ].max(1)
        n_plot = 100

        # 埋め込みベクトルの可視化
        # 1. IDごとに色を変えて正常〇と異常✕を各50サンプルをプロット（4色）
        # 2. 異なるマシンタイプも色を変える（1色）
        # 3. 外れ値も色を変える（閾値ごとに色を変える 0, 0.5, 0.999）

        eval_df = pd.read_csv(
            os.path.join(checkpoint_dir, "checkpoint-100epochs_eval_mean.csv")
        )
        eval_df["fid"] = eval_df["path"].map(
            lambda x: int(x.split("_")[-1].split(".")[0])
        )
        eval_df = eval_df[eval_df["fid"] < n_plot]

        dcase_train_df = pd.read_csv(
            os.path.join(checkpoint_dir, "checkpoint-100epochs_dcase_train_mean.csv")
        )
        dcase_train_df["fid"] = dcase_train_df["path"].map(
            lambda x: int(x.split("_")[-1].split(".")[0])
        )
        dcase_train_df["section"] = dcase_train_df["path"].map(
            lambda x: int(x.split("_")[-2])
        )
        dcase_train_df["machine"] = dcase_train_df["path"].map(
            lambda x: x.split("/")[2]
        )
        dcase_train_df = dcase_train_df[dcase_train_df["fid"] < n_plot]
        dcase_train_df["label"] = None
        dcase_valid_df = pd.read_csv(
            os.path.join(checkpoint_dir, "checkpoint-100epochs_dcase_valid_mean.csv")
        )
        dcase_valid_df["fid"] = dcase_valid_df["path"].map(
            lambda x: int(x.split("_")[-1].split(".")[0])
        )
        dcase_valid_df["section"] = dcase_valid_df["path"].map(
            lambda x: int(x.split("_")[-2])
        )
        dcase_valid_df["machine"] = dcase_valid_df["path"].map(
            lambda x: x.split("/")[2]
        )
        dcase_valid_df = dcase_valid_df[dcase_valid_df["fid"] < n_plot]

        for sec in dev_section:
            eval_df.loc[
                (eval_df["section"] == sec) & (eval_df["is_normal"] == 1),
                "label",
            ] = f"{sec}_normal"
            eval_df.loc[
                (eval_df["section"] == sec) & (eval_df["is_normal"] == 0),
                "label",
            ] = f"{sec}_anomaly"
            if use_train:
                idx = (dcase_train_df["section"] == sec) & (
                    dcase_train_df["machine"] == machine
                )
                dcase_train_df.loc[
                    idx,
                    "label",
                ] = f"{sec}_train_normal"
            for om in other_machines:
                dcase_valid_df.loc[
                    (dcase_valid_df["section"] == sec)
                    & (dcase_valid_df["machine"] == om)
                    & (dcase_valid_df["fid"] < n_plot // 2),
                    "label",
                ] = "pseudo-anomaly"
                if use_train:
                    dcase_train_df.loc[
                        (dcase_train_df["section"] == sec)
                        & (dcase_train_df["machine"] == om)
                        & (dcase_train_df["fid"] < n_plot // 3),
                        "label",
                    ] = "train_pseudo-anomaly"
        embed_cols = ["label"] + [f"e{i}" for i in range(128)]
        zero_df = pd.DataFrame([["zero"] + [0 for _ in range(128)]], columns=embed_cols)

        algorithm = "umap"
        use_df = pd.concat(
            [
                eval_df[embed_cols],
                # dcase_train_df[embed_cols],
                dcase_valid_df[embed_cols],
                zero_df,
            ]
        )
        use_df = use_df[~use_df["label"].isna()]
        embed = use_df[embed_cols[1:]].values
        n_neighbors = 30
        if algorithm == "tsne":
            tsne = TSNE(
                n_components=2, random_state=2022, perplexity=n_neighbors, n_iter=1000
            )
            X_embedded = tsne.fit_transform(embed)
        elif algorithm == "umap":
            mapper = umap.UMAP(densmap=True, n_neighbors=n_neighbors, random_state=2022)
            X_embedded = mapper.fit_transform(embed)

        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X_embedded)
        d = neigh.kneighbors(X_embedded)[0].sum(-1)
        label_list = sorted(list(use_df["label"].unique()))
        cmap_names = ["tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r"]
        cm = plt.cm.get_cmap(cmap_names[0])
        fig, ax = plt.subplots(figsize=(8, 9))
        for label in label_list:
            idx = (use_df["label"] == label) & (d < np.percentile(d, 99.5))
            id_ = label.split("_")[0]
            print(id_)
            if id_ in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                rgb = cm.colors[int(id_)]
                alpha = 0.7
                s = 30
                if "_normal" in label:
                    marker = "o"
                    plt.scatter(
                        [],
                        [],
                        label=f"ID {id_}",
                        marker="s",
                        color=rgb,
                        alpha=1.0,
                        s=s,
                    )
                elif "_anomaly" in label:
                    marker = "x"
                plt.scatter(
                    X_embedded[idx, 0],
                    X_embedded[idx, 1],
                    label=None,
                    marker=marker,
                    color=rgb,
                    alpha=alpha,
                    s=s,
                )
            elif id_ == "pseudo-anomaly":
                rgb = cm.colors[8]
                alpha = 0.7
                plt.scatter(
                    X_embedded[idx, 0],
                    X_embedded[idx, 1],
                    label="p-anomaly",
                    marker="^",
                    color=rgb,
                    alpha=alpha,
                    s=60,
                )
            elif id_ == "zero":
                rgb = cm.colors[16]
                alpha = 1
                idx = use_df["label"] == label
                marker = "D"
                s = 100
                plt.scatter(
                    X_embedded[idx, 0],
                    X_embedded[idx, 1],
                    label=id_,
                    marker=marker,
                    color=rgb,
                    alpha=alpha,
                    s=s,
                )
        plt.scatter(
            [],
            [],
            label="normal",
            marker="o",
            color=(0.5, 0.5, 0.5),
            alpha=1.0,
            s=100,
        )
        plt.scatter(
            [],
            [],
            label="anomaly",
            marker="x",
            color=(0.5, 0.5, 0.5),
            alpha=1.0,
            s=100,
        )
        plt.legend(loc="best", fontsize=19, ncol=2)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"exp/fig/paper_{no}_dev_dcase_{col}_{machine}_{algorithm}{n_neighbors}.png"
        )

# %%
