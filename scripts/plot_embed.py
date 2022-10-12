# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.core.display import display
from asd_tools.utils import sigmoid
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors

machines = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

# %%
for machine in machines:
    other_machines = machines.copy()
    other_machines.remove(machine)
    if machine in ["fan", "pump", "slider", "valve"]:
        dev_section = [0, 2, 4, 6]
        dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyCar":
        dev_section = [0, 1, 2, 3]
        dev_section = [0, 1, 2, 3, 4, 5, 6]
    elif machine == "ToyConveyor":
        dev_section = [0, 1, 2]
        dev_section = [0, 1, 2, 3, 4, 5]
    # for machine in machines:
    no = "v029"
    # for simple ASD model
    # col_name = ""
    checkpoint_dir = f"exp/{machine}/audioset_{no}_0.15_seed0/checkpoint-100epochs"
    # for using anomaly data model
    # no = "v200"
    # checkpoint_dir = (
    #     f"exp/{machine}/audioset_{no}_0.15_anomaly32_max6_seed0/checkpoint-100epochs"
    # )
    # for using outlier data model
    only_dcase = True
    use_train = False
    col_name = "machine2"
    col = "pred_machine"
    if col_name == "machine2":
        threshold_list = [
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
        ]
    elif col_name == "machine":
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
        ]
    elif col_name in ["outlier", "section"]:
        threshold_list = [
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            # 0.95,
            # 0.99,
            # 0.995,
            # 0.999,
            # 0.9995,
            0.9999,
        ]
        if col_name == "section":
            col = "pred_section_max"
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
            0.3,
            0.5,
            0.7,
            1,
        ]
        if col_name == "section2":
            col = "pred_section_max"
    for threshold in threshold_list:
        checkpoint_dir = f"exp/{machine}/audioset_{no}_{col_name}{threshold}_0.15_seed0/checkpoint-100epochs"
        agg_df = pd.read_csv(
            os.path.join(checkpoint_dir, "checkpoint-100epochs_outlier_mean.csv"),
        )
        agg_df["pred_section_max"] = agg_df[
            [f"pred_section{i}" for i in dev_section]
        ].max(1)
        n_plot = 100
        h_agg_df = (
            agg_df[sigmoid(agg_df[col]) > 0.9]
            .sort_values(by=col, ascending=False)
            .iloc[:n_plot]
        )
        h_agg_df["label"] = "high"
        m_agg_df = agg_df[
            (sigmoid(agg_df[col]) > 0.4) & (sigmoid(agg_df[col]) < 0.6)
        ].iloc[:n_plot]
        m_agg_df["label"] = "mid"
        l_agg_df = agg_df[sigmoid(agg_df[col]) < 0.1].iloc[:n_plot]
        l_agg_df["label"] = "low"

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
                    & (dcase_valid_df["fid"] < n_plot // 3),
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
        zeros = ["zero"] + [0 for _ in range(128)]
        zero_df = pd.DataFrame([zeros], columns=embed_cols)

        algorithm = "umap"
        use_df = pd.concat(
            [
                eval_df[embed_cols],
                dcase_train_df[embed_cols],
                dcase_valid_df[embed_cols],
                # h_agg_df[embed_cols],
                # m_agg_df[embed_cols],
                # l_agg_df[embed_cols],
                zero_df,
            ]
        )
        use_df = use_df[~use_df["label"].isna()]
        embed = use_df[embed_cols[1:]].values
        # for n_neighbors in [10, 15, 20, 25]:

        n_neighbors = 10
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
        plt.figure(figsize=(16, 12))
        for label in label_list:
            idx = (use_df["label"] == label) & (d < np.percentile(d, 99.5))
            id_ = label.split("_")[-1]
            if id_ in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                rgb = cm.colors[int(id_)]
                id_ = f"ID {id_}"
            elif id_ == "pseudo-anomaly":
                rgb = cm.colors[8]
            elif id_ == "high":
                rgb = cm.colors[10]
                print("hight", idx.sum())
            elif id_ == "mid":
                rgb = cm.colors[11]
            elif id_ == "low":
                rgb = cm.colors[12]
            elif id_ == "zero":
                rgb = cm.colors[16]
            s = 30
            if "normal" in label:
                marker = "o"
            elif "anomaly_" in label:
                marker = "x"
                id_ = None
            elif "train_normal" in label:
                marker = "o"
            elif "zero" == label:
                marker = "D"
                s = 200
            else:
                marker = ","
            plt.scatter(
                X_embedded[idx, 0],
                X_embedded[idx, 1],
                label=id_,
                marker=marker,
                color=rgb,
                alpha=0.8,
                s=s,
            )
        plt.legend(fontsize=20)
        plt.title(f"{col} {machine} {algorithm}{n_neighbors}")
        plt.tight_layout()
        plt.savefig(
            f"exp/fig/{no}_dcase_{col_name}_{col}{threshold}_{machine}_{algorithm}{n_neighbors}.png"
        )
# %%
seed = 0
for no in ["v030"]:  # ,
    for machine in machines:
        other_machines = machines.copy()
        other_machines.remove(machine)
        if machine in ["fan", "pump", "slider", "valve"]:
            dev_section = [0, 2, 4, 6]
            # dev_section = [0, 1, 2, 3, 4, 5, 6]
        elif machine == "ToyCar":
            dev_section = [0, 1, 2, 3]
            # dev_section = [0, 1, 2, 3, 4, 5, 6]
        elif machine == "ToyConveyor":
            dev_section = [0, 1, 2]
            # dev_section = [0, 1, 2, 3, 4, 5]
        # for machine in machines:
        use_train = False
        # no = "v020"
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
        h_agg_df = (
            agg_df[sigmoid(agg_df[col]) > 0.9]
            .sort_values(by=col, ascending=False)
            .iloc[:n_plot]
        )
        h_agg_df["label"] = "high"
        m_agg_df = agg_df[
            (sigmoid(agg_df[col]) > 0.4) & (sigmoid(agg_df[col]) < 0.6)
        ].iloc[:n_plot]
        m_agg_df["label"] = "mid"
        l_agg_df = agg_df[sigmoid(agg_df[col]) < 0.1].iloc[:n_plot]
        l_agg_df["label"] = "low"

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
                    & (dcase_valid_df["fid"] < n_plot // 3),
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
        zeros = ["zero"] + [0 for _ in range(128)]
        zero_df = pd.DataFrame([zeros], columns=embed_cols)

        algorithm = "umap"
        use_df = pd.concat(
            [
                eval_df[embed_cols],
                dcase_train_df[embed_cols],
                dcase_valid_df[embed_cols],
                # h_agg_df[embed_cols],
                # m_agg_df[embed_cols],
                # l_agg_df[embed_cols],
                zero_df,
            ]
        )
        use_df = use_df[~use_df["label"].isna()]
        embed = use_df[embed_cols[1:]].values
        # for n_neighbors in [10, 15, 20, 25]:

        n_neighbors = 10
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
        label_list = label_list[-2:] + label_list[:-2]
        cmap_names = ["tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r"]
        cm = plt.cm.get_cmap(cmap_names[0])
        plt.figure(figsize=(16, 12))
        for label in label_list:
            idx = (use_df["label"] == label) & (d < np.percentile(d, 99.5))
            id_ = label.split("_")[-1]
            if id_ in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                rgb = cm.colors[int(id_)]
                id_ = f"ID {id_}"
            elif id_ == "pseudo-anomaly":
                rgb = cm.colors[8]
            elif id_ == "high":
                rgb = cm.colors[10]
                print("hight", idx.sum())
            elif id_ == "mid":
                rgb = cm.colors[11]
            elif id_ == "low":
                rgb = cm.colors[12]
            elif id_ == "zero":
                rgb = cm.colors[16]
            s = 30
            if "normal" in label:
                marker = "o"
                id_ = f"ID {label.split('_')[1]} normal"
            elif "anomaly_" in label:
                marker = "x"
                id_ = f"ID {label.split('_')[1]} anomaly"
            elif "train_normal" in label:
                marker = "o"
            elif "zero" == label:
                idx = use_df["label"] == label
                marker = "D"
                s = 200
            else:
                marker = ","
            plt.scatter(
                X_embedded[idx, 0],
                X_embedded[idx, 1],
                label=id_,
                marker=marker,
                color=rgb,
                alpha=0.8,
                s=s,
            )
        plt.legend(fontsize=20)
        plt.title(f"no:{no} seed:{seed} {col} {machine} {algorithm}{n_neighbors}")
        plt.tight_layout()
        plt.savefig(
            f"exp/fig/{no}_dev_dcase_{col}_{machine}_{algorithm}{n_neighbors}_seed{seed}.png"
        )

# %%
