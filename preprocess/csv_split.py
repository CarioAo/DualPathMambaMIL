import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ======================
# 配置
# ======================
csv_path = "preprocess/GBM_LGG.csv"
out_dir = "../splits/GBM_LGG_100"
n_splits = 10
seed = 1
val_frac = 0.1

os.makedirs(out_dir, exist_ok=True)

# ======================
# 读取数据
# ======================
df = pd.read_csv(csv_path)
case_ids = df["case_id"].values
labels = df["label"].values
unique_labels = sorted(df["label"].unique())

skf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=seed
)

# ======================
# 生成 splits
# ======================
for fold, (trainval_idx, test_idx) in enumerate(skf.split(case_ids, labels)):
    rng = np.random.RandomState(seed)

    trainval_df = df.iloc[trainval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # ---- 从 trainval 中再分 val（分层）
    val_indices = []
    for label in unique_labels:
        label_df = trainval_df[trainval_df["label"] == label]
        n_val = max(1, int(len(label_df) * val_frac))
        val_indices.extend(
            rng.choice(label_df.index, size=n_val, replace=False)
        )

    val_df = df.loc[val_indices]
    train_df = trainval_df.drop(val_indices)

    # ======================
    # split_k.csv
    # ======================
    max_len = max(len(train_df), len(val_df), len(test_df))

    split_df = pd.DataFrame({
        "train": list(train_df["case_id"]) + [np.nan] * (max_len - len(train_df)),
        "val": list(val_df["case_id"]) + [np.nan] * (max_len - len(val_df)),
        "test": list(test_df["case_id"]) + [np.nan] * (max_len - len(test_df)),
    })

    split_df.to_csv(
        os.path.join(out_dir, f"split_{fold}.csv")
    )

    # ======================
    # split_k_bool.csv
    # ======================
    bool_df = pd.DataFrame(
        index=df["case_id"],
        columns=["train", "val", "test"]
    ).fillna(False)

    bool_df.loc[train_df["case_id"], "train"] = True
    bool_df.loc[val_df["case_id"], "val"] = True
    bool_df.loc[test_df["case_id"], "test"] = True

    bool_df.to_csv(
        os.path.join(out_dir, f"split_{fold}_bool.csv")
    )

    # ======================
    # split_k_descriptor.csv
    # ======================
    desc_df = pd.DataFrame(
        index=unique_labels,
        columns=["train", "val", "test"]
    ).fillna(0)

    for label in unique_labels:
        desc_df.loc[label, "train"] = sum(train_df["label"] == label)
        desc_df.loc[label, "val"] = sum(val_df["label"] == label)
        desc_df.loc[label, "test"] = sum(test_df["label"] == label)

    desc_df.to_csv(
        os.path.join(out_dir, f"split_{fold}_descriptor.csv")
    )

    print(f"✅ Fold {fold} saved")
