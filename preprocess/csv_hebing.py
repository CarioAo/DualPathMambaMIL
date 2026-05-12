import pandas as pd
import os

# ===== 输入文件 =====
gbm_csv = "preprocess/GBM.csv"
lgg_csv = "preprocess/LGG.csv"

# ===== 输出文件 =====
output_csv = "preprocess/GBM_LGG.csv"

# ===== 读取 =====
df_gbm = pd.read_csv(gbm_csv)
df_lgg = pd.read_csv(lgg_csv)

# ===== 基本检查 =====
required_cols = {"dir", "case_id", "slide_id", "label"}
assert required_cols.issubset(df_gbm.columns), "GBM.csv 列名不符合要求"
assert required_cols.issubset(df_lgg.columns), "LGG.csv 列名不符合要求"

# ===== 合并 =====
df_all = pd.concat([df_gbm, df_lgg], ignore_index=True)

# ===== 去重（以 case_id 为准，防止重复）=====
df_all = df_all.drop_duplicates(subset=["case_id"])

# ===== 排序（可选，方便查看）=====
df_all = df_all.sort_values(by=["label", "case_id"]).reset_index(drop=True)

# ===== 创建输出目录（安全）=====
out_dir = os.path.dirname(output_csv)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# ===== 保存 =====
df_all.to_csv(output_csv, index=False)

print("✅ 合并完成")
print(f"GBM 数量: {len(df_gbm)}")
print(f"LGG 数量: {len(df_lgg)}")
print(f"总计数量: {len(df_all)}")
print(f"📁 输出文件: {output_csv}")
