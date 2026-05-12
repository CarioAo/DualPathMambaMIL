import pandas as pd
import os

# ===== 路径配置 =====
input_csv = "process_list_autogen_LGG.csv"          # 你的原始 csv
output_csv = "preprocess/LGG.csv"    # 输出到项目 dataset_csv 下

FIXED_DIR = "/root/sj-tmp/CLAM_output_LGG_512/pt_files/pt_files/"
FIXED_LABEL = "LGG"

# ===== 读取原始 CSV =====
df = pd.read_csv(input_csv)

# ===== 检查必要列 =====
if "slide_id" not in df.columns:
    raise ValueError("输入 CSV 中不存在 slide_id 列")

# ===== 提取 case_id / slide_id（去掉 .svs）=====
def strip_svs(name):
    if isinstance(name, str) and name.endswith(".svs"):
        return name[:-4]
    return name

df["case_id"] = df["slide_id"].apply(strip_svs)
df["slide_id"] = df["slide_id"].apply(strip_svs)

# ===== 构造目标格式 =====
out_df = pd.DataFrame({
    "dir": FIXED_DIR,
    "case_id": df["case_id"],
    "slide_id": df["slide_id"],
    "label": FIXED_LABEL
})

# ===== 去重（非常重要，防止重复 slide）=====
out_df = out_df.drop_duplicates(subset=["case_id"])

# ===== 确保输出目录存在 =====
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ===== 保存 =====
out_df.to_csv(output_csv, index=False)

print(f"✅ 转换完成，共 {len(out_df)} 个样本")
print(f"📁 输出文件：{output_csv}")
