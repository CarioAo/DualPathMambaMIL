#!/bin/bash

############################################
# Dual-Path MambaMIL training script
# Author: you 😄
############################################

model_name="dualpath_mamba_mil"
backbones="resnet50 plip"

declare -A in_dim
in_dim["resnet50"]=1024
in_dim["plip"]=512

# GPU setting
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

# Task & paths
task="GBM_LGG"
results_dir="./experiments/train/${task}"
split_dir="./splits/GBM_LGG_100"

# Training config
lr="2e-4"
patch_size=512
preloading="no"

# Mamba-related params（保持接口一致，方便消融）
mambamil_rate=5
mambamil_layer=2
mambamil_type="Mamba"   # 实际上 DualPath 内部你已固定，这里只是占位

for backbone in $backbones
do
    exp="${model_name}/${backbone}"
    echo "=========================================="
    echo "Running experiment: $exp"
    echo "GPU: $gpu_id"
    echo "=========================================="

    python main.py \
        --drop_out 0 \
        --early_stopping \
        --lr $lr \
        --k 10 \
        --k_start -1 \
        --k_end -1 \
        --label_frac 1.0 \
        --exp_code $exp \
        --patch_size $patch_size \
        --weighted_sample \
        --task $task \
        --backbone $backbone \
        --results_dir $results_dir \
        --model_type $model_name \
        --log_data \
        --split_dir $split_dir \
        --preloading $preloading \
        --in_dim ${in_dim[$backbone]} \
        --mambamil_rate $mambamil_rate \
        --mambamil_layer $mambamil_layer \
        --mambamil_type $mambamil_type
done