#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8

MASTER_PORT=${MASTER_PORT:-29500}
CONFIGS=(
  # config_clip_fare2.yaml
  # config_clip_tecoa2.yaml
  # config_clip_openai.yaml
  # config_convnext_base_cvst.yaml
  config_convnext_base.yaml
  config_convnextv2_base.yaml
  config_vit_base.yaml
  config_vit_base_cvst.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "[eval] ${cfg}"
  PYTHONPATH=${PYTHONPATH:-}:../../../ GLOG_vmodule=MemcachedClient=-1 \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    -m prototype.prototype.solver.imgnet_a_o_eval_solver --config "${cfg}" --evaluate
done
