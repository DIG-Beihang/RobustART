#!/bin/bash
set -euo pipefail

ROOT="/data/RobustART/exprs/exp"
TASKS=(\
  imagenet_c_loop_mini \
  imagenet_s_loop \
  imagenet-a_o-loop \
  imagenet-p-loop-mini \
)

for task in "${TASKS[@]}"; do
  echo "[task] ${task}"
  (cd "${ROOT}/${task}" && bash eval.sh)
done
