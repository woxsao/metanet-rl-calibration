#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <base_path>"
    exit 1
fi

BASE_PATH=$1
TOTAL_TIMESTEPS=1000000
SMOOTHNESS=0.97

python -u metanet_sb3.py \
  --base_path $BASE_PATH \
  --total_timesteps $TOTAL_TIMESTEPS \
  --update_interval 1 \
  --bc_smoothness $SMOOTHNESS \
  --save_dir results/batched_smoothness_sweep/smoothness_$SMOOTHNESS \
  --tensorboard_log ./tensorboard/smoothness_$SMOOTHNESS \
  --num_cpus 4 &


wait
echo "All experiments done"