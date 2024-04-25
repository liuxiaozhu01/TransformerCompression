#!/bin/bash

# Get all parameters passed to the script
GPUS="$@"

# Check if GPU parameters are passed
if [ -z "$GPUS" ]; then
  echo "Usage: $0 <gpu1> [gpu2 gpu3 ...]"
  exit 1
fi

# Splices GPU parameters into a string, separated by commas','
GPU_STRING=$(IFS=,; echo "$GPUS")

# Define an array with the rates you want to use
rates=("0.30" "0.35" "0.40" "0.45" "0.50")
# rates=("0.45")

# make directory
base_dir="exp/summary/baichuan2_7b_debug"

# Loop over the rates
for rate in "${rates[@]}"
do
  # Create a subdirectory for each rate
  exp_dir="$base_dir/$rate"
  mkdir -p $exp_dir
  
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPU_STRING" python experiments/run_slicegpt.py \
  --model baichuan-inc/Baichuan2-7B-Base \
  --model-path /root/home/workspace/LLM/baichuan/baichuan-inc/Baichuan2-7B-Base \
  --save-dir $exp_dir \
  --sparsity $rate \
  --device cuda:0 \
  --cal-dataset c4 \
  --cal-nsamples 128 \
  --cal-batch-size 8 \
  --cal-max-seqlen 2048 \
  --eval-dataset wikitext2 \
  --ppl-eval-seqlen 128 \
  --ppl-eval-batch-size 8 \
  --no-wandb
done