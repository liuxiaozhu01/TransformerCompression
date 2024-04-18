# Get all parameters passed to the script
GPUS="$@"

# Check if GPU parameters are passed
if [ -z "$GPUS" ]; then
  echo "Usage: $0 <gpu1> [gpu2 gpu3 ...]"
  exit 1
fi

# Splices GPU parameters into a string, separated by commas','
GPU_STRING=$(IFS=,; echo "$GPUS")

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPU_STRING" python experiments/run_slicegpt.py \
--model meta-llama/Llama-2-7b-hf \
--model-path /root/home/workspace/LLM/llama/meta-llama/Llama-2-7b-hf \
--save-dir exp/llama_v2_7b \
--sparsity 0.20 \
--device cuda:0 \
--cal-dataset wikitext2 \
--cal-nsamples 128 \
--cal-batch-size 8 \
--cal-max-seqlen 2048 \
--eval-dataset c4 \
--ppl-eval-seqlen 2048 \
--ppl-eval-batch-size 4 \
--no-wandb