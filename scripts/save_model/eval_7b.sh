# bug fix: https://github.com/huggingface/evaluate/issues/590
CUDA_VISIBLE_DEVICES=4 python experiments/run_lm_eval.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.30" \
        --save-dir "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.30" \
        --sparsity 0.30 \
        --batch-size 4 \
        --distribute-model \
        --no-wandb

CUDA_VISIBLE_DEVICES=6 python experiments/run_lm_eval.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.40" \
        --save-dir "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.40" \
        --sparsity 0.40 \
        --batch-size 4 \
        --no-wandb

CUDA_VISIBLE_DEVICES=7 python experiments/run_lm_eval.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.50" \
        --save-dir "/root/home/workspace/TransformerCompression/exp/save_model/llama_v2_7b/0.50" \
        --sparsity 0.50 \
        --batch-size 4 \
        --no-wandb