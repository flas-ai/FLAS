#!/usr/bin/env bash
# Launch a FLAS base-model 46k run (bf16, no-self-attn, alpaca prompt format,
# 100k opt-steps, fixed held-out). Base models have no usable chat template, so
# --prompt-format alpaca uses '### Instruction:\n{}\n\n### Response:\n'.
#
# Usage: scripts/run_base_46k.sh <model_id> <run_name> <gpu>
# Models: meta-llama/Llama-3.1-8B | Qwen/Qwen3-8B-Base | google/gemma-3-4b-pt
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; RUN=$2; GPU=$3

CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 \
.venv/bin/python -m flas.train \
  --data-dir data/flas-concept46k --layer 20 --num-blocks 1 --disable-self-attn \
  --batch-size 16 --grad-accum 2 --lr 5e-5 --enc-lr 1e-5 --div-weight 0.1 \
  --n-steps 3 --T-min 0.5 --T-max 2.0 --heldout-ids-file data/flas46k_heldout500.json \
  --n-val-samples 100 --precision bf16-mixed --total-steps 100000 --warmup-steps 2000 \
  --val-every 500 --num-workers 4 --output-dir checkpoints \
  --prompt-format alpaca \
  --model-id "$MODEL" --run-name "$RUN" > "checkpoints/$RUN.log" 2>&1
echo "$RUN exited: $?"
