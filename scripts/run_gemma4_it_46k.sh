#!/usr/bin/env bash
# FLAS Gemma-4-12B-it 46k run (instruct, chat prompt format, non-thinking).
#
# gemma-4-12B is natively multimodal (gemma4_unified); FLAS steers the TEXT
# decoder only. Self-attention is NOT cloned into the FlowBlock (gemma4 attn
# uses per-layer-type head_dim / k==v / shared-KV) -> --disable-self-attn is
# mandatory. The chat template ships enable_thinking; build_chat_prompt forces
# the non-thinking (empty thought channel) prompt for direct-answer targets.
#
# Layer 28/48 (~58% depth) matches the relative steering depth of the other
# 46k runs (layer 20 of 32-36 layers). Batch 8 x grad-accum 4 = eff. 32
# (smaller per-step batch than the 8B recipe to fit the 12B on one 80GB A100).
#
# Usage: scripts/run_gemma4_it_46k.sh <gpu>   (default GPU 0)
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}
RUN=flas46k_gemma4_12b_it

CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 \
.venv/bin/python -m flas.train \
  --data-dir data/flas-concept46k --layer 28 --num-blocks 1 --disable-self-attn \
  --batch-size 8 --grad-accum 4 --lr 5e-5 --enc-lr 1e-5 --div-weight 0.1 \
  --n-steps 3 --T-min 0.5 --T-max 2.0 --heldout-ids-file data/flas46k_heldout500.json \
  --n-val-samples 100 --precision bf16-mixed --total-steps 100000 --warmup-steps 2000 \
  --val-every 500 --num-workers 4 --output-dir checkpoints \
  --prompt-format chat \
  --model-id google/gemma-4-12B-it --run-name "$RUN" > "checkpoints/$RUN.log" 2>&1
echo "$RUN exited: $?"
