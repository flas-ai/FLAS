#!/usr/bin/env bash
# Reproduce qwen3-4b on AxBench Concept16k.
# Two runs (launch on separate GPUs):
#   GPU 0: TARGET     — self-attn ablated (the deliverable)
#   GPU 1: CORRECTNESS — self-attn ON (reproduces the flowsteer-qwen relic, val~1.23)
# Recipe matches flowsteer_qwen/checkpoints/qwen3_4b_minimal/config.json (fp32).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=thirdparty/axbench/axbench/concept16k/prod_2b_l20_v1/generate
COMMON="--data-dir $DATA --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --layer 20 --num-blocks 1 \
  --batch-size 16 --grad-accum 2 --lr 5e-5 --enc-lr 1e-5 --div-weight 0.1 \
  --n-steps 3 --T-min 0.5 --T-max 2.0 --val-n-concepts 500 \
  --precision 32 --total-steps 30000 --warmup-steps 1000 --val-every 500 \
  --num-workers 4 --output-dir checkpoints"

run_target() {
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m flas.train $COMMON \
    --disable-self-attn --run-name qwen3_4b_c16k_noselfattn
}
run_correctness() {
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m flas.train $COMMON \
    --run-name qwen3_4b_c16k_selfattn
}

"$@"
