# FLAS

**Flow-based Activation Steering for Inference-Time Intervention.**

FLAS learns a concept-conditioned velocity field $v_\theta(h, t, c)$ that transports an unsteered activation $h$ to a steered activation $h'$ by integrating a flow ODE:

$$h' = \varphi_T(h) = h + \int_0^T v_\theta\!\bigl(\varphi_t(h),\, t,\, c\bigr)\, dt$$

The flow time $T$ serves as a continuous steering-strength parameter; sampling $T \sim \mathrm{Uniform}[T_{\min}, T_{\max}]$ during training enables zero-shot strength control at inference. FLAS is the first learned steering method to consistently outperform in-context prompting on AxBench.

<p align="center">
  <img src="figs/main.png" width="90%" />
</p>

> This repository is released for double-blind review. It contains no author-identifying information.

## Install

We recommend using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import flas; print('ok')"
```

`uv` will read `pyproject.toml` and install a matching CUDA-enabled torch wheel automatically. 

If you prefer pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The base LLM (Gemma-2-2B-IT or Gemma-2-9B-IT) is downloaded from Hugging Face. Make sure you have run `hf login` and accepted the Gemma-2 license.

## Data

FLAS trains on AxBench data. Clone the AxBench repo:

```bash
git clone https://github.com/stanfordnlp/axbench thirdparty/axbench
```

Training data are at `thirdparty/axbench/axbench/<concept_set>/prod_<model>_<layer>_v1/generate/`.

For example, Concept500 on Gemma-2-2B-IT layer 20:

```
thirdparty/axbench/axbench/concept500/prod_2b_l20_v1/generate/
├── train_data.parquet
└── metadata.jsonl
```

## Quick start

Train on Concept500 (single 18 GB+ GPU, Gemma-2-2B-IT):

```bash
uv run python -m flas.train \
    --data-dir thirdparty/axbench/axbench/concept500/prod_2b_l20_v1/generate \
    --output-dir checkpoints --run-name flas_2b_c500
```

To train on Concept16k instead, add `--val-n-concepts` for training-time held-out evaluation:

```bash
uv run python -m flas.train \
    --data-dir thirdparty/axbench/axbench/concept16k/prod_2b_l20_v1/generate \
    --val-n-concepts 500 \
    --output-dir checkpoints --run-name flas_2b_c16k
```

Generate steered outputs:

```bash
uv run python scripts/eval.py \
    --flow-ckpt checkpoints/flas_2b_c500/best_step*.pt \
    --output-dir results/flas_2b_c500 \
    --num-eval-prompts 10 --max-tokens 256 \
    --flowtimes 1.0 1.5 2.0 2.5 3.0
```

Score generations with GPT-4o-mini (AxBench-aligned C/I/F judge):

```bash
uv run python scripts/judge_openai.py \
    --results-file results/flas_2b_c500/results_shard0.json \
    --output       results/flas_2b_c500/judged.json \
    --api-key "$OPENAI_API_KEY" --concurrency 8
```

Interactive TUI for testing:

```bash
uv run python scripts/chat.py \
    --flow-ckpt checkpoints/flas_2b_c500/best_step*.pt
```

## Evaluation protocol

Following AxBench (Wu et al., 2025):

1. **Generate.** For each held-out concept × AlpacaEval prompt × flowtime $T$, generate steered text with `scripts/eval.py`.
2. **Judge.** GPT-4o-mini scores each generation on Concept ($C$), Instruction-following ($I$), and Fluency ($F$), each in $\{0, 1, 2\}$.
3. **Aggregate.** Per-factor max with prompt-level 50/50 split: pick the best $T$ per concept on "train" prompts, report HMean = $3 / (1/C + 1/I + 1/F)$ on "test" prompts.

## Project layout

```
flas/
├── src/flas/
│   ├── model.py          # FlowBlock
│   ├── train.py          # PyTorch Lightning training
│   ├── generate.py       # Batched generation
├── scripts/
│   ├── eval.py           # AxBench-aligned generation
│   ├── judge_openai.py   # GPT-4o-mini judge
│   └── chat.py           # interactive TUI
├── pyproject.toml
├── LICENSE
└── README.md
```

## Acknowledgements and data

This codebase uses data and evaluation pipelines from the AxBench. Base models are Gemma-2-2B-IT and Gemma-2-9B-IT. Citations are deferred to the paper to preserve double-blind review.

## License

Released under the Apache 2.0 (see [LICENSE](LICENSE)).
