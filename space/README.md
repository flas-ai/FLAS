---
title: FLAS Demo
emoji: 🧭
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
hardware: zero-a10g
pinned: false
license: apache-2.0
short_description: Steer 8 open LLMs toward any natural-language concept
---

# FLAS Demo — Flow-based Activation Steering

Interactive demo of **FLAS** across **8 released checkpoints** — Gemma-2 (2B/9B),
Gemma-3 (4B), Qwen3-8B and Llama-3.1-8B, in both instruct and base variants. Pick a
model, type a steering concept (e.g. *"talk like a pirate"*), set the strength `T`, and
the model rewrites itself — steered vs baseline side-by-side. No fine-tuning, no
per-concept training.

- 🌐 Project: <https://flas-ai.github.io>
- 📄 Paper: <https://arxiv.org/abs/2605.05892>
- 💻 Code: <https://github.com/flas-ai/FLAS>
- 🤗 Checkpoints: <https://huggingface.co/flas-ai>

## Maintainer notes

- **ZeroGPU** hardware, size `large` (48 GB). Only one model is kept resident at a time
  (others are evicted) so the slice is never oversubscribed, even for the 8B/9B bases.
  Hub downloads run *outside* the `@spaces.GPU` function so they don't consume GPU quota.
- **Gated models**: set an `HF_TOKEN` Space secret whose owner has accepted the Gemma-2,
  Gemma-3 and Llama-3.1 licenses (Qwen3 is open). Without it those models fail to load.
- Base variants (`… · base`) have no chat template; they use a saved Alpaca prompt format
  read back from each checkpoint's `config.json` automatically.
