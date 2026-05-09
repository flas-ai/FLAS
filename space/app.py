"""FLAS interactive demo (Gemma-2-2B-IT).

Pulls the bf16 checkpoint from the Hugging Face Hub on first launch, then
serves a Gradio UI that runs steered vs baseline generation side-by-side.

Runs locally with `python app.py` (any CUDA GPU >= 6 GB) and on Hugging Face
Spaces with the same code (the @gpu decorator becomes a ZeroGPU slice).
"""

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

# ZeroGPU decorator on Spaces; no-op locally so the same code runs unchanged.
try:
    import spaces
    gpu = spaces.GPU(duration=120)
except ImportError:
    def gpu(fn):
        return fn


HF_REPO = "flas-ai/flas-gemma-2-2b-it"
_gen = None
_ckpt_path = None


def _ensure_loaded():
    """Lazy init: download weights and build FlasGenerator on first call."""
    global _gen, _ckpt_path
    if _gen is None:
        from flas.generate import load_generator
        if _ckpt_path is None:
            _ckpt_path = hf_hub_download(HF_REPO, "flas-gemma-2-2b-it.safetensors")
            hf_hub_download(HF_REPO, "config.json")  # cached alongside
        _gen = load_generator(_ckpt_path)
    return _gen


@gpu
def steer(concept, prompt, flowtime, n_steps, max_tokens, temperature):
    gen = _ensure_loaded()
    if not prompt.strip():
        return "(prompt is empty)", "(prompt is empty)"

    # Generate baseline (no steering) and steered side-by-side. We pass T=0
    # for baseline; the velocity field still runs but contributes nothing.
    baseline = gen.generate_batch(
        [prompt], concept or " ",
        flowtimes=[0.0], n_steps=int(n_steps),
        max_tokens=int(max_tokens), temperature=float(temperature),
        max_batch=1,
    )[0]["generation"]

    if not concept.strip():
        return baseline, "(set a concept to see the steered output)"

    steered = gen.generate_batch(
        [prompt], concept,
        flowtimes=[float(flowtime)], n_steps=int(n_steps),
        max_tokens=int(max_tokens), temperature=float(temperature),
        max_batch=1,
    )[0]["generation"]
    return steered, baseline


EXAMPLES = [
    ["Talk like a pirate", "Tell me about your day."],
    ["Respond as a noir detective", "How do I make a good cup of coffee?"],
    ["Always reference places in Minnesota", "Plan me a perfect Sunday."],
    ["Frame everything as a musical performance", "Explain quantum mechanics like I'm new to it."],
    ["French words and phrases related to months and days", "Describe the weather in autumn."],
    ["Speak in programming terms", "What does it feel like to be tired?"],
]

INTRO = """
# FLAS — Flow-based Activation Steering

Steer **Gemma-2-2B-IT** toward any concept you can describe in words. Drop in a
phrase like *"talk like a pirate"* or *"always reference places in Minnesota"*,
adjust the strength, and the model rewrites itself accordingly. No fine-tuning,
no per-concept training.

[📄 Paper](https://arxiv.org/abs/2605.05892) · [💻 Code](https://github.com/flas-ai/FLAS) · [🤝 Model card](https://huggingface.co/flas-ai/flas-gemma-2-2b-it)
"""

with gr.Blocks(title="FLAS — Flow-based Activation Steering") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            concept = gr.Textbox(
                label="Steering concept",
                placeholder="e.g. talk like a pirate",
                value="Talk like a pirate", lines=2,
            )
            prompt = gr.Textbox(
                label="Your prompt",
                value="Tell me about your day.", lines=3,
            )
            with gr.Row():
                flowtime = gr.Slider(0.0, 4.0, value=2.0, step=0.1,
                                     label="Flow time T (steering strength)")
                n_steps = gr.Slider(1, 10, value=3, step=1, label="Euler steps N")
            with gr.Row():
                max_tokens = gr.Slider(32, 256, value=128, step=32, label="Max tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            steered_out = gr.Textbox(
                label="Steered (FLAS @ chosen T)", lines=10,
            )
            baseline_out = gr.Textbox(
                label="Baseline (no steering)", lines=10,
            )

    gr.Examples(EXAMPLES, inputs=[concept, prompt],
                label="Try one of these:")

    run_btn.click(
        steer,
        inputs=[concept, prompt, flowtime, n_steps, max_tokens, temperature],
        outputs=[steered_out, baseline_out],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
