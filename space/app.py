"""FLAS interactive demo — multi-model (ZeroGPU).

Hosts all 8 released FLAS flow-steering checkpoints behind a model selector.
Base models (no chat template) are driven with their saved Alpaca prompt format
automatically (read back from each checkpoint's config.json by the loader).

ZeroGPU pattern
---------------
The model is loaded **inside** the `@spaces.GPU` function (where real CUDA is
available) and cached in a module-level dict; only ONE model is kept resident
(others are evicted) so the slice is never oversubscribed, even for the 8B/9B
bases. First use of each model pays a one-time download+load; afterwards it is
reused from the warm cache.

Runs locally too: `python app.py` (the @spaces.GPU decorator no-ops off-Spaces).
"""

import gc
import gradio as gr
import torch
from huggingface_hub import hf_hub_download

# ZeroGPU decorator on Spaces; transparent no-op locally.
try:
    import spaces

    def GPU(duration=60):
        return spaces.GPU(duration=duration)
except ImportError:  # local / CI
    def GPU(duration=60):
        def _deco(fn):
            return fn
        return _deco


# ---------------------------------------------------------------- model registry
# label -> (hf repo, safetensors filename, base model, kind)
MODELS = {
    "Gemma-2-2B-IT":          ("flas-ai/flas-gemma-2-2b-it",         "flas-gemma-2-2b-it.safetensors",         "google/gemma-2-2b-it",             "instruct"),
    "Gemma-2-9B-IT":          ("flas-ai/flas-gemma-2-9b-it",         "flas-gemma-2-9b-it.safetensors",         "google/gemma-2-9b-it",             "instruct"),
    "Gemma-3-4B-IT":          ("flas-ai/flas-gemma-3-4b-it",         "flas-gemma-3-4b-it.safetensors",         "google/gemma-3-4b-it",             "instruct"),
    "Gemma-3-4B-PT (base)":   ("flas-ai/flas-gemma-3-4b-pt",         "flas-gemma-3-4b-pt.safetensors",         "google/gemma-3-4b-pt",             "base"),
    "Qwen3-8B":               ("flas-ai/flas-qwen3-8b",              "flas-qwen3-8b.safetensors",              "Qwen/Qwen3-8B",                    "instruct"),
    "Qwen3-8B-Base (base)":   ("flas-ai/flas-qwen3-8b-base",         "flas-qwen3-8b-base.safetensors",         "Qwen/Qwen3-8B-Base",               "base"),
    "Llama-3.1-8B-Instruct":  ("flas-ai/flas-llama-3.1-8b-instruct", "flas-llama-3.1-8b-instruct.safetensors", "meta-llama/Llama-3.1-8B-Instruct", "instruct"),
    "Llama-3.1-8B (base)":    ("flas-ai/flas-llama-3.1-8b",          "flas-llama-3.1-8b.safetensors",          "meta-llama/Llama-3.1-8B",          "base"),
}
DEFAULT_MODEL = "Gemma-2-2B-IT"


def _is_qwen3_instruct(label):
    """Thinking-mode applies only to the Qwen3 *instruct* model (the base uses Alpaca)."""
    info = MODELS.get(label)
    return bool(info and info[3] == "instruct" and "Qwen3" in info[2])


_loaded = {}      # label -> generator (only the active one is kept)


def _ensure(model_name):
    """Load (download if needed) + cache the generator, evicting any other model.
    Called INSIDE the @spaces.GPU function so CUDA placement runs on a real GPU."""
    if model_name in _loaded:
        return _loaded[model_name]
    for k in list(_loaded):          # keep only one model resident
        del _loaded[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from flas.generate import load_generator
    repo, fname, _base, _kind = MODELS[model_name]
    ckpt = hf_hub_download(repo, fname)
    hf_hub_download(repo, "config.json")     # cached next to the ckpt; loader reads it
    _loaded[model_name] = load_generator(ckpt)
    return _loaded[model_name]


def _duration(model_name, concept, prompt, flowtime, n_steps, max_tokens, temperature, think=False):
    big = ("8B" in model_name) or ("9B" in model_name)
    # first load of a model happens inside the GPU fn, so budget generously
    return min(180, (120 if big else 80) + int(max_tokens) // 32 * 6)


@GPU(duration=_duration)
def steer(model_name, concept, prompt, flowtime, n_steps, max_tokens, temperature, think=False):
    if not prompt.strip():
        return "(prompt is empty)", "(prompt is empty)"
    gen = _ensure(model_name)
    # Only pass enable_thinking when explicitly on, so the default path still works on
    # a flas build that predates the parameter (Qwen3-only; ignored elsewhere).
    et_kw = {"enable_thinking": True} if think else {}
    baseline = gen.generate_batch(
        [prompt], concept or " ",
        flowtimes=[0.0], n_steps=int(n_steps),
        max_tokens=int(max_tokens), temperature=float(temperature), max_batch=1,
        **et_kw,
    )[0]["generation"]
    if not concept.strip():
        return "(set a concept to see the steered output)", baseline
    steered = gen.generate_batch(
        [prompt], concept,
        flowtimes=[float(flowtime)], n_steps=int(n_steps),
        max_tokens=int(max_tokens), temperature=float(temperature), max_batch=1,
        **et_kw,
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

HERO = """
<div id="flas-hero">
  <h1>🧭 FLAS · Flow-based Activation Steering</h1>
  <p>Steer an open LLM toward <em>any concept you can describe in words</em> — “talk like a pirate”,
  “respond as a noir detective”, “use mathematical notation.” One concept-conditioned velocity field,
  no fine-tuning and no per-concept training. Pick a model, set the strength&nbsp;<b>T</b>, and watch
  the steered output diverge from the baseline.</p>
  <div class="links">
    <a href="https://flas-ai.github.io" target="_blank">🌐 Project</a>
    <a href="https://arxiv.org/abs/2605.05892" target="_blank">📄 Paper</a>
    <a href="https://github.com/flas-ai/FLAS" target="_blank">💻 Code</a>
    <a href="https://huggingface.co/flas-ai" target="_blank">🤗 Checkpoints</a>
  </div>
</div>
"""

# ---- Morandi · flat-card · white-ground theme ----------------------------------
SAGE, SAGE_D, CLAY = "#8C9A8E", "#7C8A7E", "#C4A39A"
INK, MUTED, LINE, PAGE, CARD = "#33312E", "#867F74", "#E8E5DF", "#F6F5F2", "#FFFFFF"

# Custom dusty-sage primary ramp → no bright/neon green anywhere (focus rings, etc.)
SAGE_RAMP = gr.themes.colors.Color(
    name="sage",
    c50="#F4F6F3", c100="#E8ECE5", c200="#D2DACD", c300="#B7C3B2",
    c400="#9FAE99", c500="#8C9A8E", c600="#7C8A7E", c700="#67756A",
    c800="#525E55", c900="#424B45", c950="#2A302C",
)

theme = gr.themes.Soft(
    primary_hue=SAGE_RAMP, neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill=PAGE, body_background_fill_dark=PAGE,
    block_background_fill=CARD, block_background_fill_dark=CARD,
    block_border_color=LINE, block_border_width="1px",
    block_label_text_color=MUTED, body_text_color=INK,
    button_primary_background_fill=SAGE, button_primary_background_fill_hover=SAGE_D,
    button_primary_text_color="#FFFFFF", button_primary_border_color=SAGE,
)

CSS = f"""
:root {{ color-scheme: light; }}
.gradio-container {{ max-width: 1460px !important; margin: 0 auto !important;
  padding: 14px 32px 36px !important; background: {PAGE}; }}
/* let the controls row breathe so slider labels + number boxes stay on one line */
.flas-card .gr-slider, .flas-card .wrap label {{ min-width: 0; }}
footer {{ display: none !important; }}

/* hero */
#flas-hero {{ background:{CARD}; border:1px solid {LINE}; border-radius:18px;
  padding:30px 34px; margin-bottom:18px; }}
#flas-hero h1 {{ margin:0 0 10px; font-size:1.72rem; font-weight:700; color:{INK}; letter-spacing:-.02em; }}
#flas-hero p {{ margin:0; color:{MUTED}; font-size:1rem; line-height:1.65; max-width:820px; }}
#flas-hero .links {{ margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }}
#flas-hero .links a {{ text-decoration:none; font-size:.85rem; color:{SAGE_D};
  background:#EFF1EC; border:1px solid #DEE4DB; padding:6px 14px; border-radius:999px; transition:.15s; }}
#flas-hero .links a:hover {{ background:#E5EAE2; }}

/* flat cards — border only, no shadow */
.flas-card {{ background:{CARD} !important; border:1px solid {LINE} !important;
  border-radius:16px !important; padding:22px !important; box-shadow:none !important; }}
.flas-card + .flas-card {{ margin-top:16px; }}
.gradio-container label span {{ color:{MUTED} !important; font-weight:500 !important; }}

/* primary button — sage, full, no glow */
.flas-go button {{ border-radius:12px !important; font-weight:600 !important;
  box-shadow:none !important; min-height:46px; }}
.flas-go button:hover, .flas-go button:focus {{ box-shadow:none !important; }}

/* kill the green focus glow → soft sage ring */
.gradio-container textarea:focus, .gradio-container input:focus,
.gradio-container textarea:focus-visible, .gradio-container input:focus-visible,
.gradio-container .gr-box:focus-within, .gradio-container .wrap:focus-within {{
  box-shadow: 0 0 0 3px rgba(140,154,142,.16) !important;
  border-color: {SAGE} !important; outline: none !important; }}

.flas-steered textarea {{ background:#FBF6F3 !important; border-color:#EAD9D0 !important; }}
.flas-status {{ color:{MUTED}; font-size:.85rem; min-height:1.1em; padding-left:2px; }}
input[type=range] {{ accent-color:{SAGE}; }}

/* tighten Examples spacing (no big empty gap under the table) */
.flas-card .examples {{ margin-top:6px !important; }}
.flas-card table {{ margin:0 !important; }}
"""

with gr.Blocks(title="FLAS — Flow-based Activation Steering") as demo:
    gr.HTML(HERO)

    with gr.Group(elem_classes="flas-card"):
        model = gr.Dropdown(
            choices=list(MODELS), value=DEFAULT_MODEL, label="Model",
            info="8 FLAS checkpoints · base variants use an Alpaca prompt automatically",
        )
        think_toggle = gr.Checkbox(
            value=False, visible=_is_qwen3_instruct(DEFAULT_MODEL),
            label="Qwen3 thinking mode (experimental)",
            info="Let Qwen3 emit its <think> reasoning while being steered. FLAS was trained "
                 "on the non-thinking template, so this is off-distribution — exploratory.",
        )
        with gr.Row():
            concept = gr.Textbox(label="Steering concept",
                                 placeholder="e.g. talk like a pirate",
                                 value="Talk like a pirate", lines=2, scale=1)
            prompt = gr.Textbox(label="Your prompt",
                                value="Tell me about your day.", lines=2, scale=1)
        with gr.Row():
            flowtime = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Flow time T",
                                 info="steering strength")
            n_steps = gr.Slider(1, 10, value=3, step=1, label="Euler steps N")
            max_tokens = gr.Slider(32, 256, value=128, step=32, label="Max tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
        run_btn = gr.Button("Generate", variant="primary", elem_classes="flas-go")
        status = gr.Markdown("", elem_classes="flas-status")

    with gr.Row():
        steered_out = gr.Textbox(label="Steered  ·  FLAS @ chosen T", lines=11,
                                 elem_classes="flas-steered", scale=1)
        baseline_out = gr.Textbox(label="Baseline  ·  no steering", lines=11, scale=1)

    with gr.Group(elem_classes="flas-card"):
        gr.Examples(EXAMPLES, inputs=[concept, prompt], label="Try one of these",
                    examples_per_page=6)

    # Thinking toggle only makes sense for the Qwen3 instruct model; hide it otherwise.
    model.change(lambda m: gr.update(visible=_is_qwen3_instruct(m)), model, think_toggle)

    inputs = [model, concept, prompt, flowtime, n_steps, max_tokens, temperature, think_toggle]
    run_btn.click(lambda m: f"⏳ running {m} … (first use of a model downloads it)", model, status).then(
        steer, inputs, [steered_out, baseline_out]).then(
        lambda: "", None, status)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
