"""FLAS generation with self-attn KV caching.

KV cache structure: _sa_caches[euler_step][block_idx] = DynamicCache
Each Euler step has its own set of per-block KV caches because the activations
entering self-attention differ across steps (h is modified by previous steps).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from flas.model import FlowFunction, ConceptEncoder


def _load_ckpt(path):
    """Load weights from .safetensors or .pt. Returns a state_dict-like mapping
    that may also contain a top-level 'flow_fn' / 'concept_enc' wrapper."""
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return {"flow_fn": load_file(p, device="cpu")}
    return torch.load(p, weights_only=True, map_location="cpu")


class FlasGenerator:
    """Steered generation with ODE integration."""

    def __init__(self, llm, tokenizer, flow_fn, concept_enc, layer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.flow_fn = flow_fn
        self.concept_enc = concept_enc
        self.layer = layer
        self._flow_dtype = next(flow_fn.parameters()).dtype

        # Hook state
        self._hook_handle = None
        self._active = False
        self._flowtimes = None
        self._n_steps = 2
        self._concept_hidden = None
        self._concept_mask = None
        self._padding_mask = None      # padding mask (grows during generation)
        self._sa_caches = None         # Per-step list of per-block KV caches
        self._is_prefill = True
        self._past_len = 0             # Position offset for RoPE / KV cache
        self._position_ids = None      # [batch, seq_len] per-sample positions (handles left-padding)

    def _hook_fn(self, module, input, output):
        if not self._active:
            return output
        is_tuple = isinstance(output, tuple)
        h_orig = output[0] if is_tuple else output
        h = h_orig.to(self._flow_dtype)
        bsz = h.size(0)
        dt = (self._flowtimes[:bsz] / self._n_steps).to(self._flow_dtype)

        for k in range(self._n_steps):
            t_k = dt * k
            if self._is_prefill:
                v, kv_caches = self.flow_fn(
                    h, self._concept_hidden[:bsz],
                    self._concept_mask[:bsz], t=t_k,
                    padding_mask=self._padding_mask[:bsz],
                    use_cache=True, past_len=0,
                    position_ids=self._position_ids[:bsz])
                self._sa_caches[k] = kv_caches
            else:
                v, kv_caches = self.flow_fn(
                    h, self._concept_hidden[:bsz],
                    self._concept_mask[:bsz], t=t_k,
                    self_attn_caches=self._sa_caches[k],
                    padding_mask=self._padding_mask[:bsz],
                    use_cache=True, past_len=self._past_len,
                    position_ids=self._position_ids[:bsz])
                self._sa_caches[k] = kv_caches
            h = h + dt.unsqueeze(1).unsqueeze(2) * v

        h_out = h.to(h_orig.dtype)
        return (h_out,) + output[1:] if is_tuple else h_out

    def _install_hook(self):
        if self._hook_handle is None:
            self._hook_handle = self.llm.model.layers[self.layer].register_forward_hook(
                self._hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def encode_concept(self, text, max_len=64):
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).to("cuda")
        hidden = self.concept_enc(enc.input_ids, enc.attention_mask)
        return hidden, enc.attention_mask.float()

    @torch.no_grad()
    def generate_batch(self, prompts, concept_text, flowtimes, n_steps=2,
                       max_tokens=128, temperature=1.0, max_batch=16):
        self._n_steps = n_steps

        concept_hidden, concept_mask = self.encode_concept(concept_text)

        pairs = []
        formatted = []
        for pi, prompt in enumerate(prompts):
            msgs = [{"role": "user", "content": prompt}]
            fmt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            for flowtime in flowtimes:
                pairs.append((pi, flowtime))
                formatted.append(fmt)

        total = len(pairs)
        all_results = [None] * total

        for chunk_start in range(0, total, max_batch):
            chunk_end = min(chunk_start + max_batch, total)
            chunk_fmt = formatted[chunk_start:chunk_end]
            chunk_pairs = pairs[chunk_start:chunk_end]
            bsz = len(chunk_fmt)

            enc = self.tokenizer(
                chunk_fmt, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
                add_special_tokens=False,  # template already includes <bos>; matches train.py
            ).to("cuda")
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
            prompt_len = input_ids.shape[1]

            # Set up hook state for prefill
            self._concept_hidden = concept_hidden.expand(bsz, -1, -1).contiguous()
            self._concept_mask = concept_mask.expand(bsz, -1).contiguous()
            self._flowtimes = torch.tensor(
                [d for _, d in chunk_pairs],
                device="cuda", dtype=torch.float32)
            self._padding_mask = attention_mask.float()
            self._sa_caches = [None] * self._n_steps
            self._is_prefill = True
            self._past_len = 0

            # Per-sample position_ids (correct for left-padded batches):
            # pad positions → 0, real positions → 0, 1, 2, ...
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
            self._position_ids = position_ids

            self._install_hook()
            self._active = True

            # Prefill
            out = self.llm(
                input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

            # Switch to generation mode
            self._is_prefill = False
            self._past_len = prompt_len

            generated = input_ids
            unfinished = torch.ones(bsz, dtype=torch.bool, device="cuda")

            for _ in range(max_tokens):
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                next_token = next_token.masked_fill(
                    ~unfinished.unsqueeze(1), self.tokenizer.pad_token_id)
                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, unfinished.unsqueeze(1).long()], dim=1)

                eos_hit = (next_token.squeeze(1) == self.tokenizer.eos_token_id)
                unfinished = unfinished & ~eos_hit
                if not unfinished.any():
                    break

                # Update padding mask and position_ids before hook runs.
                # New token's position per-sample = previous max real position + 1.
                self._padding_mask = attention_mask.float()
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                self._position_ids = position_ids[:, -1:]  # only new token's position

                out = self.llm(
                    next_token, attention_mask=attention_mask,
                    position_ids=self._position_ids,
                    past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]

                # Hook read self._past_len BEFORE this increment (it ran during the
                # llm() call above). Increment now so the NEXT generation step sees
                # the correct past length.
                self._past_len += 1

            self._active = False

            for i in range(bsz):
                pi, flowtime = chunk_pairs[i]
                gen_ids = generated[i, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                idx = chunk_start + i
                all_results[idx] = {
                    "prompt": prompts[pi],
                    "prompt_idx": pi,
                    "flowtime": flowtime,
                    "generation": text,
                }

            del past_kv, out
            self._sa_caches = None
            torch.cuda.empty_cache()

        self._remove_hook()
        return all_results


def load_generator(flow_ckpt, model_id=None, layer=None, num_blocks=None):
    """Load a FlasGenerator from a checkpoint.

    Auto-reads model_id/layer/num_blocks from {ckpt_dir}/config.json if not
    provided, preventing num_blocks / layer mismatch crashes.
    """
    import json
    from pathlib import Path

    # Try to load training config from checkpoint directory
    cfg_path = Path(flow_ckpt).parent / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Loaded training config from {cfg_path}", flush=True)

    # Resolve params: explicit arg > config.json > default
    model_id = model_id or cfg.get("model_id", "google/gemma-2-2b-it")
    layer = layer if layer is not None else cfg.get("layer", 20)
    num_blocks = num_blocks if num_blocks is not None else cfg.get("num_blocks", 2)
    print(f"  model_id={model_id}, layer={layer}, num_blocks={num_blocks}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p in llm.parameters():
        p.requires_grad = False

    config = AutoConfig.from_pretrained(model_id)
    flow_fn = FlowFunction(config, num_blocks=num_blocks, time_conditioned=True,
                           layer_idx=layer,
                           disable_cross_attn=cfg.get("disable_cross_attn", False),
                           disable_self_attn=cfg.get("disable_self_attn", False),
                           disable_mlp=cfg.get("disable_mlp", False))

    ckpt = _load_ckpt(flow_ckpt)
    sd = ckpt["flow_fn"] if "flow_fn" in ckpt else ckpt
    target_dtype = next(iter(sd.values())).dtype
    print(f"  flow weights dtype: {target_dtype}", flush=True)
    flow_fn.to(target_dtype)
    flow_fn.load_state_dict(sd)
    flow_fn = flow_fn.to("cuda").eval()

    concept_enc = ConceptEncoder(model_id, num_layers=2).to(target_dtype).to("cuda")
    if "concept_enc" in ckpt:
        concept_enc.load_state_dict(ckpt["concept_enc"])

    return FlasGenerator(llm, tokenizer, flow_fn, concept_enc, layer)
