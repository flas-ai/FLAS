"""FLAS model: time-conditioned velocity field v_theta(h, t, c).

- ConceptEncoder: a few frozen base-model layers that embed the concept text c.
- FlowFunction: B FlowBlocks + a sinusoidal time embedder, predicting the
  velocity v(h, t, c) used by the N-step Euler integrator.
- FlowBlock: TimeEmbed -> CrossAttn(Q=h, KV=concept) -> Causal SelfAttn -> MLP,
  each wrapped with a residual connection and a learnable per-channel gate.

Multi-family support
--------------------
The FlowBlock *structure* is fixed across model families; only the underlying
nn modules (RMSNorm / RotaryEmbedding / Attention) and a few config-derived
behaviors vary. These are resolved once per model by ``get_arch_spec(config)``
keyed on ``config.model_type`` (gemma2, qwen3, ...). To add a family, add one
registry entry — no changes to the block/encoder logic.

``FlowCrossAttention`` is deliberately family-agnostic: it has no module-level
import of any family's classes. The RMSNorm class and a RoPE instance are
injected by the caller (cross-attention is not a component the base model has,
so it should not depend on a specific family's modeling module). The RoPE math
(``rotate_half``) and GQA expansion (``repeat_kv``) are identical across
families and defined locally.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Family-independent RoPE / GQA helpers (identical across gemma2/qwen3/llama)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input (standard RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    """Expand KV heads to match Q heads for GQA."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _apply_rope_single(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (cheaper than apply_rotary_pos_emb(x, x, ...))."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class _RopeLayerTypeWrapper(nn.Module):
    """Adapt a layer-type-keyed RoPE (Gemma3RotaryEmbedding) to the standard
    rope(x, position_ids) -> (cos, sin) call used by FlowCrossAttention."""

    def __init__(self, rope, layer_type):
        super().__init__()
        self.rope = rope
        self.layer_type = layer_type

    def forward(self, x, position_ids):
        return self.rope(x, position_ids, layer_type=self.layer_type)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

@dataclass
class ArchSpec:
    """Per-family modules + config-derived behaviors for the FlowFunction.

    Fields:
        rms_norm_cls:  family RMSNorm class, ``cls(dim, eps=...)``.
        rotary_emb_cls: family RotaryEmbedding class, ``cls(config=config)``.
        attention_cls: family causal-attention class for the self-attn phase.
        embed_scale:   whether token embeddings are scaled by sqrt(hidden)
                       (Gemma family) — only applied when the embedding layer
                       does not already scale internally.
        needs_layer_types: whether a fresh-from-dict config must have
                       ``layer_types`` repopulated (Qwen3/Llama).
        self_attn_layer_idx_fn: maps the target layer index to the layer_idx
                       passed to the self-attn module (controls sliding/global
                       selection; Gemma2 alternates by parity, others use 0).
        norm_init_map: maps FlowBlock norm attribute -> source decoder-layer
                       norm attribute (or None to leave at default weight=1)
                       when initializing from the base model's layer.
    """
    rms_norm_cls: Type[nn.Module]
    rotary_emb_cls: Type[nn.Module]
    attention_cls: Type[nn.Module]
    embed_scale: bool
    needs_layer_types: bool
    self_attn_layer_idx_fn: Callable[[int], int]
    norm_init_map: Dict[str, Optional[str]]
    # Gemma3's RotaryEmbedding holds per-layer-type inv_freqs (global vs local
    # sliding) and requires a `layer_type` arg. When set, the rope is called as
    # rope(x, pos, layer_type=...). None = standard single rope(x, pos).
    rope_layer_type: Optional[str] = None


def _gemma2_spec():
    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2RMSNorm, Gemma2RotaryEmbedding, Gemma2Attention)
    return ArchSpec(
        rms_norm_cls=Gemma2RMSNorm,
        rotary_emb_cls=Gemma2RotaryEmbedding,
        attention_cls=Gemma2Attention,
        embed_scale=True,
        needs_layer_types=False,
        # Gemma2 alternates sliding (even) / global (odd) attention per layer;
        # replicate the target layer's behavior by its parity.
        self_attn_layer_idx_fn=lambda layer_idx: layer_idx % 2,
        norm_init_map={
            "pre_sa_norm": "input_layernorm",
            "post_sa_norm": "post_attention_layernorm",
            "pre_mlp_norm": "pre_feedforward_layernorm",
            "post_mlp_norm": "post_feedforward_layernorm",
        },
    )


def _qwen3_spec():
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3Attention)
    return ArchSpec(
        rms_norm_cls=Qwen3RMSNorm,
        rotary_emb_cls=Qwen3RotaryEmbedding,
        attention_cls=Qwen3Attention,
        embed_scale=False,
        needs_layer_types=True,
        # Qwen3 uses full_attention everywhere (no sliding alternation).
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        # Qwen3 has only 2 norms per layer: input_layernorm (pre-attn) and
        # post_attention_layernorm (really pre-MLP). No post norms — the
        # FlowBlock keeps post_sa/post_mlp norms at default (weight=1), absorbed
        # by the learnable per-feature gates.
        norm_init_map={
            "pre_sa_norm": "input_layernorm",
            "post_sa_norm": None,
            "pre_mlp_norm": "post_attention_layernorm",
            "post_mlp_norm": None,
        },
    )


def _llama_spec():
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention)
    return ArchSpec(
        rms_norm_cls=LlamaRMSNorm,
        rotary_emb_cls=LlamaRotaryEmbedding,
        attention_cls=LlamaAttention,
        embed_scale=False,
        needs_layer_types=True,
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        norm_init_map={
            "pre_sa_norm": "input_layernorm",
            "post_sa_norm": None,
            "pre_mlp_norm": "post_attention_layernorm",
            "post_mlp_norm": None,
        },
    )


def _gemma3_spec():
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3RMSNorm, Gemma3RotaryEmbedding, Gemma3Attention)
    return ArchSpec(
        rms_norm_cls=Gemma3RMSNorm,
        rotary_emb_cls=Gemma3RotaryEmbedding,
        attention_cls=Gemma3Attention,
        embed_scale=True,  # Gemma family scales embeddings (handled in-layer if scaled embedding)
        needs_layer_types=True,  # gemma3 alternates sliding/full; force full for the (unused) self-attn build
        self_attn_layer_idx_fn=lambda layer_idx: 0,
        # Gemma3 keeps gemma2's 4-norm layout (pre/post around attn and MLP).
        norm_init_map={
            "pre_sa_norm": "input_layernorm",
            "post_sa_norm": "post_attention_layernorm",
            "pre_mlp_norm": "pre_feedforward_layernorm",
            "post_mlp_norm": "post_feedforward_layernorm",
        },
        # Cross-attn uses the global rope (FLAS cross-attn is a learned module,
        # not tied to a specific decoder layer's sliding/global type).
        rope_layer_type="full_attention",
    )


_ARCH_REGISTRY: Dict[str, Callable[[], ArchSpec]] = {
    "gemma2": _gemma2_spec,
    "qwen3": _qwen3_spec,
    "llama": _llama_spec,
    "gemma3": _gemma3_spec,
    "gemma3_text": _gemma3_spec,
}


def get_arch_spec(config) -> ArchSpec:
    mt = getattr(config, "model_type", None)
    if mt not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unsupported model_type {mt!r}. Supported: "
            f"{sorted(_ARCH_REGISTRY)}. Add a registry entry in model.py.")
    return _ARCH_REGISTRY[mt]()


def get_text_config(config):
    """Unwrap the text sub-config for multimodal models (gemma3-4b nests the
    decoder config under `.text_config`); pass-through for text-only models."""
    return getattr(config, "text_config", None) or config


def get_text_decoder(model):
    """Return the text decoder module exposing .layers/.embed_tokens/.norm/
    .rotary_emb. Multimodal wrappers (gemma3-4b Gemma3ForConditionalGeneration)
    nest it under model.model.language_model; text-only models expose it at
    model.model."""
    base = model.model
    if hasattr(base, "language_model"):
        return base.language_model
    return base


# ---------------------------------------------------------------------------
# Cross-attention (family-agnostic; norm class + RoPE injected by caller)
# ---------------------------------------------------------------------------

class FlowCrossAttention(nn.Module):
    """GQA cross-attention: activation queries (Q=h) attend to encoded concept
    (KV). RoPE applied to both Q and K sides. QK-norm applied per head_dim.

    Family-agnostic: ``rms_norm_cls`` and ``rotary_emb`` are injected so this
    module does not import any family's modeling classes. Optional logit
    softcapping (Gemma family) via ``softcap``.
    """

    def __init__(self, *, hidden_size, num_heads, num_kv_heads, head_dim,
                 rms_norm_eps, rotary_emb, rms_norm_cls,
                 attn_bias=False, softcap=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.softcap = softcap

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=attn_bias)

        self.q_norm = rms_norm_cls(self.head_dim, eps=rms_norm_eps)
        self.k_norm = rms_norm_cls(self.head_dim, eps=rms_norm_eps)
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, encoder_hidden_states,
                encoder_attention_mask=None, q_pos_offset=0,
                q_position_ids=None, kv_position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)

        # Norm-then-reshape: q_norm/k_norm normalize the last dim (head_dim).
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, Q, D]

        k = self.k_proj(encoder_hidden_states).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [B, Hkv, K, D]

        v = self.v_proj(encoder_hidden_states).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        # RoPE — prefer explicit per-sample position_ids (correct for left-padded
        # batches); fall back to arange + offset for training/same-length batches.
        if q_position_ids is not None:
            q_pos = q_position_ids
        else:
            q_pos = torch.arange(q_len, device=q.device).unsqueeze(0) + q_pos_offset
        if kv_position_ids is not None:
            kv_pos = kv_position_ids
        else:
            kv_pos = torch.arange(kv_len, device=k.device).unsqueeze(0)
        q_cos, q_sin = self.rotary_emb(q, q_pos)
        k_cos, k_sin = self.rotary_emb(k, kv_pos)
        q = _apply_rope_single(q, q_cos, q_sin)
        k = _apply_rope_single(k, k_cos, k_sin)

        # GQA
        k_exp = repeat_kv(k, self.num_kv_groups)
        v_exp = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * self.scaling

        if self.softcap is not None:
            attn_weights = attn_weights / self.softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.softcap

        if encoder_attention_mask is not None:
            mask = encoder_attention_mask[:, None, None, :]
            # Direct fill with -1e4: dominates valid logits without -inf arithmetic.
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        emb = sinusoidal_time_embedding(t, self.freq_dim)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# FlowBlock: cross-attention + causal self-attention + MLP
# ---------------------------------------------------------------------------

class FlowBlock(nn.Module):
    """One FlowBlock: TimeEmbed -> CrossAttn -> Causal SelfAttn -> MLP.

    Each component is wrapped with a residual connection and a learnable
    per-channel gate (init `init_gate`). KV cache for self-attention is
    extended over decoding steps; the cross-attention KV (concept side) is
    cached once outside the block and reused across all integration steps.
    """

    def __init__(self, config, spec: ArchSpec, rotary_emb, init_gate=0.1,
                 layer_idx=0, disable_cross_attn=False,
                 disable_self_attn=False, disable_mlp=False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        rms_norm_eps = config.rms_norm_eps
        RMSNorm = spec.rms_norm_cls
        self.layer_idx = layer_idx
        self.disable_cross_attn = disable_cross_attn
        self.disable_self_attn = disable_self_attn
        self.disable_mlp = disable_mlp

        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        attn_bias = getattr(config, "attention_bias", False)
        softcap = getattr(config, "attn_logit_softcapping", None)

        # Cross-attention: activation queries the encoded concept.
        self.pre_cross_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = FlowCrossAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rotary_emb=rotary_emb,
            rms_norm_cls=RMSNorm,
            attn_bias=attn_bias,
            softcap=softcap)
        self.post_cross_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        # Causal self-attention (family Attention) on the activation stream.
        # Only built when enabled — the no-self-attn baseline skips it entirely
        # (leaner checkpoints, and families whose self-attn we don't clone, e.g.
        # gemma3 with its dual-rope sliding attention, never need to build it).
        if not disable_self_attn:
            sa_config = type(config).from_dict(config.to_dict())
            sa_config._attn_implementation = "eager"
            if spec.needs_layer_types:
                # Fresh-from-dict configs may drop layer_types; repopulate so the
                # attention module resolves a valid (full) attention type.
                sa_config.layer_types = ["full_attention"] * sa_config.num_hidden_layers
            self.pre_sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            sa_layer_idx = spec.self_attn_layer_idx_fn(layer_idx)
            self.self_attn = spec.attention_cls(sa_config, layer_idx=sa_layer_idx)
            self.self_attn_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        # Gated MLP (gate + up + down proj), family activation.
        self.pre_mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Qwen3/Llama use `hidden_act`; the Gemma family uses `hidden_activation`.
        act_name = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None)
        self.act_fn = ACT2FN[act_name]
        self.mlp_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

    def forward(self, h, concept_hidden, concept_mask=None,
                t_emb=None, self_attn_cache=None,
                position_embeddings=None,
                padding_mask=None, use_cache=False, q_pos_offset=0,
                activation_position_ids=None):
        # Time conditioning
        if t_emb is not None:
            h = h + t_emb[:, None, :]

        # Cross-attention: activation queries the encoded concept.
        if not self.disable_cross_attn:
            h_normed = self.pre_cross_norm(h)
            ca_delta = self.cross_attn(
                h_normed, concept_hidden, concept_mask,
                q_pos_offset=q_pos_offset,
                q_position_ids=activation_position_ids)
            ca_delta = self.post_cross_norm(ca_delta)
            h = h + self.cross_gate * ca_delta

        # Causal self-attention.
        if self.disable_self_attn:
            new_cache = self_attn_cache if use_cache else None
        else:
            h_normed_sa = self.pre_sa_norm(h)
            q_len = h.size(1)
            past_len = self_attn_cache.get_seq_length() if self_attn_cache is not None else 0
            kv_len = past_len + q_len

            MASK_VAL = -1e4  # safe for float32/float16, no overflow when summed
            # Vectorized causal mask: row i can attend to columns 0..past_len+i
            row_idx = torch.arange(q_len, device=h.device).unsqueeze(1)      # [q, 1]
            col_idx = torch.arange(kv_len, device=h.device).unsqueeze(0)     # [1, kv]
            causal_mask = torch.where(
                col_idx <= row_idx + past_len,
                torch.zeros(1, device=h.device, dtype=h.dtype),
                torch.full((1,), MASK_VAL, device=h.device, dtype=h.dtype),
            )  # [q_len, kv_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

            # Combine with padding mask (prevents attending to left-padded positions)
            if padding_mask is not None:
                pm = padding_mask[:, :kv_len]  # [batch, kv_len]
                pad_4d = (1.0 - pm[:, None, None, :].to(h.dtype)) * MASK_VAL
                causal_mask = causal_mask + pad_4d

            sa_out = self.self_attn(
                h_normed_sa,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=self_attn_cache,
                cache_position=None,
            )
            sa_delta = sa_out[0]
            # Cache is updated in-place by past_key_values.update()
            new_cache = self_attn_cache if use_cache else None
            sa_delta = self.post_sa_norm(sa_delta)
            h = h + self.self_attn_gate * sa_delta

        # Gated MLP.
        if not self.disable_mlp:
            x = self.pre_mlp_norm(h)
            x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            x = self.post_mlp_norm(x)
            h = h + self.mlp_gate * x

        return h, new_cache


# ---------------------------------------------------------------------------
# FlowFunction: velocity field v(h, t, c)
# ---------------------------------------------------------------------------

class FlowFunction(nn.Module):
    """Velocity field v_theta(h, t, concept) = blocks(h, t, c) - h_input."""

    def __init__(self, config, num_blocks=3, time_conditioned=True, layer_idx=0,
                 disable_cross_attn=False,
                 disable_self_attn=False, disable_mlp=False):
        super().__init__()
        self.config = config
        self.spec = get_arch_spec(config)
        self.hidden_size = config.hidden_size
        self.num_blocks = num_blocks
        self.time_conditioned = time_conditioned

        if time_conditioned:
            self.time_embed = TimeEmbedder(config.hidden_size)

        # One shared RoPE instance: feeds the self-attn phase (via
        # position_embeddings) and is injected into each block's cross-attn.
        base_rope = self.spec.rotary_emb_cls(config=config)
        if self.spec.rope_layer_type is not None:
            self.rotary_emb = _RopeLayerTypeWrapper(base_rope, self.spec.rope_layer_type)
        else:
            self.rotary_emb = base_rope

        self.blocks = nn.ModuleList(
            [FlowBlock(config, self.spec, self.rotary_emb, layer_idx=layer_idx,
                       disable_cross_attn=disable_cross_attn,
                       disable_self_attn=disable_self_attn, disable_mlp=disable_mlp)
             for _ in range(num_blocks)])

    def forward(self, h, concept_hidden, concept_mask=None, t=None,
                self_attn_caches=None,
                use_cache=False,
                padding_mask=None,
                past_len=0,
                position_ids=None):
        h_input = h
        t_emb = self.time_embed(t) if self.time_conditioned and t is not None else None

        seq_len = h.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=h.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self_attn_caches is not None:
                sc = self_attn_caches[i]
            elif use_cache:
                sc = DynamicCache()
            else:
                sc = None
            h, kv_cache = block(
                h, concept_hidden, concept_mask,
                t_emb=t_emb, self_attn_cache=sc,
                position_embeddings=position_embeddings,
                padding_mask=padding_mask, use_cache=use_cache,
                q_pos_offset=past_len,
                activation_position_ids=position_ids)
            if use_cache:
                new_caches.append(kv_cache)

        velocity = h - h_input
        return velocity, new_caches


# ---------------------------------------------------------------------------
# Concept encoder (frozen N-layer base model)
# ---------------------------------------------------------------------------

class ConceptEncoder(nn.Module):
    """Frozen N-layer base-model encoder for the steering prompt text."""

    def __init__(self, model_id="google/gemma-2-2b-it", num_layers=2):
        super().__init__()
        config = get_text_config(AutoConfig.from_pretrained(model_id))
        self.spec = get_arch_spec(config)
        # Force eager attention for compatibility with our additive 4D mask.
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager")
        base = get_text_decoder(full_model)

        self.embed_tokens = base.embed_tokens
        self.layers = nn.ModuleList([
            base.layers[i] for i in range(num_layers)
        ])
        self.norm = self.spec.rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.load_state_dict(base.norm.state_dict())
        self.hidden_size = config.hidden_size
        self.rotary_emb = base.rotary_emb
        # Per-layer attention type (gemma3 sliding/full) for dual-rope dispatch.
        self._layer_types = list(getattr(config, "layer_types", None) or [])[:num_layers]

        del full_model
        torch.cuda.empty_cache()

        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_base_model(cls, base_model, num_layers=2):
        """Build a ConceptEncoder that shares modules with an already-loaded
        base LLM (avoids loading a second copy of the embedding table and the
        first `num_layers` decoder layers).

        Inherits the base model's dtype and device. Use for inference; for
        training, prefer the regular constructor since training-time gradient
        flow into shared weights would mutate the frozen base LLM.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        cfg = get_text_config(base_model.config)
        self.spec = get_arch_spec(cfg)
        base = get_text_decoder(base_model)
        self.embed_tokens = base.embed_tokens
        self.layers = nn.ModuleList(list(base.layers[:num_layers]))
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.hidden_size = cfg.hidden_size
        self._layer_types = list(getattr(cfg, "layer_types", None) or [])[:num_layers]
        for p in self.parameters():
            p.requires_grad = False
        return self

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        # Gemma scales embeddings by sqrt(hidden_size); Qwen3/Llama do not.
        # In transformers >= 5.0 Gemma2's embed_tokens is a
        # Gemma2TextScaledWordEmbedding that already applies the scaling; only
        # scale manually when the embedding layer does not.
        if self.spec.embed_scale and not hasattr(self.embed_tokens, "embed_scale"):
            h = h * (self.hidden_size ** 0.5)

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)

        # Causal + padding mask. For gemma3 sliding layers, sliding_window
        # (512) >> concept length (<=64), so the sliding mask reduces to the
        # plain causal mask — one mask works for both layer types here.
        MIN_VAL = torch.finfo(h.dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), MIN_VAL, device=h.device, dtype=h.dtype),
            diagonal=1).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * MIN_VAL
            mask_4d = (causal + pad_mask).clamp(min=MIN_VAL)
        else:
            mask_4d = causal.expand(bsz, -1, -1, -1)

        # Gemma3's RoPE is keyed by layer type (global vs local sliding); feed
        # each frozen layer the position embeddings for its own type. Other
        # families use a single rope for all layers.
        if self.spec.rope_layer_type is not None and self._layer_types:
            pe_by_type = {lt: self.rotary_emb(h, position_ids, layer_type=lt)
                          for lt in set(self._layer_types)}
            for layer, lt in zip(self.layers, self._layer_types):
                out = layer(h, attention_mask=mask_4d, position_ids=position_ids,
                            position_embeddings=pe_by_type[lt])
                h = out[0] if isinstance(out, tuple) else out
        else:
            position_embeddings = self.rotary_emb(h, position_ids)
            for layer in self.layers:
                # Gemma2DecoderLayer returns a tuple; Qwen3/Llama return a Tensor.
                out = layer(h, attention_mask=mask_4d, position_ids=position_ids,
                            position_embeddings=position_embeddings)
                h = out[0] if isinstance(out, tuple) else out

        return self.norm(h)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _xavier_init_linear(module):
    """Xavier init for nn.Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_flow_model(model_id="google/gemma-2-2b-it", layer=20, num_blocks=2,
                     time_conditioned=True, init_from_gemma=True,
                     disable_cross_attn=False,
                     disable_self_attn=False, disable_mlp=False):
    """Build FlowFunction + ConceptEncoder.

    ``init_from_gemma`` is kept as the kwarg name for backward compat with the
    train.py CLI (``--no-gemma-mlp-init``); it now means "init the FlowBlocks
    (MLP + self-attn + pre/post norms) from the chosen LLM's layer-N weights"
    regardless of architecture.
    """
    config = get_text_config(AutoConfig.from_pretrained(model_id))
    spec = get_arch_spec(config)
    if not disable_self_attn and str(getattr(config, "model_type", "")).startswith("gemma3"):
        raise ValueError(
            "gemma3 self-attention is not supported in FlowBlock (its dual-rope "
            "sliding attention is not cloned). Run gemma3 with --disable-self-attn.")
    flow_fn = FlowFunction(config, num_blocks=num_blocks,
                           time_conditioned=time_conditioned,
                           layer_idx=layer,
                           disable_cross_attn=disable_cross_attn,
                           disable_self_attn=disable_self_attn,
                           disable_mlp=disable_mlp)

    if init_from_gemma:
        print(f"Initializing FlowFunction ({num_blocks} blocks) from {model_id} layer {layer}...")
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager")
        src_layer = get_text_decoder(full_model).layers[layer]
        for block in flow_fn.blocks:
            # MLP from the base layer.
            block.gate_proj.load_state_dict(src_layer.mlp.gate_proj.state_dict())
            block.up_proj.load_state_dict(src_layer.mlp.up_proj.state_dict())
            block.down_proj.load_state_dict(src_layer.mlp.down_proj.state_dict())
            # Self-attention (exact module match) — only when built.
            if not disable_self_attn:
                block.self_attn.load_state_dict(src_layer.self_attn.state_dict())
            # Norms per the family's norm_init_map (None -> default; skip any
            # norm the block doesn't have, e.g. sa-norms when self-attn is off).
            for dst_attr, src_attr in spec.norm_init_map.items():
                if src_attr is None or not hasattr(block, dst_attr):
                    continue
                getattr(block, dst_attr).load_state_dict(
                    getattr(src_layer, src_attr).state_dict())
        del full_model
        torch.cuda.empty_cache()
    else:
        print(f"FlowFunction ({num_blocks} blocks) Xavier initialized")
        for block in flow_fn.blocks:
            # Xavier init MLP
            _xavier_init_linear(block.gate_proj)
            _xavier_init_linear(block.up_proj)
            _xavier_init_linear(block.down_proj)
            # Xavier init self-attn projections (only when built)
            if not disable_self_attn:
                for name, module in block.self_attn.named_modules():
                    _xavier_init_linear(module)

    n_params = sum(p.numel() for p in flow_fn.parameters())
    n_trainable = sum(p.numel() for p in flow_fn.parameters() if p.requires_grad)
    print(f"FlowFunction: {n_params / 1e6:.1f}M params ({n_trainable / 1e6:.1f}M trainable)")

    print(f"Building ConceptEncoder (2-layer {config.model_type}, frozen)...")
    concept_enc = ConceptEncoder(model_id, num_layers=2)

    return flow_fn, concept_enc
