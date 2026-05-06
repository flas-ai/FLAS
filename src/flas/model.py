"""FLAS model: time-conditioned velocity field v_theta(h, t, c).

- ConceptEncoder: a few frozen base-model layers that embed the concept text c.
- FlowFunction: B FlowBlocks + a sinusoidal time embedder, predicting the
  velocity v(h, t, c) used by the N-step Euler integrator.
- FlowBlock: TimeEmbed -> CrossAttn(Q=h, KV=concept) -> Causal SelfAttn -> MLP,
  each wrapped with a residual connection and a learnable per-channel gate.
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    Gemma2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)


def _apply_rope_single(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (cheaper than apply_rotary_pos_emb(x, x, ...))."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Cross-attention (Gemma2 GQA with RoPE, QK norm, softcapping)
# ---------------------------------------------------------------------------

class FlowCrossAttention(nn.Module):
    """Cross-attention with Gemma2 GQA config.
    RoPE applied to both Q and K sides.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.softcap = config.attn_logit_softcapping
        hidden_size = config.hidden_size

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def forward(self, hidden_states, encoder_hidden_states,
                encoder_attention_mask=None, q_pos_offset=0,
                q_position_ids=None, kv_position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)

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
            # Direct fill with -1e4 (softcap keeps valid logits in [-30, 30],
            # so -1e4 dominates; avoids -inf arithmetic).
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
# FlowBlock: cross-attention + MLP + causal self-attention
# ---------------------------------------------------------------------------

class FlowBlock(nn.Module):
    """One FlowBlock: TimeEmbed -> CrossAttn -> Causal SelfAttn -> MLP.

    Each component is wrapped with a residual connection and a learnable
    per-channel gate (init `init_gate`). KV cache for self-attention is
    extended over decoding steps; the cross-attention KV (concept side) is
    cached once outside the block and reused across all integration steps.
    """

    def __init__(self, config, init_gate=0.1, layer_idx=0,
                 disable_cross_attn=False,
                 disable_self_attn=False, disable_mlp=False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        rms_norm_eps = config.rms_norm_eps
        self.layer_idx = layer_idx
        self.disable_cross_attn = disable_cross_attn
        self.disable_self_attn = disable_self_attn
        self.disable_mlp = disable_mlp

        # Cross-attention: activation queries the encoded concept.
        self.pre_cross_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = FlowCrossAttention(config)
        self.post_cross_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        # Causal self-attention (Gemma2Attention) on the activation stream.
        sa_config = type(config).from_dict(config.to_dict())
        sa_config._attn_implementation = "eager"
        self.pre_sa_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_sa_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        # Gemma2 alternates sliding (even) / global (odd) attention per layer;
        # layer_idx % 2 replicates the behavior of the target layer.
        self.self_attn = Gemma2Attention(sa_config, layer_idx=layer_idx % 2)
        self.self_attn_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        # Gemma-style gated MLP (gate + up + down proj, GELU-tanh).
        self.pre_mlp_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_mlp_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.act_fn = nn.GELU(approximate='tanh')
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

        # Causal self-attention (Gemma2Attention).
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
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=self_attn_cache,
            )
            sa_delta = sa_out[0]
            # Cache is updated in-place by past_key_values.update()
            new_cache = self_attn_cache if use_cache else None
            sa_delta = self.post_sa_norm(sa_delta)
            h = h + self.self_attn_gate * sa_delta

        # Gemma-style gated MLP.
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
        self.hidden_size = config.hidden_size
        self.num_blocks = num_blocks
        self.time_conditioned = time_conditioned

        if time_conditioned:
            self.time_embed = TimeEmbedder(config.hidden_size)

        self.blocks = nn.ModuleList(
            [FlowBlock(config, layer_idx=layer_idx,
                       disable_cross_attn=disable_cross_attn,
                       disable_self_attn=disable_self_attn, disable_mlp=disable_mlp)
             for _ in range(num_blocks)])
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

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
# Concept encoder (frozen 2-layer Gemma)
# ---------------------------------------------------------------------------

class ConceptEncoder(nn.Module):
    """Frozen 2-layer Gemma encoder for steering prompt text."""

    def __init__(self, model_id="google/gemma-2-2b-it", num_layers=2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        full_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32)

        self.embed_tokens = full_model.model.embed_tokens
        self.layers = nn.ModuleList([
            full_model.model.layers[i] for i in range(num_layers)
        ])
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.load_state_dict(full_model.model.norm.state_dict())
        self.hidden_size = config.hidden_size
        self.rotary_emb = full_model.model.rotary_emb

        del full_model
        torch.cuda.empty_cache()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        h = h * (self.hidden_size ** 0.5)

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        # Causal + padding mask
        MIN_VAL = torch.finfo(h.dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), MIN_VAL, device=h.device, dtype=h.dtype),
            diagonal=1).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * MIN_VAL
            mask_4d = (causal + pad_mask).clamp(min=MIN_VAL)
        else:
            mask_4d = causal.expand(bsz, -1, -1, -1)

        for layer in self.layers:
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
    """Build FlowFunction + ConceptEncoder."""
    config = AutoConfig.from_pretrained(model_id)
    flow_fn = FlowFunction(config, num_blocks=num_blocks,
                           time_conditioned=time_conditioned,
                           layer_idx=layer,
                           disable_cross_attn=disable_cross_attn,
                           disable_self_attn=disable_self_attn,
                           disable_mlp=disable_mlp)

    if init_from_gemma:
        print(f"Initializing FlowFunction ({num_blocks} blocks) from Gemma layer {layer}...")
        full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        src_layer = full_model.model.layers[layer]
        for block in flow_fn.blocks:
            # MLP from Gemma
            block.gate_proj.load_state_dict(src_layer.mlp.gate_proj.state_dict())
            block.up_proj.load_state_dict(src_layer.mlp.up_proj.state_dict())
            block.down_proj.load_state_dict(src_layer.mlp.down_proj.state_dict())
            block.pre_mlp_norm.load_state_dict(src_layer.pre_feedforward_layernorm.state_dict())
            block.post_mlp_norm.load_state_dict(src_layer.post_feedforward_layernorm.state_dict())
            # Self-attention from Gemma (only the attention part, not MLP)
            block.self_attn.load_state_dict(src_layer.self_attn.state_dict())
            block.pre_sa_norm.load_state_dict(src_layer.input_layernorm.state_dict())
            block.post_sa_norm.load_state_dict(src_layer.post_attention_layernorm.state_dict())
        del full_model
        torch.cuda.empty_cache()
    else:
        print(f"FlowFunction ({num_blocks} blocks) Xavier initialized")
        for block in flow_fn.blocks:
            # Xavier init MLP
            _xavier_init_linear(block.gate_proj)
            _xavier_init_linear(block.up_proj)
            _xavier_init_linear(block.down_proj)
            # Xavier init self-attn projections
            for name, module in block.self_attn.named_modules():
                _xavier_init_linear(module)

    n_params = sum(p.numel() for p in flow_fn.parameters())
    n_trainable = sum(p.numel() for p in flow_fn.parameters() if p.requires_grad)
    print(f"FlowFunction: {n_params / 1e6:.1f}M params ({n_trainable / 1e6:.1f}M trainable)")

    print("Building ConceptEncoder (2-layer Gemma, frozen)...")
    concept_enc = ConceptEncoder(model_id, num_layers=2)

    return flow_fn, concept_enc
