#!/usr/bin/env python3
"""
Unified Model — combines LLMs-from-scratch (GPT) + reasoning-from-scratch (Qwen3).

Every component is reproduced exactly from the source projects. The architecture
is selected by the config dict: include `"pos_encoding": "learned"` for GPT-style,
or omit it (uses RoPE) for Qwen3-style. All config keys match the original projects.

Usage:
    python unified_model.py generate --prompt "Hello" --arch gpt
    python unified_model.py generate --prompt "Solve: 2+2" --arch qwen3 --tokenizer tokenizer-base.json
    python unified_model.py train --epochs 1 --batch 2
    python unified_model.py chat
    python unified_model.py grpo --steps 10 --tokenizer tokenizer-base.json
    python unified_model.py download --kind base --tokenizer-only
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import textwrap
import time
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# ##############################################################################
# CONFIGURATIONS — exact copies from both projects
# ##############################################################################

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}


def resolve_config(cfg):
    cfg = dict(cfg)
    if cfg.get("pos_encoding") is None:
        if "wpe" in str(cfg) or "qkv_bias" in str(cfg):
            cfg.setdefault("pos_encoding", "learned")
        else:
            cfg.setdefault("pos_encoding", "rope")
    cfg.setdefault("norm_type", "layernorm" if cfg["pos_encoding"] == "learned" else "rmsnorm")
    cfg.setdefault("attn_type", "mha" if cfg["pos_encoding"] == "learned" else "gqa")
    cfg.setdefault("ffn_type", "gelu" if cfg["pos_encoding"] == "learned" else "swiglu")
    cfg.setdefault("drop_rate", 0.0)
    cfg.setdefault("qkv_bias", False)
    cfg.setdefault("head_dim", cfg.get("head_dim") or (cfg["emb_dim"] // cfg["n_heads"]))
    cfg.setdefault("n_kv_groups", cfg.get("n_kv_groups") or cfg["n_heads"])
    cfg.setdefault("hidden_dim", cfg.get("hidden_dim") or 4 * cfg["emb_dim"])
    cfg.setdefault("rope_base", 10_000.0)
    cfg.setdefault("qk_norm", False)
    cfg.setdefault("dtype", torch.float32)
    return cfg


# ##############################################################################
# BUILDING BLOCKS — exact copies from both projects
# ##############################################################################

# --- LayerNorm (LLMs-from-scratch ch04) ---

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# --- RMSNorm (reasoning-from-scratch qwen3.py) ---

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        return norm_x.to(input_dtype)


# --- GELU (LLMs-from-scratch ch04) ---

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# --- RoPE (reasoning-from-scratch qwen3.py) ---

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


# --- MultiHeadAttention (LLMs-from-scratch ch03) ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# --- GroupedQueryAttention (reasoning-from-scratch qwen3.py) ---

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0
            keys, values = keys_new, values_new
        next_cache = (keys, values)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


# --- FeedForward GELU (LLMs-from-scratch ch04) ---

class FeedForwardGELU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# --- FeedForward SwiGLU (reasoning-from-scratch qwen3.py) ---

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# --- TransformerBlock GPT-style (LLMs-from-scratch ch04) ---

class TransformerBlockGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x


# --- TransformerBlock Qwen3-style (reasoning-from-scratch qwen3.py) ---

class TransformerBlockQwen3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"], num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"], dtype=cfg["dtype"])
        self.ff = FeedForwardSwiGLU(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache


# --- KVCache (reasoning-from-scratch qwen3.py) ---

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


# ##############################################################################
# UNIFIED MODEL
# ##############################################################################

class UnifiedModel(nn.Module):
    """Auto-detects architecture from config keys.

    Configs from either project work directly:
        UnifiedModel(GPT_CONFIG_124M)  -> GPT-style (Learned pos, MHA, GELU, LayerNorm)
        UnifiedModel(QWEN_CONFIG_06_B) -> Qwen3-style (RoPE, GQA, SwiGLU, RMSNorm)
    """

    def __init__(self, cfg):
        super().__init__()
        cfg = resolve_config(cfg)
        self.cfg = cfg
        self.pos_encoding = cfg["pos_encoding"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"],
                                    dtype=cfg["dtype"])

        if self.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])
            self.trf_blocks = nn.Sequential(
                *[TransformerBlockGPT(cfg) for _ in range(cfg["n_layers"])])
            self.final_norm = LayerNorm(cfg["emb_dim"])
        else:
            self.trf_blocks = nn.ModuleList(
                [TransformerBlockQwen3(cfg) for _ in range(cfg["n_layers"])])
            self.final_norm = RMSNorm(cfg["emb_dim"])
            cos, sin = compute_rope_params(
                head_dim=cfg["head_dim"],
                theta_base=cfg["rope_base"],
                context_length=cfg["context_length"])
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
            self.current_pos = 0

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"],
                                  bias=(self.pos_encoding == "learned"),
                                  dtype=cfg["dtype"])

    def forward(self, in_idx, cache=None):
        if self.pos_encoding == "learned":
            return self._forward_gpt(in_idx)
        return self._forward_qwen3(in_idx, cache)

    def _forward_gpt(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def _forward_qwen3(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        mask = mask[None, None, :, :]
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


# ##############################################################################
# GENERATION FUNCTIONS — exact copies from both projects
# ##############################################################################

# --- generate_text_simple (LLMs-from-scratch ch04) ---

@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# --- generate (LLMs-from-scratch ch05: temperature, top-k, eos) ---

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# --- generate_text_basic (reasoning-from-scratch ch02) ---

@torch.inference_mode()
def generate_text_basic(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    model.eval()
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None and next_token.item() == eos_token_id):
            break
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids[:, input_length:]


# --- generate_text_basic_cache (reasoning-from-scratch ch02) ---

@torch.inference_mode()
def generate_text_basic_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
    generated_tokens = []
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None and next_token.item() == eos_token_id):
            break
        generated_tokens.append(next_token)
        out = model(next_token, cache=cache)[:, -1]
    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]


# --- generate_text_basic_stream_cache (reasoning-from-scratch ch02) ---

@torch.inference_mode()
def generate_text_basic_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]


# --- generate_text_temp_stream_cache (reasoning-from-scratch ch04) ---

def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumprobas = torch.cumsum(sorted_probas, dim=1)
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True
    kept_sorted = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom


@torch.inference_mode()
def generate_text_temp_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None, temperature=0.):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]


@torch.inference_mode()
def generate_text_top_p_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None, temperature=0., top_p=None):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
            probas = top_p_filter(probas, top_p)
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]


# --- generate_text_stream_concat_flex (reasoning-from-scratch ch04) ---

def generate_text_stream_concat_flex(model, tokenizer, prompt, device, max_new_tokens,
                                     verbose=False, generate_func=None, **generate_kwargs):
    if generate_func is None:
        generate_func = generate_text_basic_stream_cache
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generated_ids = []
    for token in generate_func(model=model, token_ids=input_ids,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=tokenizer.eos_token_id, **generate_kwargs):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)
    return tokenizer.decode(generated_ids)


# ##############################################################################
# MATH-500 EVALUATION (reasoning-from-scratch ch03)
# ##############################################################################

RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
LATEX_FIXES = [
    (r"\\left\s*", ""), (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""), (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"), (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")
SUPERSCRIPT_MAP = {chr(0x2070+i): str(i) for i in range(10)}
SUPERSCRIPT_MAP.update({"⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")"})


def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None
    current_idx = boxed_start_idx + len(r"\boxed")
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1
    if current_idx >= len(text) or text[current_idx] != "{":
        return None
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx
    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1
    if brace_depth != 0:
        return None
    return text[content_start_idx:current_idx-1]


def extract_final_candidate(text, fallback="number_then_full"):
    result = ""
    if text:
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]
            elif fallback == "number_then_full":
                result = text
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
                  lambda m: convert_superscripts(m.group(2), base=m.group(1)), text)
    text = convert_superscripts(text)
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(r"\\sqrt\s*\{([^}]*)\}", lambda m: f"sqrt({m.group(1)})", text)
    text = re.sub(r"\\sqrt\s+([^\\\s{}]+)", lambda m: f"sqrt({m.group(1)})", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", lambda m: f"({m.group(1)})/({m.group(2)})", text)
    text = re.sub(r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)", lambda m: f"({m.group(1)})/({m.group(2)})", text)
    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", lambda m: "+" + m.group(1), text)
    text = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", text)
    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    from sympy.parsing import sympy_parser as spp
    from sympy.core.sympify import SympifyError
    from sympy.polys.polyerrors import PolynomialError
    from tokenize import TokenError
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(expr, transformations=(*spp.standard_transformations,
                                                       spp.implicit_multiplication_application),
                              evaluate=True)
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None


def equality_check(expr_gtruth, expr_pred):
    from sympy import simplify
    from sympy.core.sympify import SympifyError
    if expr_gtruth == expr_pred:
        return True
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)
    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass
    return False


def split_into_parts(text):
    result = [text]
    if text:
        if len(text) >= 2 and text[0] in "([" and text[-1] in ")]" and "," in text[1:-1]:
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        result = []
    return result


def grade_answer(pred_text, gt_text):
    result = False
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(normalize_text(gt_text))
        pred_parts = split_into_parts(normalize_text(pred_text))
        if gt_parts and pred_parts and len(gt_parts) == len(pred_parts):
            result = all(equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts))
    return result


def render_prompt(prompt):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )


# ##############################################################################
# REASONING FUNCTIONS
# ##############################################################################

# --- heuristic_score (reasoning-from-scratch ch05) ---

def heuristic_score(answer, prompt=None, brevity_bonus=500.0, boxed_bonus=2.0,
                    extract_bonus=1.0, fulltext_bonus=0.0):
    score = 0.0
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(answer, fallback="number_then_full")
            if cand:
                score += fulltext_bonus
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


# --- self_consistency_vote (reasoning-from-scratch ch04) ---

def self_consistency_vote(model, tokenizer, prompt, device, num_samples=10,
                          temperature=0.8, top_p=0.9, max_new_tokens=2048,
                          show_progress=True, show_long_answer=False, seed=None):
    full_answers, short_answers = [], []
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)
        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p)
        short = extract_final_candidate(answer, fallback="number_then_full")
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] -> {short!r}")
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)
    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None
    return {"full_answers": full_answers, "short_answers": short_answers,
            "counts": dict(counts), "groups": groups,
            "majority_winners": majority_winners, "final_answer": final_answer}


# --- self_refinement_loop (reasoning-from-scratch ch05) ---

def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\nCritique:"
    )


def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\nRevised answer:"
    )


def self_refinement_loop(model, tokenizer, raw_prompt, device, iterations=2,
                         max_response_tokens=2048, max_critique_tokens=256,
                         score_fn=None, prompt_renderer=render_prompt,
                         prompt_suffix="", verbose=False, temperature=0.7, top_p=0.9):
    steps = []
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat_flex(
        model=model, tokenizer=tokenizer, prompt=prompt, device=device,
        max_new_tokens=max_response_tokens, verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature, top_p=top_p)
    current_extracted = extract_final_candidate(current_full, fallback="number_then_full")
    current_score = score_fn(answer=current_full, prompt=prompt) if score_fn else 0.0

    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score
        critique_prompt = make_critique_prompt(raw_prompt, draft_before_full)
        critique_full = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=critique_prompt, device=device,
            max_new_tokens=max_critique_tokens, verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p)
        refine_prompt = make_refine_prompt(raw_prompt, draft_before_full, critique_full)
        revised_full = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=refine_prompt, device=device,
            max_new_tokens=max_response_tokens, verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p)
        revised_extracted = extract_final_candidate(revised_full, fallback="number_then_full")
        revised_score = score_fn(answer=revised_full, prompt=prompt) if score_fn else 0.0
        step = {"iteration": it + 1, "draft_full": draft_before_full,
                "draft_extracted": draft_before_extracted, "critique": critique_full,
                "revised_full": revised_full, "revised_extracted": revised_extracted,
                "score_before": score_before, "score_after": revised_score}
        steps.append(step)
        if verbose:
            print(f"[Refinement {it+1}/{iterations}]\nCurrent: {draft_before_extracted}"
                  f"\nRevised: {revised_extracted}\nScore before: {score_before:.3f}"
                  f"\nScore after: {revised_score:.3f}\n{'='*25}")
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score
    return {"final_full": current_full, "final_extracted": current_extracted, "steps": steps}


# ##############################################################################
# TRAINING FUNCTIONS from LLMs-from-scratch
# ##############################################################################

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.cfg["context_length"]
    if model.pos_encoding == "learned":
        context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded,
                                         max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


# --- GRPO training functions (reasoning-from-scratch ch06) ---

def sample_response(model, tokenizer, prompt, device,
                    max_new_tokens=512, temperature=0.8, top_p=0.9):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]
    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas.cpu(), num_samples=1).to(device)
        token_id = next_token.item()
        generated.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        logits = model(next_token, cache=cache)[:, -1]
    full_token_ids = torch.cat([input_ids, torch.tensor(generated, device=device, dtype=input_ids.dtype)])
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(answer_text, fallback=None)
    if not extracted:
        return 0.0
    return float(grade_answer(extracted, ground_truth))


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])


def compute_grpo_loss(model, tokenizer, example, device, num_rollouts=2,
                      max_new_tokens=256, temperature=0.8, top_p=0.9):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])
    was_training = model.training
    model.eval()
    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        reward = reward_rlvr(text, example["answer"])
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append({"text": text, "reward": reward, "gen_len": token_ids.numel() - prompt_len})
    if was_training:
        model.train()
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss
    return {"loss": loss.item(), "pg_loss": pg_loss.item(), "rewards": roll_rewards,
            "advantages": advantages.detach().cpu().tolist(), "samples": samples,
            "loss_tensor": loss}


def train_rlvr_grpo(model, tokenizer, math_data, device, steps=None,
                    num_rollouts=2, max_new_tokens=256, temperature=0.8, top_p=0.9,
                    lr=1e-5, checkpoint_every=50, checkpoint_dir=".", csv_log_path=None):
    if steps is None:
        steps = len(math_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    if csv_log_path is None:
        csv_log_path = f"train_rlvr_grpo_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        for step in range(steps):
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]
            stats = compute_grpo_loss(model=model, tokenizer=tokenizer, example=example,
                                      device=device, num_rollouts=num_rollouts,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_p=top_p)
            stats["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(s["gen_len"] for s in stats["samples"])
            avg_response_len = step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            Path(csv_log_path).parent.mkdir(parents=True, exist_ok=True)
            if not csv_log_path.exists():
                csv_log_path.write_text("step,total_steps,loss,reward_avg,avg_response_len\n")
            with csv_log_path.open("a") as f:
                f.write(f"{current_step},{steps},{stats['loss']:.6f},{reward_avg:.6f},{avg_response_len:.6f}\n")
            print(f"[Step {current_step}/{steps}] loss={stats['loss']:.4f} reward_avg={reward_avg:.3f} avg_resp_len={avg_response_len:.1f}")
            if current_step % 10 == 0:
                print(f"[Step {current_step}] sample outputs:")
                for i, s in enumerate(stats["samples"][:3]):
                    print(f"  {i+1}) reward={s['reward']:.3f} len={s['gen_len']}: {s['text'][:100]}")
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"unified-grpo-step{current_step:05d}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
    except KeyboardInterrupt:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"unified-grpo-step{max(1,current_step):05d}-interrupt.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\nInterrupted. Saved checkpoint to {ckpt_path}")
    return model


# ##############################################################################
# TOKENIZERS
# ##############################################################################

class GPT2Tokenizer:
    """Thin wrapper around tiktoken for GPT-2 compatibility."""

    def __init__(self):
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.tokenizer.eot_token
        self.pad_token_id = self.tokenizer.eot_token

    def encode(self, text, allowed_special=None):
        if allowed_special is None:
            allowed_special = set()
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)


class Qwen3Tokenizer:
    """Exact copy from reasoning-from-scratch qwen3.py."""

    _SPECIALS = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path=None,
                 apply_chat_template=False, add_generation_prompt=False, add_thinking=False):
        from tokenizers import Tokenizer
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        if tokenizer_file_path is None:
            tokenizer_file_path = Path("tokenizer-base.json")
        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tok_path}' not found.")
        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            s = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
            if self.add_generation_prompt:
                s += "<|im_start|>assistant"
                if self.add_thinking:
                    s += "\n"
                else:
                    s += "\n<think>\n\n</think>\n\n"
            prompt = s
        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tok.decode(token_ids, skip_special_tokens=False)


# ##############################################################################
# WEIGHT LOADING
# ##############################################################################

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))


def load_gpt2_weights(model, params):
    """Load OpenAI GPT-2 checkpoint format into UnifiedModel (GPT mode)."""
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.weight = assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
        model.trf_blocks[b].att.W_key.weight = assign(model.trf_blocks[b].att.W_key.weight, k_w.T)
        model.trf_blocks[b].att.W_value.weight = assign(model.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.bias = assign(model.trf_blocks[b].att.W_query.bias, q_b)
        model.trf_blocks[b].att.W_key.bias = assign(model.trf_blocks[b].att.W_key.bias, k_b)
        model.trf_blocks[b].att.W_value.bias = assign(model.trf_blocks[b].att.W_value.bias, v_b)
        model.trf_blocks[b].att.out_proj.weight = assign(
            model.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model.trf_blocks[b].att.out_proj.bias = assign(
            model.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
        model.trf_blocks[b].ff.layers[0].weight = assign(
            model.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model.trf_blocks[b].ff.layers[0].bias = assign(
            model.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        model.trf_blocks[b].ff.layers[2].weight = assign(
            model.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model.trf_blocks[b].ff.layers[2].bias = assign(
            model.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
        model.trf_blocks[b].norm1.scale = assign(
            model.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        model.trf_blocks[b].norm1.shift = assign(
            model.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        model.trf_blocks[b].norm2.scale = assign(
            model.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        model.trf_blocks[b].norm2.shift = assign(
            model.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])
    model.out_head.weight = assign(model.out_head.weight, params["wte"])


def load_hf_qwen3_weights(model, params, n_layers):
    """Load HuggingFace Qwen3 checkpoint format into UnifiedModel (Qwen3 mode)."""
    def _assign(left, right, name=""):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch {name}: {left.shape} vs {right.shape}")
        left.copy_(right)

    _assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "embed_tokens")
    for l in range(n_layers):
        block = model.trf_blocks[l]
        att = block.att
        _assign(att.W_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"], f"q_proj.{l}")
        _assign(att.W_key.weight, params[f"model.layers.{l}.self_attn.k_proj.weight"], f"k_proj.{l}")
        _assign(att.W_value.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"], f"v_proj.{l}")
        _assign(att.out_proj.weight, params[f"model.layers.{l}.self_attn.o_proj.weight"], f"o_proj.{l}")
        if hasattr(att, "q_norm") and att.q_norm is not None:
            _assign(att.q_norm.scale, params[f"model.layers.{l}.self_attn.q_norm.weight"], f"q_norm.{l}")
        if hasattr(att, "k_norm") and att.k_norm is not None:
            _assign(att.k_norm.scale, params[f"model.layers.{l}.self_attn.k_norm.weight"], f"k_norm.{l}")
        _assign(block.norm1.scale, params[f"model.layers.{l}.input_layernorm.weight"], f"input_ln.{l}")
        _assign(block.ff.fc1.weight, params[f"model.layers.{l}.mlp.gate_proj.weight"], f"gate.{l}")
        _assign(block.ff.fc2.weight, params[f"model.layers.{l}.mlp.up_proj.weight"], f"up.{l}")
        _assign(block.ff.fc3.weight, params[f"model.layers.{l}.mlp.down_proj.weight"], f"down.{l}")
        _assign(block.norm2.scale, params[f"model.layers.{l}.post_attention_layernorm.weight"], f"post_ln.{l}")
    _assign(model.final_norm.scale, params["model.norm.weight"], "final_norm")
    if "lm_head.weight" in params:
        _assign(model.out_head.weight, params["lm_head.weight"], "lm_head")
    else:
        model.out_head.weight = model.tok_emb.weight


# ##############################################################################
# DEVICE DETECTION
# ##############################################################################

def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ##############################################################################
# DOWNLOAD FUNCTIONS from reasoning-from-scratch
# ##############################################################################

def _download_error_message(filename, url, primary_error, backup_url, backup_error):
    msg = f"Failed to download '{filename}' from {url}\n"
    msg += f"  Reason: {primary_error}\n"
    if backup_url:
        msg += f"  Backup {backup_url} also failed: {backup_error}\n"
        msg += "  This can happen on work/school machines where a VPN, proxy, or "
        msg += "antivirus tool intercepts HTTPS certificates.\n"
    msg += "  See troubleshooting: "
    msg += "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md"
    return msg


def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"  {dest} already up-to-date")
                    return True, None
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(f"\r{filename}: {pct:3d}% ({downloaded // (1024*1024)} MiB / {size_remote // (1024*1024)} MiB)")
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True, None
        except requests.RequestException as exc:
            return False, exc

    success, primary_error = try_download(url)
    if success:
        return dest

    backup_error = None
    if backup_url:
        print(f"Primary URL failed.\nTrying backup URL...")
        success, backup_error = try_download(backup_url)
        if success:
            return dest

    raise RuntimeError(_download_error_message(filename, url, primary_error, backup_url, backup_error))


def download_qwen3_small(kind="base", tokenizer_only=False, out_dir="."):
    files = {
        "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
        "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")
    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]
    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)


def cli_download(args):
    download_qwen3_small(kind=args.kind, tokenizer_only=args.tokenizer_only, out_dir=args.out_dir)
    return 0


# ##############################################################################
# CLI
# ##############################################################################

def cli_generate(args):
    arch = args.arch or "gpt"
    if arch == "gpt":
        cfg = GPT_CONFIG_124M
        cfg["context_length"] = args.context or 256
        tokenizer = GPT2Tokenizer()
    else:
        cfg = QWEN_CONFIG_06_B
        cfg["context_length"] = args.context or 1024
        if args.tokenizer:
            tokenizer = Qwen3Tokenizer(args.tokenizer)
        else:
            print("Error: --tokenizer required for qwen3 arch")
            return 1

    model = UnifiedModel(cfg)
    device = get_device()
    model.to(device)

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "Every effort moves you"

    if args.max_new_tokens is None:
        args.max_new_tokens = 50

    if arch == "gpt":
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate_text_simple(model, encoded, args.max_new_tokens, cfg["context_length"])
        print(token_ids_to_text(out, tokenizer))
    else:
        encoded = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        if model.pos_encoding == "rope":
            out_ids = generate_text_basic_cache(model, encoded, args.max_new_tokens,
                                                eos_token_id=tokenizer.eos_token_id)
        else:
            out_ids = generate_text_basic(model, encoded, args.max_new_tokens,
                                          eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out_ids.squeeze(0)))
    return 0


def cli_train(args):
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    cfg = {**GPT_CONFIG_124M, "context_length": args.context or 256}
    model = UnifiedModel(cfg)
    device = get_device()
    model.to(device)

    from torch.utils.data import Dataset, DataLoader

    class SimpleTextDataset(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids, self.target_ids = [], []
            token_ids = tokenizer.encode(txt)
            for i in range(0, len(token_ids) - max_length, stride):
                self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
                self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, i):
            return self.input_ids[i], self.target_ids[i]

    sample = "Every effort moves you. Every effort moves you forward. " * 100
    train_data = SimpleTextDataset(sample, tokenizer, args.context or 256, 256)
    val_data = SimpleTextDataset(sample, tokenizer, args.context or 256, 256)
    train_loader = DataLoader(train_data, batch_size=args.batch or 2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch or 2, shuffle=False, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs=args.epochs or 1, eval_freq=50, eval_iter=5,
                       start_context="Every effort", tokenizer=tokenizer)
    torch.save(model.state_dict(), "unified_model_trained.pth")
    print("Model saved to unified_model_trained.pth")
    return 0


def cli_chat(args):
    arch = args.arch or "gpt"
    if arch == "gpt":
        cfg = GPT_CONFIG_124M
        tokenizer = GPT2Tokenizer()
    else:
        cfg = QWEN_CONFIG_06_B
        if args.tokenizer:
            tokenizer = Qwen3Tokenizer(args.tokenizer)
        else:
            print("Error: --tokenizer required for qwen3 arch")
            return 1

    model = UnifiedModel(cfg)
    device = get_device()
    model.to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

    print(f"Unified {arch.upper()} Chat (Ctrl+D to exit)\n")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if not prompt.strip():
            continue
        encoded = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        if model.pos_encoding == "rope":
            out_ids = generate_text_basic_cache(model, encoded,
                                                args.max_tokens or 128,
                                                eos_token_id=tokenizer.eos_token_id)
        else:
            out_ids = generate_text_basic(model, encoded,
                                          args.max_tokens or 128,
                                          eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out_ids.squeeze(0)))
    return 0


def cli_grpo(args):
    if not args.tokenizer:
        print("Error: --tokenizer required for GRPO")
        return 1
    cfg = QWEN_CONFIG_06_B
    model = UnifiedModel(cfg)
    device = get_device()
    model.to(device)

    tokenizer = Qwen3Tokenizer(args.tokenizer, add_generation_prompt=True, add_thinking=True)

    import json
    if args.data:
        with open(args.data) as f:
            math_data = json.load(f)
    else:
        math_data = [{"problem": "Compute 1/2 + 1/6.", "answer": "2/3"},
                     {"problem": "What is 2+2?", "answer": "4"}]

    train_rlvr_grpo(model, tokenizer, math_data, device, steps=args.steps or 5,
                    num_rollouts=args.rollouts or 2,
                    max_new_tokens=args.max_tokens or 128,
                    temperature=0.8, top_p=0.9,
                    lr=1e-5, checkpoint_dir=args.checkpoint_dir or "grpo_checkpoints")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Unified LLM + Reasoning Model")
    sub = parser.add_subparsers(dest="mode")

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--prompt", default=None)
    p_gen.add_argument("--arch", choices=["gpt", "qwen3"], default="gpt")
    p_gen.add_argument("--context", type=int)
    p_gen.add_argument("--max-new-tokens", type=int)
    p_gen.add_argument("--tokenizer")

    p_train = sub.add_parser("train", help="Train GPT model")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch", type=int, default=2)
    p_train.add_argument("--context", type=int, default=256)

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--arch", choices=["gpt", "qwen3"], default="gpt")
    p_chat.add_argument("--max-tokens", type=int, default=128)
    p_chat.add_argument("--tokenizer")
    p_chat.add_argument("--checkpoint")

    p_grpo = sub.add_parser("grpo", help="GRPO reinforcement learning")
    p_grpo.add_argument("--tokenizer", required=True)
    p_grpo.add_argument("--data")
    p_grpo.add_argument("--steps", type=int, default=5)
    p_grpo.add_argument("--rollouts", type=int, default=2)
    p_grpo.add_argument("--max-tokens", type=int, default=128)
    p_grpo.add_argument("--checkpoint-dir", default="grpo_checkpoints")

    p_dl = sub.add_parser("download", help="Download Qwen3 model/tokenizer")
    p_dl.add_argument("--kind", choices=["base", "reasoning"], default="base")
    p_dl.add_argument("--tokenizer-only", action="store_true")
    p_dl.add_argument("--out-dir", default=".")

    args = parser.parse_args()
    if args.mode == "generate":
        return cli_generate(args)
    elif args.mode == "train":
        return cli_train(args)
    elif args.mode == "chat":
        return cli_chat(args)
    elif args.mode == "grpo":
        return cli_grpo(args)
    elif args.mode == "download":
        return cli_download(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit(main())
