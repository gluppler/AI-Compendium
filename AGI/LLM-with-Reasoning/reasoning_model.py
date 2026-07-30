#!/usr/bin/env python3
"""
Reasoning Model — Qwen3-style architecture from "Reasoning from Scratch"
(Raschka, 2025). Reproduced exactly from the original project's qwen3.py,
ch02-ch08.

Components: RoPE, Grouped-Query Attention, SwiGLU, RMSNorm, KV cache.
Includes MATH-500 evaluation, self-consistency, self-refinement, and GRPO.

Usage:
    python reasoning_model.py generate --prompt "Solve: 2+2" --tokenizer tokenizer-base.json
    python reasoning_model.py chat --tokenizer tokenizer-base.json
    python reasoning_model.py grpo --steps 5 --rollouts 4 --tokenizer tokenizer-base.json
    python reasoning_model.py download --kind base --tokenizer-only
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class TransformerBlock(nn.Module):
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


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        cos, sin = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"],
                                  bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx, cache=None):
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
                ckpt_path = ckpt_dir / f"qwen3-grpo-step{current_step:05d}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
    except KeyboardInterrupt:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"qwen3-grpo-step{max(1,current_step):05d}-interrupt.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\nInterrupted. Saved checkpoint to {ckpt_path}")
    return model


class Qwen3Tokenizer:
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


def load_qwen3_weights(model, params, n_layers):
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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
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


def cli_generate(args):
    cfg = dict(QWEN_CONFIG_06_B)
    cfg["context_length"] = args.context or 1024
    if args.tokenizer:
        tokenizer = Qwen3Tokenizer(args.tokenizer)
    else:
        print("Error: --tokenizer required")
        return 1
    model = Qwen3Model(cfg)
    device = get_device()
    model.to(device)
    prompt = args.prompt or "Solve the following."
    max_new = args.max_new_tokens or 50
    encoded = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    out_ids = generate_text_basic_cache(model, encoded, max_new,
                                        eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out_ids.squeeze(0)))
    return 0


def cli_chat(args):
    if args.tokenizer:
        tokenizer = Qwen3Tokenizer(args.tokenizer)
    else:
        print("Error: --tokenizer required")
        return 1
    cfg = dict(QWEN_CONFIG_06_B)
    model = Qwen3Model(cfg)
    device = get_device()
    model.to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    max_tokens = args.max_tokens or 128
    print("Qwen3 Chat (Ctrl+D to exit)\n")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if not prompt.strip():
            continue
        encoded = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        out_ids = generate_text_basic_cache(model, encoded, max_tokens,
                                            eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out_ids.squeeze(0)))
    return 0


def cli_grpo(args):
    if not args.tokenizer:
        print("Error: --tokenizer required for GRPO")
        return 1
    cfg = dict(QWEN_CONFIG_06_B)
    model = Qwen3Model(cfg)
    device = get_device()
    model.to(device)
    tokenizer = Qwen3Tokenizer(args.tokenizer, add_generation_prompt=True, add_thinking=True)

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


def cli_download(args):
    download_qwen3_small(kind=args.kind, tokenizer_only=args.tokenizer_only, out_dir=args.out_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qwen3-style Reasoning Model")
    sub = parser.add_subparsers(dest="mode")

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--prompt", default=None)
    p_gen.add_argument("--context", type=int)
    p_gen.add_argument("--max-new-tokens", type=int)
    p_gen.add_argument("--tokenizer")

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--max-tokens", type=int, default=128)
    p_chat.add_argument("--tokenizer", required=True)
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
