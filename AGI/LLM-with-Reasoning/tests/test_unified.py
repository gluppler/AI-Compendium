"""Tests for unified_model.py — config resolution and cross-architecture."""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unified_model import (
    GPT_CONFIG_124M, QWEN_CONFIG_06_B, resolve_config,
    UnifiedModel, generate_text_simple, generate_text_basic,
)


class TestResolveConfig:
    def test_gpt_config(self):
        cfg = resolve_config(GPT_CONFIG_124M)
        assert cfg["pos_encoding"] == "learned"
        assert cfg["norm_type"] == "layernorm"
        assert cfg["attn_type"] == "mha"

    def test_qwen3_config(self):
        cfg = resolve_config(QWEN_CONFIG_06_B)
        assert cfg["pos_encoding"] == "rope"
        assert cfg["norm_type"] == "rmsnorm"
        assert cfg["attn_type"] == "gqa"


class TestUnifiedModel:
    def test_gpt_forward(self):
        cfg = resolve_config({**GPT_CONFIG_124M, "context_length": 64})
        model = UnifiedModel(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 50257)

    def test_qwen3_forward(self):
        cfg = resolve_config({**QWEN_CONFIG_06_B, "context_length": 64, "dtype": torch.float32})
        model = UnifiedModel(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 151_936)

    def test_gpt_generate(self):
        cfg = resolve_config({**GPT_CONFIG_124M, "context_length": 64})
        model = UnifiedModel(cfg)
        x = torch.randint(0, 100, (1, 8))
        out = generate_text_simple(model, x, max_new_tokens=5, context_size=64)
        assert out.shape == (1, 13)

    def test_qwen3_generate_basic(self):
        cfg = resolve_config({**QWEN_CONFIG_06_B, "context_length": 64, "dtype": torch.float32})
        model = UnifiedModel(cfg)
        x = torch.randint(0, 100, (1, 8))
        out = generate_text_basic(model, x, max_new_tokens=5)
        assert out.shape == (1, 5)
