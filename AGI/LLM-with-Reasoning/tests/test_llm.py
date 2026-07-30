"""Tests for llm_model.py — model construction, generation, and math grading."""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_model import GPT_CONFIG_124M, GPTModel, generate_text_simple, GPT2Tokenizer


class TestGPTConfig:
    def test_config_keys(self):
        assert GPT_CONFIG_124M["vocab_size"] == 50257
        assert GPT_CONFIG_124M["emb_dim"] == 768
        assert GPT_CONFIG_124M["n_heads"] == 12
        assert GPT_CONFIG_124M["n_layers"] == 12


class TestGPTModel:
    def test_model_creation(self):
        cfg = dict(GPT_CONFIG_124M, context_length=64)
        model = GPTModel(cfg)
        assert isinstance(model, GPTModel)
        total = sum(p.numel() for p in model.parameters())
        # GPT-2 124M with untied weights is ~162M
        assert 150_000_000 < total < 175_000_000

    def test_forward_shape(self):
        cfg = dict(GPT_CONFIG_124M, context_length=64)
        model = GPTModel(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 50257)

    def test_generate_shape(self):
        cfg = dict(GPT_CONFIG_124M, context_length=64)
        model = GPTModel(cfg)
        x = torch.randint(0, 100, (1, 8))
        out = generate_text_simple(model, x, max_new_tokens=5, context_size=64)
        assert out.shape == (1, 13)


class TestGPT2Tokenizer:
    def test_encode_decode(self):
        tok = GPT2Tokenizer()
        ids = tok.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        text = tok.decode(ids)
        assert "Hello" in text
        assert tok.eos_token_id is not None
