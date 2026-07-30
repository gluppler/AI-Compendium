"""Tests for reasoning_model.py — model, tokenizer, math grading, and GRPO."""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning_model import (
    QWEN_CONFIG_06_B, Qwen3Model, KVCache,
    get_last_boxed, extract_final_candidate, normalize_text,
    equality_check, grade_answer, heuristic_score,
)


class TestQwen3Config:
    def test_config_keys(self):
        assert QWEN_CONFIG_06_B["vocab_size"] == 151_936
        assert QWEN_CONFIG_06_B["emb_dim"] == 1024
        assert QWEN_CONFIG_06_B["n_heads"] == 16
        assert QWEN_CONFIG_06_B["n_layers"] == 28
        assert QWEN_CONFIG_06_B["n_kv_groups"] == 8


class TestQwen3Model:
    def test_model_creation(self):
        cfg = dict(QWEN_CONFIG_06_B, context_length=64, dtype=torch.float32)
        model = Qwen3Model(cfg)
        assert isinstance(model, Qwen3Model)

    def test_forward_shape(self):
        cfg = dict(QWEN_CONFIG_06_B, context_length=64, dtype=torch.float32)
        model = Qwen3Model(cfg)
        x = torch.randint(0, 100, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 151_936)

    def test_kv_cache(self):
        cfg = dict(QWEN_CONFIG_06_B, context_length=64, dtype=torch.float32)
        model = Qwen3Model(cfg)
        cache = KVCache(n_layers=cfg["n_layers"])
        model.reset_kv_cache()
        x = torch.randint(0, 100, (1, 8))
        logits1 = model(x, cache=cache)
        assert logits1.shape == (1, 8, 151_936)
        assert model.current_pos == 8
        x2 = torch.randint(0, 100, (1, 1))
        logits2 = model(x2, cache=cache)
        assert logits2.shape == (1, 1, 151_936)


class TestMathGrading:
    def test_get_last_boxed(self):
        assert get_last_boxed(r"The answer is \boxed{4}") == "4"
        assert get_last_boxed(r"\boxed{2/3}") == "2/3"
        assert get_last_boxed("no box") is None

    def test_extract_final_candidate(self):
        assert extract_final_candidate(r"\boxed{42}") == "42"
        assert extract_final_candidate("plain 3.14") == "3.14"

    def test_equality_check(self):
        assert equality_check("4", "4") is True
        assert equality_check("1/2", "0.5") is True
        assert equality_check("4", "5") is False

    def test_grade_answer_pipeline(self):
        pred = extract_final_candidate(r"\boxed{4}")
        assert grade_answer(pred, "4") is True
        pred = extract_final_candidate(r"\boxed{2/3}")
        assert grade_answer(pred, "2/3") is True
        pred = extract_final_candidate("wrong")
        assert grade_answer(pred, "4") is False

    def test_heuristic_score(self):
        s = heuristic_score(r"\boxed{42}")
        assert s > 1.0
        s2 = heuristic_score("no box here")
        assert s2 < 2.0
