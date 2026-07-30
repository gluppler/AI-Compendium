"""Tests for finetune.py — dataset, formatting, model building, LoRA, training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune import (
    FTConfig,
    InstructionDataset,
    format_gpt_entry,
    format_qwen_entry,
    build_model_and_tokenizer,
    apply_lora,
    get_trainable_params,
    load_jsonl,
    validate_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(n: int = 3) -> list[dict[str, str]]:
    return [
        {"instruction": f"Question {i}", "output": f"Answer {i}"}
        for i in range(n)
    ]


def _make_chat_data(n: int = 2) -> list[dict[str, str]]:
    return [
        {"messages": [
            {"role": "user", "content": f"Q{i}"},
            {"role": "assistant", "content": f"A{i}"},
        ]}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatGPT:
    def test_basic(self) -> None:
        result = format_gpt_entry({"instruction": "Hi", "output": "Hello"})
        assert "### Instruction:" in result
        assert "Hi" in result
        assert "Hello" in result
        assert "### Response:" in result

    def test_with_input(self) -> None:
        result = format_gpt_entry({"instruction": "Add", "input": "1 2", "output": "3"})
        assert "### Input:" in result
        assert "1 2" in result

    def test_prompt_response_keys(self) -> None:
        result = format_gpt_entry({"prompt": "Q", "response": "A"})
        assert "Q" in result
        assert "A" in result


class TestFormatQwen:
    def test_basic(self) -> None:
        result = format_qwen_entry({"instruction": "Hi", "output": "Hello"})
        assert "<|im_start|>user" in result
        assert "<|im_end|>" in result
        assert "Hi" in result
        assert "Hello" in result

    def test_chat_messages(self) -> None:
        result = format_qwen_entry({
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]
        })
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "Q1" in result
        assert "A1" in result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestInstructionDataset:
    def _get_tokenizer(self, model: str = "gpt"):
        from llm_model import GPT2Tokenizer
        return GPT2Tokenizer()

    def test_len(self) -> None:
        tok = self._get_tokenizer()
        ds = InstructionDataset(_make_test_data(5), tok, 128, "gpt")
        assert len(ds) == 5

    def test_getitem_shapes(self) -> None:
        tok = self._get_tokenizer()
        ds = InstructionDataset(_make_test_data(3), tok, 64, "gpt")
        input_ids, labels = ds[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape == (64,)
        assert labels.shape == (64,)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long

    def test_loss_masking(self) -> None:
        tok = self._get_tokenizer()
        ds = InstructionDataset(_make_test_data(3), tok, 256, "gpt")
        _, labels = ds[0]
        n_ignore = (labels == -100).sum().item()
        n_train = (labels != -100).sum().item()
        assert n_ignore > 0, "instruction tokens should be masked"
        assert n_train > 0, "response tokens should be trainable"

    def test_padding(self) -> None:
        tok = self._get_tokenizer()
        ds = InstructionDataset(_make_test_data(3), tok, 512, "gpt")
        input_ids, labels = ds[0]
        n_pad = (input_ids == tok.pad_token_id).sum().item()
        assert n_pad > 0, "shorter sequences should be padded"

    def test_qwen_format(self) -> None:
        from reasoning_model import Qwen3Tokenizer
        tok = Qwen3Tokenizer(tokenizer_file_path=str(
            Path(__file__).resolve().parent.parent / "tokenizer-base.json"
        ))
        ds = InstructionDataset(_make_test_data(3), tok, 128, "qwen3")
        input_ids, labels = ds[0]
        assert input_ids.shape == (128,)
        assert labels.shape == (128,)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        with path.open("w") as f:
            for d in _make_test_data(3):
                f.write(json.dumps(d) + "\n")
        data = load_jsonl(path)
        assert len(data) == 3

    def test_load_json(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        with path.open("w") as f:
            json.dump(_make_test_data(3), f)
        data = load_jsonl(path)
        assert len(data) == 3

    def test_validate_data_good(self) -> None:
        validate_data(_make_test_data(1))
        validate_data(_make_chat_data(1))
        validate_data([{"prompt": "q", "response": "r"}])

    def test_validate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            validate_data([])


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_gpt_defaults(self) -> None:
        cfg = FTConfig(model="gpt", data_path=Path("/dev/null"), epochs=1)
        model, tokenizer = build_model_and_tokenizer(cfg)
        assert model is not None
        assert tokenizer is not None
        assert hasattr(tokenizer, "eos_token_id")
        # Should be on CPU since no device passed
        assert next(model.parameters()).device.type in ("cpu", "mps", "cuda")

    def test_qwen3_defaults(self) -> None:
        tok_path = Path(__file__).resolve().parent.parent / "tokenizer-base.json"
        if not tok_path.exists():
            pytest.skip("tokenizer-base.json not found")
        cfg = FTConfig(
            model="qwen3",
            data_path=Path("/dev/null"),
            tokenizer_path=tok_path,
            epochs=1,
        )
        model, tokenizer = build_model_and_tokenizer(cfg)
        assert model is not None
        assert hasattr(tokenizer, "eos_token_id")

    def test_unified_gpt(self) -> None:
        cfg = FTConfig(model="unified", arch="gpt", data_path=Path("/dev/null"), epochs=1)
        model, tokenizer = build_model_and_tokenizer(cfg)
        assert model is not None
        assert tokenizer is not None


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class TestLoRA:
    def test_apply_lora_reduces_trainable(self) -> None:
        from llm_model import GPT_CONFIG_124M, GPTModel
        cfg = {**GPT_CONFIG_124M, "context_length": 64}
        model = GPTModel(cfg)
        all_params = sum(p.numel() for p in model.parameters())
        model = apply_lora(model, r=8)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < all_params
        assert trainable > 0

    def test_lora_forward(self) -> None:
        from llm_model import GPT_CONFIG_124M, GPTModel
        cfg = {**GPT_CONFIG_124M, "context_length": 64}
        model = GPTModel(cfg)
        model = apply_lora(model, r=8)
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 50257)

    def test_get_trainable_params(self) -> None:
        from llm_model import GPT_CONFIG_124M, GPTModel
        model = GPTModel({**GPT_CONFIG_124M, "context_length": 64})
        model = apply_lora(model, r=8)
        params = get_trainable_params(model)
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# End-to-end training (tiny smoke test)
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    def _make_test_data_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "train.jsonl"
        data = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(4)]
        with p.open("w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        return p

    def test_gpt_train_one_step(self, tmp_path: Path) -> None:
        from torch.utils.data import DataLoader
        from finetune import train_sft
        from llm_model import GPT_CONFIG_124M, GPTModel, GPT2Tokenizer

        data_path = self._make_test_data_file(tmp_path)
        data = load_jsonl(data_path)
        from finetune import InstructionDataset

        tok = GPT2Tokenizer()
        ds = InstructionDataset(data, tok, 64, "gpt")
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        model = GPTModel({**GPT_CONFIG_124M, "context_length": 64})
        device = torch.device("cpu")
        model.to(device)

        cfg = FTConfig(model="gpt", data_path=data_path, epochs=1, save_path=tmp_path / "out.pth")
        try:
            train_sft(model, tok, loader, loader, device, cfg)
        except Exception as e:
            pytest.fail(f"train_sft raised: {e}")

        assert (tmp_path / "out.pth").exists()
