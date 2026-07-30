#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) for all three model variants.

Supports:
  - llm_model.py (GPT-2 style, learned pos, MHA, GELU)
  - reasoning_model.py (Qwen3 style, RoPE, GQA, SwiGLU)
  - unified_model.py (auto-detected from config)

Data formats (JSONL):
  Alpaca:     {"instruction": "...", "input": "", "output": "..."}
  Simple:     {"prompt": "...", "response": "..."}
  Chat:       {"messages": [{"role": "user", "content": "..."},
                            {"role": "assistant", "content": "..."}]}

Usage:
    # GPT on Alpaca data
    python finetune.py --model gpt --data train.jsonl --epochs 3

    # Qwen3 with chat data, load pretrained weights
    python finetune.py --model qwen3 --data train.jsonl \\
        --load qwen3-0.6B-base.pth --tokenizer tokenizer-base.json

    # Unified with auto-detect
    python finetune.py --model unified --data train.jsonl \\
        --load gpt2-124M.pth --arch gpt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FTConfig:
    """Fine-tuning configuration with sensible defaults."""

    model: Literal["gpt", "qwen3", "unified"] = "gpt"
    arch: Literal["gpt", "qwen3"] | None = None
    data_path: Path | None = None
    load_path: Path | None = None
    save_path: Path = Path("finetuned.pth")
    tokenizer_path: Path | None = None
    context_length: int = 512
    batch_size: int = 4
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 20
    eval_freq: int = 50
    eval_steps: int = 5
    max_new_tokens: int = 128
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    seed: int = 42
    device: str | None = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

GPT_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

GPT_INPUT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

QWEN_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def format_gpt_entry(entry: dict[str, str]) -> str:
    """Format an Alpaca-style entry for GPT models."""
    instruction = entry.get("instruction", entry.get("prompt", ""))
    inp = entry.get("input", "")
    response = entry.get("output", entry.get("response", ""))
    if inp:
        return GPT_INPUT_TEMPLATE.format(instruction=instruction, input=inp, response=response)
    return GPT_PROMPT_TEMPLATE.format(instruction=instruction, response=response)


def format_qwen_entry(entry: dict[str, str]) -> str:
    """Format an entry for Qwen3 chat models."""
    if "messages" in entry:
        parts: list[str] = []
        for msg in entry["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)
    instruction = entry.get("instruction", entry.get("prompt", ""))
    response = entry.get("output", entry.get("response", ""))
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction, response=response)


def get_formatter(model_type: str) -> Callable[[dict[str, str]], str]:
    if model_type in ("qwen3", "unified"):
        return format_qwen_entry
    return format_gpt_entry


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for instruction fine-tuning with loss masking.

    The instruction portion of each example is masked (label = -100) so the
    loss is only computed over the response tokens.
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
        context_length: int,
        model_type: str,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.formatter = get_formatter(model_type)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.data[idx]
        full_text = self.formatter(entry)
        instruction_text, response_text = self._split(entry, full_text)

        full_ids: list[int] = self._tokenize(full_text)
        instruction_ids: list[int] = self._tokenize(instruction_text)

        resp_start = min(len(instruction_ids), len(full_ids))
        input_len = min(len(full_ids), self.context_length)

        input_ids: list[int] = full_ids[:input_len]
        labels: list[int] = (
            [-100] * min(len(instruction_ids), input_len)
            + full_ids[resp_start:input_len]
        )

        if len(labels) < self.context_length:
            pad_len = self.context_length - len(labels)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)

        input_ids = input_ids[:self.context_length]
        labels = labels[:self.context_length]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _split(self, entry: dict[str, str], full_text: str) -> tuple[str, str]:
        response = entry.get("output", entry.get("response", ""))
        if "messages" in entry:
            parts: list[str] = []
            for msg in entry["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            user_text = "\n".join(parts[:-1]) + "\n" if len(parts) > 1 else ""
            return user_text, parts[-1] if parts else ""
        if "<|im_start|>assistant" in full_text:
            idx = full_text.rfind("<|im_start|>assistant")
            prefix = full_text[:idx]
            return prefix.strip(), response
        prefix = full_text[:full_text.rfind(response)] if response else full_text
        return prefix.strip(), response

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer."""

    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.weight = original.weight
        self.bias = original.bias
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        result = result + self.scaling * lora_update
        return result


def _add_lora(
    module: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    names: frozenset[str] = frozenset(),
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (not names or child_name in names):
            lora = LoRALinear(child, r, alpha, dropout)
            setattr(module, child_name, lora)
        else:
            _add_lora(child, r, alpha, dropout, names)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Wrap attention projections with LoRA adapters."""
    _add_lora(model, r, alpha, dropout, names=frozenset({
        "W_query", "W_key", "W_value", "out_proj",
        "fc1", "fc2",
    }))
    return model


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute average loss over a data loader."""
    model.eval()
    total, count = 0.0, 0
    for i, (input_ids, labels) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        total += loss.item()
        count += 1
    model.train()
    return total / count if count else float("nan")


def generate_sample(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    """Generate a sample response for evaluation during training."""
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    generated = list(input_ids.squeeze(0).tolist())
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if hasattr(model, "pos_encoding"):
                logits = model(input_ids)
            else:
                logits = model(input_ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            generated.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id], dim=1)
    model.train()
    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_sft(
    model: nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: FTConfig,
) -> nn.Module:
    """Run supervised fine-tuning."""
    param_groups: list[dict] = []
    trainable = get_trainable_params(model)
    if trainable:
        param_groups.append({"params": trainable, "weight_decay": cfg.weight_decay})
    else:
        param_groups.append({"params": model.parameters(), "weight_decay": cfg.weight_decay})

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(cfg.warmup_steps, 1)),
    )

    global_step = 0
    best_val_loss = float("inf")

    print(f"Fine-tuning on {device} | "
          f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % cfg.eval_freq == 0:
                val_loss = evaluate(model, val_loader, device, cfg.eval_steps)
                avg_train = epoch_loss / max(epoch_steps, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Step {global_step:>6d} | epoch {epoch}/{cfg.epochs} | "
                      f"train {avg_train:.4f} | val {val_loss:.4f} | lr {lr_now:.2e}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  -> new best val loss ({best_val_loss:.4f})")

        avg_train = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch}/{cfg.epochs} complete | avg train loss {avg_train:.4f}")

    torch.save(model.state_dict(), cfg.save_path)
    print(f"Model saved to {cfg.save_path}")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load a JSONL or JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open() as f:
        first = f.read(1024).strip()
        f.seek(0)
        if first.startswith("["):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _has_only_keys(entry: dict, keys: set[str]) -> bool:
    return set(entry.keys()).issubset(keys)


def validate_data(data: list[dict[str, str]]) -> None:
    """Check that data entries have expected keys."""
    if not data:
        raise ValueError("Empty dataset")
    sample = data[0]
    valid_formats = [
        {"instruction", "output", "input"},
        {"instruction", "output"},
        {"prompt", "response"},
        {"messages"},
    ]
    if not any(_has_only_keys(sample, fmt) for fmt in valid_formats):
        print(f"Warning: unexpected keys {set(sample.keys())}. "
              f"Expected one of: instruction+output, prompt+response, messages")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(
    cfg: FTConfig,
) -> tuple[nn.Module, Any]:
    """Build model, tokenizer, and optionally load pretrained weights."""
    device = _resolve_device(cfg.device)

    if cfg.model == "gpt":
        from llm_model import GPT_CONFIG_124M, GPTModel, GPT2Tokenizer, load_gpt2_weights
        import numpy as np

        model_cfg = {**GPT_CONFIG_124M, "context_length": cfg.context_length}
        model: nn.Module = GPTModel(model_cfg)
        tokenizer = GPT2Tokenizer()

        if cfg.load_path:
            if cfg.load_path.suffix == ".npz":
                params = np.load(str(cfg.load_path), allow_pickle=True)["params"].item()
            elif cfg.load_path.suffix == ".pth":
                state = torch.load(str(cfg.load_path), map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)
                params = None
            else:
                raise ValueError(f"Unrecognized weight file: {cfg.load_path}")
            if params is not None:
                load_gpt2_weights(model, params)

    elif cfg.model == "qwen3":
        from reasoning_model import QWEN_CONFIG_06_B, Qwen3Model, Qwen3Tokenizer, load_qwen3_weights

        model_cfg = {**QWEN_CONFIG_06_B, "context_length": cfg.context_length, "dtype": torch.float32}
        model = Qwen3Model(model_cfg)
        tokenizer_path = cfg.tokenizer_path or Path("tokenizer-base.json")
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(tokenizer_path))

        if cfg.load_path:
            state = torch.load(str(cfg.load_path), map_location="cpu", weights_only=True)
            load_qwen3_weights(model, state, n_layers=QWEN_CONFIG_06_B["n_layers"])

    elif cfg.model == "unified":
        from unified_model import GPT_CONFIG_124M, QWEN_CONFIG_06_B, UnifiedModel
        from unified_model import GPT2Tokenizer, Qwen3Tokenizer, resolve_config
        from unified_model import load_gpt2_weights, load_hf_qwen3_weights
        import numpy as np

        arch = cfg.arch or "gpt"
        if arch == "gpt":
            base_cfg = {**GPT_CONFIG_124M, "context_length": cfg.context_length}
            model = UnifiedModel(base_cfg)
            tokenizer = GPT2Tokenizer()
            if cfg.load_path:
                if cfg.load_path.suffix == ".npz":
                    params = np.load(str(cfg.load_path), allow_pickle=True)["params"].item()
                else:
                    state = torch.load(str(cfg.load_path), map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=False)
                    params = None
                if params is not None:
                    load_gpt2_weights(model, params)
        else:
            base_cfg = {**QWEN_CONFIG_06_B, "context_length": cfg.context_length, "dtype": torch.float32}
            model = UnifiedModel(base_cfg)
            tp = cfg.tokenizer_path or Path("tokenizer-base.json")
            tokenizer = Qwen3Tokenizer(tokenizer_file_path=str(tp))
            if cfg.load_path:
                state = torch.load(str(cfg.load_path), map_location="cpu", weights_only=True)
                load_hf_qwen3_weights(model, state, n_layers=QWEN_CONFIG_06_B["n_layers"])
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    if cfg.lora_r > 0:
        model = apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
        print(f"Applied LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})")

    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> FTConfig:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for all models")
    parser.add_argument("--model", choices=["gpt", "qwen3", "unified"], default="gpt")
    parser.add_argument("--arch", choices=["gpt", "qwen3"], help="Arch for --model unified")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--load", type=Path, help="Pretrained weights")
    parser.add_argument("--save", type=Path, default=Path("finetuned.pth"))
    parser.add_argument("--tokenizer", type=Path, help="Qwen3 tokenizer JSON")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    return FTConfig(
        model=args.model,
        arch=args.arch,
        data_path=args.data,
        load_path=args.load,
        save_path=args.save,
        tokenizer_path=args.tokenizer,
        context_length=args.context,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        eval_freq=args.eval_freq,
        eval_steps=args.eval_steps,
        max_new_tokens=args.max_new_tokens,
        lora_r=args.lora_r,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    torch.manual_seed(cfg.seed)

    data = load_jsonl(cfg.data_path)
    validate_data(data)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    print(f"Loaded {len(data)} examples | train {len(train_data)} val {len(val_data)}")

    model, tokenizer = build_model_and_tokenizer(cfg)
    device = _resolve_device(cfg.device)
    print(f"Device: {device} | Model: {cfg.model}")

    train_ds = InstructionDataset(train_data, tokenizer, cfg.context_length, cfg.model)
    val_ds = InstructionDataset(val_data, tokenizer, cfg.context_length, cfg.model)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    model = train_sft(model, tokenizer, train_loader, val_loader, device, cfg)

    return 0


if __name__ == "__main__":
    exit(main())
