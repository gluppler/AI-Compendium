# LLM-with-Reasoning

A consolidated implementation of two architectures from scratch — GPT-style (from
[*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch))
and Qwen3-style with reinforcement learning (from
[*Reasoning from Scratch*](https://github.com/rasbt/reasoning-from-scratch)).

All components are in plain PyTorch — no HF Transformers, no TensorFlow, no black boxes.

## Files

| File | Lines | What |
|---|---|---|
| `llm_model.py` | 461 | GPT-2-style: learned pos embeddings, MHA, GELU, LayerNorm |
| `reasoning_model.py` | 1062 | Qwen3-style: RoPE, GQA, SwiGLU, RMSNorm, MATH eval, GRPO |
| `unified_model.py` | 1590 | Both architectures in one file, auto-detected from config |
| `finetune.py` | 610 | Supervised fine-tuning (SFT) + LoRA for all three models |
| `Makefile` | 39 | Common tasks: generate, test, train, clean |
| `pyproject.toml` | — | Project metadata + CLI entry points |
| `requirements.txt` | — | Dependencies |
| `tests/` | 4 files, 41 tests | pytest suite for all modules |

## Quick Start

```bash
pip install -r requirements.txt

# GPT generate
python llm_model.py generate --prompt "Hello"

# GPT train on dummy text
python llm_model.py train --epochs 1 --batch 2

# Qwen3 download + generate
python reasoning_model.py download --kind base --tokenizer-only
python reasoning_model.py generate --prompt "Solve 2+2" --tokenizer tokenizer-base.json
```

## Architecture Comparison

| Component | GPT (`llm_model.py`) | Qwen3 (`reasoning_model.py`) |
|---|---|---|
| Position encoding | Learned (`nn.Embedding`) | Rotary (RoPE) |
| Attention | Multi-Head (MHA) | Grouped-Query (GQA) |
| Feed-forward | GELU (2-layer) | SwiGLU (3-layer) |
| Normalization | LayerNorm (mean+var) | RMSNorm (rms) |
| KV cache | No | Yes |
| Parameters (config) | 124M | 0.6B |

## Supervised Fine-Tuning (`finetune.py`)

Takes instruction-response pairs and trains via cross-entropy loss **only on the response tokens** (instruction tokens are masked with `-100`). Requires pretrained weights (`--load`) to produce meaningful results — without them, the model is random and SFT has nothing to build on.

### Data formats (JSONL)

```jsonl
{"instruction": "What is 2+2?", "output": "4"}
{"prompt": "What is the capital?", "response": "Paris"}
{"messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"}
]}
```

### Examples

```bash
# GPT SFT — load real GPT-2 weights, train on instructions
python finetune.py --model gpt --data instruct.jsonl \
    --load gpt2-124M.npz --epochs 3

# Qwen3 SFT with LoRA
python finetune.py --model qwen3 --data math.jsonl \
    --load qwen3-0.6B-base.pth --tokenizer tokenizer-base.json \
    --lora-r 8 --epochs 5

# Unified auto-detect
python finetune.py --model unified --arch gpt --data instruct.jsonl \
    --load gpt2-124M.npz
```

### LoRA

Pass `--lora-r 8` to wrap attention projections with low-rank adapters. Base weights are frozen; only LoRA params train. Saves standard `.pth` that loads into the full model.

## Features

### Generation
- **Greedy** — `generate_text_simple`, `generate_text_basic`
- **Temperature + top-k** — `generate`
- **KV-cache streaming** — `generate_text_basic_cache`, `generate_text_basic_stream_cache`
- **Temperature + top-p sampling** — `generate_text_temp_stream_cache`, `generate_text_top_p_stream_cache`

### MATH-500 Evaluation
```
grade_answer(extract_final_candidate(response), ground_truth)
```
- LaTeX `\boxed{}` extraction, SymPy equivalence checking, superscript normalization

### Reasoning
- **Self-consistency** — sample N responses, majority vote
- **Self-refinement** — critique-then-revise with heuristic scoring

### GRPO (Group Relative Policy Optimization)
- REINFORCE-style loss with advantage normalization
- Full training loop: checkpointing, CSV metrics, interrupt handling

## Weight Loading

```python
# GPT-2
params = np.load("gpt2-124M.npz", allow_pickle=True)["params"].item()
model = GPTModel(GPT_CONFIG_124M)
load_gpt2_weights(model, params)

# Qwen3
state = torch.load("qwen3-0.6B-base.pth", map_location="cpu")
model = Qwen3Model(QWEN_CONFIG_06_B)
load_qwen3_weights(model, state, n_layers=28)
```

## References

- [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) — Raschka, 2024
- [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch) — Raschka, 2025
- [Qwen3: Thought as a Service](https://arxiv.org/abs/2505.20988)
