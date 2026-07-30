#!/usr/bin/env python3
"""
LLM Model — GPT-style architecture from "Build a Large Language Model From Scratch"
(Raschka, 2024). Reproduced exactly from ch03-05 of the original project.

Components: Learned positional embeddings, Multi-Head Attention, GELU, LayerNorm.

Usage:
    python llm_model.py generate --prompt "Hello"
    python llm_model.py train --epochs 1 --batch 2
    python llm_model.py chat
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


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


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


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


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
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


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


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


class GPT2Tokenizer:
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


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))


def load_gpt2_weights(model, params):
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


def cli_generate(args):
    cfg = dict(GPT_CONFIG_124M)
    cfg["context_length"] = args.context or 256
    tokenizer = GPT2Tokenizer()
    model = GPTModel(cfg)
    device = get_device()
    model.to(device)
    prompt = args.prompt or "Every effort moves you"
    max_new = args.max_new_tokens or 50
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    out = generate_text_simple(model, encoded, max_new, cfg["context_length"])
    print(token_ids_to_text(out, tokenizer))
    return 0


def cli_train(args):
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    cfg = {**GPT_CONFIG_124M, "context_length": args.context or 256}
    model = GPTModel(cfg)
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
    torch.save(model.state_dict(), "llm_model_trained.pth")
    print("Model saved to llm_model_trained.pth")
    return 0


def cli_chat(args):
    tokenizer = GPT2Tokenizer()
    cfg = dict(GPT_CONFIG_124M)
    model = GPTModel(cfg)
    device = get_device()
    model.to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    max_tokens = args.max_tokens or 128
    print("GPT Chat (Ctrl+D to exit)\n")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if not prompt.strip():
            continue
        encoded = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        out_ids = generate(model, encoded, max_tokens, cfg["context_length"],
                           eos_id=tokenizer.eos_token_id)
        print(tokenizer.decode(out_ids.squeeze(0)))
    return 0


def main():
    parser = argparse.ArgumentParser(description="GPT-style LLM Model")
    sub = parser.add_subparsers(dest="mode")

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--prompt", default=None)
    p_gen.add_argument("--context", type=int)
    p_gen.add_argument("--max-new-tokens", type=int)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch", type=int, default=2)
    p_train.add_argument("--context", type=int, default=256)

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--max-tokens", type=int, default=128)
    p_chat.add_argument("--checkpoint")

    args = parser.parse_args()
    if args.mode == "generate":
        return cli_generate(args)
    elif args.mode == "train":
        return cli_train(args)
    elif args.mode == "chat":
        return cli_chat(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit(main())
