.PHONY: generate-gpt generate-qwen3 train chat grpo download finetune install test lint clean

install:
	pip install -r requirements.txt

generate-gpt:
	python llm_model.py generate --prompt "Hello"

generate-qwen3:
	python reasoning_model.py generate --prompt "Solve 2+2" --tokenizer tokenizer-base.json

train:
	python llm_model.py train --epochs 1 --batch 2

chat-gpt:
	python llm_model.py chat

chat-qwen3:
	python reasoning_model.py chat --tokenizer tokenizer-base.json

grpo:
	python reasoning_model.py grpo --steps 5 --rollouts 2 --tokenizer tokenizer-base.json

download:
	python reasoning_model.py download --kind base --tokenizer-only

finetune-gpt:
	python finetune.py --model gpt --data train.jsonl --epochs 3

finetune-qwen3:
	python finetune.py --model qwen3 --data train.jsonl --epochs 3 --tokenizer tokenizer-base.json

finetune-lora:
	python finetune.py --model gpt --data train.jsonl --lora-r 8 --epochs 3

test:
	python -m pytest tests/ -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -f *.pth
	rm -f *.csv
	rm -rf grpo_checkpoints
	rm -f tokenizer-*.json
	rm -f qwen3-0.6B-*.pth

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sort | sed 's/:/  /'
