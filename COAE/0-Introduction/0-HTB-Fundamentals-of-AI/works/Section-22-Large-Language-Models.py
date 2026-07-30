# Source: HackTheBox, Fundamentals of AI, Section 22 — Large Language Models
# Section file: ../6-Introduction-to-Generative-AI/Section-22-Large-Language-Models.md
# All fenced code blocks extracted verbatim in the order they appear in the note.
# Note: the txt block is a generation output example, not executable code.

# ------------------------------------------------------------
# Tokenization example
# ------------------------------------------------------------

# The sentence "I love artificial intelligence" tokenized as word tokens:
["I", "love", "artificial", "intelligence"]

# ------------------------------------------------------------
# LLM autoregressive generation example (output, not executable)
# ------------------------------------------------------------

# Prompt: "Once upon a time, there was a cat named Whiskers."
# Model continuation (sampled autoregressively, one token at a time):
#
# Once upon a time, there was a cat named Whiskers. Whiskers was a curious
# and adventurous cat, always exploring the world around him. One day, he
# ventured into the forest and stumbled upon a hidden village of mice...
