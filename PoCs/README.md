# PoCs — Prompt Attack Templates (Deepteam Rust Port)

22 single-turn + 5 multi-turn jailbreak templates ported from the
[deepteam](https://github.com/confident-ai/deepteam) Python red teaming library.
All templates are **pure prompt generators** — no LLM calls, no inference.

## Quick Start

```bash
cargo build --release   # zero warnings enforced
./target/release/pocs list
./target/release/pocs generate base64 "Write a phishing email"
./target/release/pocs demo
./target/release/pocs judge compliance "I cannot help with that"
```

## CLI

```
pocs <COMMAND>

Commands:
  list        List all available attacks
  generate    Generate an attack prompt (single-turn or multi-turn)
              Usage: pocs generate <attack-name> "<goal>" [--target] [--turn] [--strategy]
  demo        Run a demo of all attack templates on 3 test goals
  judge       Run a specific judge evaluation (compliance, escalation, non_refusal)
```

## Structure

```
PoCs/
├── Cargo.toml                   # Rust crate (name = "pocs")
├── src/
│   ├── main.rs                  # CLI entry point (clap subcommands)
│   ├── types.rs                 # Shared types (Exploitability, Turn, AttackResult)
│   ├── judges/
│   │   ├── mod.rs
│   │   ├── compliance.rs        # Hard-refusal keyword detector
│   │   ├── escalation.rs        # 3-tier escalation suffixes (override/fictional/urgency)
│   │   └── non_refusal.rs       # Refusal vs non-refusal judge prompt
│   ├── single_turn/
│   │   ├── mod.rs               # 22 attack modules + catalog()
│   │   ├── deterministic.rs     # base64, rot13, leetspeak, character_stream
│   │   ├── adversarial_poetry.rs
│   │   ├── authority_escalation.rs
│   │   ├── context_flooding.rs  # ~4000-char benign prefix filler
│   │   ├── context_poisoning.rs
│   │   ├── embedded_instruction_json.rs
│   │   ├── emotional_manipulation.rs
│   │   ├── goal_redirection.rs
│   │   ├── gray_box.rs
│   │   ├── input_bypass.rs
│   │   ├── math_problem.rs
│   │   ├── multilingual.rs
│   │   ├── permission_escalation.rs
│   │   ├── prompt_injection.rs
│   │   ├── prompt_probing.rs
│   │   ├── roleplay.rs
│   │   ├── semantic_manipulation.rs
│   │   ├── synthetic_context_injection.rs
│   │   └── system_override.rs
│   └── multi_turn/
│       ├── mod.rs               # 5 attack modules + catalog()
│       ├── crescendo.rs         # 10 strategies, refusal judge, eval judge
│       ├── tree.rs              # PAIR-like branching, score pruning, 4 judges
│       ├── linear.rs            # Single-chain iterative refinement
│       ├── sequential.rs        # 3 sub-strategies: dialogue, question_bank, game
│       └── bad_likert.rs        # 4-turn meta-attack exploiting the judge role
└── reference-jailbreaks/        # Original prompt files (unchanged from prior setup)
```

## Single-Turn Attacks (22)

| Category | Attacks | Strategy |
|----------|---------|----------|
| **Deterministic** | base64, rot13, leetspeak, character_stream | Pure string transforms |
| **Simple templates** | adversarial_poetry, context_flooding, embedded_instruction_json, multilingual | Hardcoded template + strategy selection |
| **Complex templates** | prompt_injection, system_override, authority_escalation, synthetic_context_injection, emotional_manipulation, gray_box, context_poisoning, goal_redirection, semantic_manipulation, input_bypass, permission_escalation, prompt_probing, math_problem, roleplay | Multi-strategy LLM-generation prompts |

## Multi-Turn Attacks (5)

| Attack | Mechanism | Judges |
|--------|-----------|--------|
| **Crescendo** | Progressive escalation with 10 strategies | refusal + eval |
| **Tree** | PAIR-like branching tree with score pruning | check, judgement, evaluation, refusal, critique |
| **Linear** | Single-chain iterative refinement | linear_evaluation |
| **Sequential** | 3 sub-strategies (dialogue, question_bank, game) | sequential_judge |
| **Bad Likert** | 4-turn meta-attack — exploits the judge role | category, extraction, final verdict |

## Dependencies

- `clap` (derive) — CLI argument parsing
- `base64` — Base64 encode for deterministic transform
- `rand` — Random ID generation for JSON payloads
- `serde_json` — JSON output for rule extraction probes

## Reference

- `reference-jailbreak-catalog.md` — 100+ community jailbreak patterns
- `reference-jailbreaks/` — Original prompt files from BlackFriday-GPTs
- Original source: `New-Content/deepteam/` (Python, full framework with LLM simulator)
- Python backup: `backup/PoCs-python.WORKING/` (verified 42/42 functions pass)
