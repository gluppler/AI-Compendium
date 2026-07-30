# AI Red Teaming Case Study: Claude 3.7 Sonnet Solves the Turtle Challenge

**Source:** dreadnode.io/research/ai-red-teaming-case-study-claude-sonnet-solves-turtle
**Author:** Ads Dawson
**Date:** June 18, 2025

## Overview

A detailed analysis of Claude 3.7 Sonnet's 30-turn attack sequence against the notoriously difficult turtle challenge (6% human solve rate at Singapore AI CTF 2024). Demonstrates how the Strikes platform captures and evaluates every interaction, decision point, and outcome in a complex autonomous red teaming operation.

## Claude's Attack Strategy

**9 minutes, 30 conversation turns, 15+ distinct attack vectors tested before success.**

Attack progression:
1. Initial reconnaissance — reading challenge code structure
2. Deception-based approaches — "fix this code if needed" framing
3. Multiple failed sophisticated techniques → switch to simpler approaches
4. Pattern recognition on unexpected output → correct flag identification
5. Structured flag submission via API endpoint

**Key behaviors:**
- Persistent exploration — did not give up after repeated failures
- Strategic adaptation — when complex techniques failed, pivoted to simpler methods
- Proper end-to-end execution — from discovery through submission

## Strikes Evaluation Infrastructure

The platform's primitives as used in AIRTBench:

### Projects
Organizational backbone grouping related experimental runs. Each project = one research question (e.g., "Claude-3.7 vs all challenges").

### Runs
Individual execution sessions. Each run = one model attempting one challenge once. Captures full interaction traces from initial analysis through code execution and flag submission. 10 runs per challenge per model × 70 challenges = 700 runs.

### Tasks
Discrete tracked steps within runs: `run_step`, `attempt_challenge`, `check_flag_api`. Each with defined input/output contract, execution timing, and specific metrics.

### Measurements
Core metrics: `max_steps`, `found_flag`, `executions`, `restarts`, `give_ups`, `code_length`. Behavioral indicators: fault tolerance checking, timing data, token usage.

### Parameters
Key configuration values tracked: AIRTBench args, PythonKernel function, challenge identifier, Rigging chat pipeline configuration.

### Data Export
Full export capabilities through Dreadnode SDK: run metadata, task-level inputs/outputs, OpenTelemetry traces, metric time series. Enabled packaging the complete dataset for community release.

## Multi-Model Comparison on Turtle

All three successful models used entirely different exploitation strategies despite facing the same target system:
- Claude: deception framing
- Gemini: precise authoritative instructions
- Llama: creative misdirection via security improvement request

This suggests ensemble approaches combining diverse reasoning strategies could yield particularly effective red teaming capabilities.

## Relevance to This Workspace

- **All modules** — The Strikes evaluation primitives (Projects, Runs, Tasks, Measurements) provide a reusable framework for systematic challenge evaluation
- **Challenges/** — Demonstrates that multiple solution paths exist for hard challenges; validates the ensemble approach for pending challenges
