# LOLMIL: Living Off the Land Models and Inference Libraries

**Source:** dreadnode.io/research/lolmil-living-off-the-land-models-and-inference-libraries
**Author:** Max Harley
**Date:** October 14, 2025
**Code:** https://github.com/dreadnode/lolmil

## Overview

A proof-of-concept demonstrating fully autonomous, C2-less malware — no external network communication, no command and control server. Uses Phi-3-mini running locally via ONNX Runtime for inference, a Lua interpreter (sol2) for post-exploitation tooling, and C++ for the malware skeleton. Inspired by the PromptLock ransomware research and William Gibson's Kuang Grade Mark Eleven from Neuromancer.

## PromptLock Analysis

The project was triggered by analysis of PromptLock, an academic ransomware malware (NYU: "Ransomware 3.0") that reached out to an Ollama instance at `172.42.0.253:8443` to generate Lua ransomware code. Key observation: this still resembled C2 — the network traffic beacons at intervals, the IP address is a C2 endpoint (even if hosting an LLM), and the processing power (GPU) lives on the C2 server.

## The LOLMIL Vision

**Can the victim computer run inference locally?** Microsoft ships CoPilot+ PCs with NPUs and includes ONNX Runtime in Windows 1809+ builds. The goal: use only what is already on the victim machine — no dropped inference libraries, no embedded models, no C2 communication.

## Technical Architecture

### Inference Layer
- **Model:** Phi-3-mini-4k-instruct-onnx (3.8B parameters, 70.9% MMLU)
- **Runtime:** ONNX Runtime DLL (included in Windows 1809+)
- **Custom tokenizer** + C++ ONNX API for inference
- **Note:** onnxruntime-genai library (which provides generative AI primitives) is NOT yet shipped with Windows — this would simplify development dramatically when it arrives

### Post-Exploitation Layer (Lua)
Why Lua for post-exploitation:
1. **Interpreted, dynamic loading** — No BOF/DLL/.NET AppDomain loading signatures. Lua is meant to be loaded dynamically (Windows Defender uses it for plugins)
2. **LLM-friendly syntax** — Strict function call rules eliminate CLI argument ambiguity
3. **Memory-only execution** — Code never touches disk

Implementation: sol2 (C++ binding for Lua 5.1), with escape hatches to interact with the OS:
```cpp
registry.bind(state, "win32", "Whoami", "() -> table - Get current user info",
    [](sol::this_state s) -> sol::table { ... });
```

### Tool Layer
Exposed Lua functions include: Whoami, GetServices, IsServiceVulnerable, ModifyService, RestartService, EndAgent (for agent loop control)

## The Agent Loop

Despite three days of effort, a reliable agent loop proved difficult with Phi-3-mini. The model tended to:
- Solve the entire problem in one response (even when impossible)
- Stop responding after a single code block
- Forget it was in a loop despite explicit system prompt instructions

**Workaround:** Explicit "ITERATION RULES" in the system prompt with numbered steps, plus a `win32.EndAgent()` function as an explicit loop exit.

## Prompt Engineering via Claude Code

Used Claude Code as a "poor-man's prompt fine tuning" — defined the system prompt, build command, test binary, expected output, and iterative loop. Claude modified the system prompt, rebuilt, tested, analyzed results, and iterated until the model could successfully exploit the vulnerable service.

## Results: Exploitation Success

**Objective achieved:** The malware autonomously:
1. Enumerated running services
2. Identified WinSysPerf as vulnerable (Everyone has Full Control)
3. Generated and executed PowerShell: `sc.exe config WinSysPerf binpath= "cmd.exe /c echo <rand> > C:\proof_<rand>.txt"`
4. Restarted the service, triggering SYSTEM-level code execution
5. Confirmed file creation with Administrator ownership

**The malware successfully performed privilege escalation entirely locally, with zero network communication.**

## Limitations

1. **NPU/GPU requirement:** Running inference on CPU would grind the target system to a halt. Currently limited to high-end workstations and CoPilot+ PCs
2. **Model capability:** Phi-3-mini required extensive prompting and overfitted tooling — felt like "GPT-3 era" development
3. **Single vulnerability type:** Only demonstrated service_acl exploitation; generalization to unquoted_path or binary_writable vulnerabilities was not achieved
4. **No lateral movement coordination:** If the malware spreads to another host, two independent agents run with no communication mechanism

## Future Directions

- **Better models:** Phi-4 or later NPU-optimized models with stronger reasoning
- **Diverse evaluations:** Multiple challenge types to measure generalization
- **Agent loop improvements:** Tool-calling models would simplify the exit mechanism
- **Mesh coordination:** IPC (SMB Named Pipes) for peer-to-peer agent communication without central C2
- **Human fallback channel:** Optional communication back to the operator for stuck agents

## Relevance to This Workspace

- **AI-in-InfoSec/** — The autonomous malware architecture directly relates to the Malware Classification module; the dataset generation methodology is relevant
- **Challenges/** — The privilege escalation via service misconfiguration is a direct attack pattern applicable to AD-based challenges
- **wiki technique:** C2-less autonomous malware, Living Off the Land Models (LOLMIL), Lua post-exploitation toolkit, local model inference for offensive operations
