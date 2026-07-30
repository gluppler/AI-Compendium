# Code Repository Summaries

## 1. Rigging
**URL:** https://github.com/dreadnode/rigging
**Owner:** dreadnode
**Description:** Lightweight LLM interaction framework for building AI-powered applications. Core dependency for all Dreadnode agent harnesses. Provides chat pipelines, tool calling (`@rg.tool`, `@rg.tool_method`), structured output parsing (Pydantic models as XML), and generator abstraction (OpenAI, Anthropic, Gemini, Together AI, local models). Used in: AIRTBench, kerberoasting eval, evals blog, PentestJudge.
**Relevance:** Directly usable for building agent harnesses in any workspace challenge. Replaces ad-hoc LLM interaction code.

## 2. Dreadnode Strikes SDK
**URL:** https://github.com/dreadnode/sdk
**Owner:** dreadnode
**Description:** Official SDK for building and running AI security challenges. Provides the `dn` module with `dn.run()`, `dn.task`, `dn.log_param()`, `dn.log_metric()`, `dn.log_input()`, `dn.log_output()` for evaluation tracking. Includes 45+ attack strategies, 450+ transforms, 130+ scorers for AI red teaming.
**Relevance:** The evaluation primitives (param/metric tracking, task spans, runs) provide a pattern for systematic challenge evaluation.

## 3. Ares
**URL:** https://github.com/dreadnode/ares
**Owner:** dreadnode
**Description:** Autonomous multi-agent system for running red and blue team evaluations against live Active Directory environments. 7 specialized red agents (recon, credential access, cracker, ACL, privilege escalation, lateral movement, coercion) + 3 blue agents (triage, threat hunter, lateral analyst). Closed-loop scoring against attacker ground truth. >95% Domain Admin success rate on three-forest AD environments.
**Relevance:** Directly applicable to AD-based challenges (Doctrine-Studio, Prometheon). The multi-agent architecture pattern is reusable.

## 4. DreadGOAD
**URL:** https://github.com/dreadnode/DreadGOAD
**Owner:** dreadnode
**Description:** Reproducible, programmatically deployable Active Directory lab environment for agent evaluation. Forked from GOAD. Features unified Go CLI, AWS Terraform deployment, automated vulnerability validation (50+ vulns), variant generator (graph-isomorphic with randomized entity names). Golden AMIs via warpgate.
**Relevance:** Directly usable for standing up reproducible AD labs for challenge testing.

## 5. nerve
**URL:** https://github.com/dreadnode/nerve
**Owner:** dreadnode
**Description:** Create LLM agents without writing code. Visual/no-code interface for defining agent behaviors, tools, and workflows.
**Relevance:** Useful for rapid prototyping of agent strategies before implementing in code.

## 6. Marque
**URL:** https://github.com/dreadnode/marque
**Owner:** dreadnode
**Description:** Experimental Python workflows for AI agent development. Research-oriented patterns and utilities for agent construction.
**Relevance:** Source of patterns for agent architecture design.

## 7. Parley
**URL:** https://github.com/dreadnode/Parley
**Owner:** dreadnode
**Description:** TAP (Tree of Attacks with Pruning) jailbreaking implementation. Generates candidate adversarial prompts, evaluates them with a judge model, and prunes low-scoring branches. Tree-structured search of prompt space.
**Relevance:** Directly applicable to Prompt-Injection-Attacks jailbreak endpoints. The TAP algorithm is referenced extensively in 186 Jailbreaks and Redefining AI Red Teaming blogs.

## 8. Agent Lens
**URL:** https://github.com/dreadnode/agent-lens
**Owner:** dreadnode
**Description:** Agent observability and replay tooling for AI safety and interpretability research. Captures and visualizes agent decision trajectories.
**Relevance:** Useful for debugging agent behavior and understanding failure modes in complex solver scripts.

## 9. dyana
**URL:** https://github.com/dreadnode/dyana
**Owner:** dreadnode
**Description:** Sandbox environment for loading, running, and profiling a range of model files. Safe execution of untrusted ML model formats.
**Relevance:** Applicable to AI-Data-Attacks pickle deserialization and model file upload challenges.

## 10. burpference
**URL:** https://github.com/dreadnode/burpference
**Owner:** dreadnode
**Description:** Add LLM inference capabilities to BurpSuite for AI-powered security testing. Integrates LLM-based analysis into web application penetration testing workflows.
**Relevance:** Relevant to web-focused challenges and LLM-Output-Attacks module.

## 11. Research
**URL:** https://github.com/dreadnode/research
**Owner:** dreadnode
**Description:** General research code and experiments from the Dreadnode team. Contains datasets, notebooks, and experimental code referenced in various blog posts.
**Relevance:** Source of reference implementations and datasets.

## 12. Charcuterie
**URL:** https://github.com/moohax/Charcuterie
**Owner:** moohax (Will Pearce)
**Description:** Collection of code execution techniques for ML systems. Demonstrates how ML model formats (pickle, PyTorch, TensorFlow, Keras, ONNX) can be weaponized for arbitrary code execution during deserialization.
**Relevance:** Directly applicable to AI-Data-Attacks/5-Pickles and CWE-502 deserialization findings.

## 13. Counterfit
**URL:** https://github.com/Azure/counterfit
**Owner:** Azure (original authors: Will Pearce et al.)
**Description:** CLI AI red team tool for assessing the security of ML systems. Command-line interface for running adversarial attacks against ML models. Predecessor to Dreadnode's current AI red teaming tooling.
**Relevance:** Foundational tool for AI-Evasion modules. The attack catalog (FGSM, DeepFool, JSMA, etc.) maps to the evasion techniques in this workspace.

## 14. Deep Drop
**URL:** https://github.com/moohax/Deep-Drop
**Owner:** moohax (Will Pearce)
**Description:** Machine learning enabled dropper for offensive security research. Demonstrates how ML models can be used to embed and deliver payloads.
**Relevance:** Connects AI-in-InfoSec (Malware) with adversarial ML techniques.

## 15. Proof Pudding
**URL:** https://github.com/moohax/Proof-Pudding
**Owner:** moohax (Will Pearce)
**Description:** Proofpoint model extraction attack research tool. Demonstrates techniques for extracting model architecture and weights through black-box API access.
**Relevance:** Directly applicable to AI-Privacy module (model theft) and Red-Teaming-AI (model extraction findings).

## 16. Koppeling
**URL:** https://github.com/monoxgas/Koppeling
**Owner:** monoxgas (Nick Landers)
**Description:** Adaptive DLL hijacking and dynamic export forwarding. Technique for hijacking DLL load order and redirecting exports for code execution.
**Relevance:** Advanced persistence technique relevant to the system-level attack surface.

## 17. sRDI
**URL:** https://github.com/monoxgas/sRDI
**Owner:** monoxgas (Nick Landers)
**Description:** Convert DLLs to position-independent shellcode. Enables reflective loading of DLLs entirely in memory without touching disk.
**Relevance:** Memory-only execution technique relevant to the malware and evasion modules.
