# Mine the Gap: Open-Source Tools for Measuring the AI Offense-Defense Gap

**Source:** dreadnode.io/research/mine-the-gap-open-source-tools-for-measuring-the-ai-offense-defense-gap
**Authors:** Jayson Grace, Martin Wendiggensen
**Date:** April 15, 2026
**Code:** https://github.com/dreadnode/ares | https://github.com/dreadnode/DreadGOAD

## Overview

Two open-source projects forming a closed-loop evaluation system for autonomous red and blue team agents: **DreadGOAD** (reproducible Active Directory lab environment) and **Ares** (autonomous multi-agent red+blue evaluation system). Together, they measure offensive and defensive AI agent performance against shared infrastructure with attacker ground truth.

## DreadGOAD: Reproducible AD Lab

A fork of the GOAD project rebuilt for automated, programmatic research use.

### Why Fork GOAD

GOAD provides one of the most realistic open-source AD training environments: multi-domain, multi-forest, 50+ real-world vulnerabilities (Kerberoasting, AS-REP roasting, ACL abuse chains, ADCS ESC1-8, delegation abuse, MSSQL attacks). But it was not built for automated research — provisioning was brittle, environments drifted between runs, and it could not spin up multiple labs in parallel.

### DreadGOAD Capabilities

1. **Unified CLI** — Single Go binary (`dreadgoad`) managing full lifecycle: deploy, provision, validate, health check, SSM access
2. **AWS IaC** — Terraform + Terragrunt with private networking, SSM (no public IPs), golden AMIs via warpgate pre-baking Windows updates, AD DS, and MSSQL
3. **Automated vulnerability validation** — Confirms all 50+ vulnerabilities present after each provisioning cycle
4. **Variant generator** — Graph-isomorphic variants with randomized entity names, preserving structural relationships and attack paths while preventing memorization

## Ares: Automated Red + Blue Team System

### Red Team: 7 Specialized Worker Agents

Orchestrated by an LLM-powered coordinator:

| Agent | Function |
|-------|----------|
| Recon Agent | Network scanning, service enumeration, BloodHound collection |
| Credential Access Agent | Password spraying, Kerberoasting, hash extraction |
| Cracker Agent | Offline hash cracking (rules + dictionary) |
| ACL Agent | AD ACL abuse, DCSync rights, ownership takeover, delegation |
| Privilege Escalation Agent | Certificate abuse (ESC1-8), delegation, CVE exploitation |
| Lateral Movement Agent | Remote execution (PSExec, WMI, WinRM), credential harvesting |
| Coercion Agent | NTLM coercion (PetitPotam, PrinterBug), relay attacks |

**Performance:** Domain dominance across a three-forest AD environment in under 6 minutes, executing multi-stage kill chains from initial credential access through Golden Ticket persistence. >95% success rate for achieving Domain Admin.

### Blue Team: 3 Specialized Investigation Agents

| Agent | Function |
|-------|----------|
| Triage Agent | Initial alert severity assessment, escalation decisions |
| Threat Hunter Agent | IOC detection, TTP identification, log correlation |
| Lateral Analyst Agent | Scope expansion analysis, compromise chain mapping |

### Investigation Stages

1. **Triage** — What is happening?
2. **Causation** — Why did it happen?
3. **Lateral analysis** — What is the scope?
4. **Synthesis** — Generate the report

Agents operate under strict token, time, and query budgets mirroring real operational constraints.

### Results from 208 Automated Investigations

- 19 alert types processed, spanning 31 MITRE ATT&CK techniques
- Average 28 evidence items per investigation
- 10 investigations independently validated as true positives via direct red-blue trace correlation

## Evaluation Workflow (6 Steps)

1. Deploy DreadGOAD lab
2. Validate all vulnerabilities present
3. Run Ares: red agents execute full kill chains while telemetry is captured
4. Blue agents triage, investigate, reconstruct attack chain
5. Score investigation against attacker ground truth
6. Tear down, generate variant, repeat

## Relevance to This Workspace

- **AI-Defense/** — The closed-loop red/blue evaluation provides a template for testing the Guardrails and Adversarial Training sections
- **Challenges/** — DreadGOAD's reproducible AD environment with 50+ vulnerabilities is directly usable for pending AD-based challenges
- **AI-in-InfoSec/** — The blue team investigation pipeline (alert triage → causation → lateral → synthesis) models the network anomaly detection workflow
- **wiki technique:** Closed-loop red/blue agent evaluation, reproducible AD lab provisioning, multi-agent autonomous pentesting
