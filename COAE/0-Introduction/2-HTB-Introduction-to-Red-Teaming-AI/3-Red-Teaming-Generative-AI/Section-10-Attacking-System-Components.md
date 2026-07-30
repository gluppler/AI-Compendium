---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 10 - Attacking System Components"]
lead: Attacking the system component of ML-based deployments — misconfigurations, open ports, and OS-level vulnerabilities in AI infrastructure.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

The system component covers the underlying hardware, operating system, system configuration, and the specifics of how the ML model is deployed. Traditional IT security risks apply, but ML deployments introduce additional attack surface that must be assessed alongside them.

---

## Risks

Misconfigured systems are among the most common and most exploitable vulnerabilities. Security settings left at defaults, improperly scoped permissions, or accidentally exposed services give attackers easy entry. Common examples:

- Open network ports
- Weak access control lists (ACLs)
- Exposed administrative interfaces
- Default credentials

Automated scanners can identify these misconfigurations quickly, lowering the bar for exploitation.

Insecure ML model deployments compound these risks. Deploying a model without authentication, encryption, or input validation exposes it to the full range of attacks described in previous sections.

Denial-of-Service (DoS) and Distributed Denial-of-Service (DDoS) attacks exhaust system resources (CPU, RAM, network bandwidth, and disk) making the service unavailable. In ML deployments, adversaries can trigger resource exhaustion by running the model at high frequency or supplying inputs engineered to consume excessive processing power. On auto-scaling infrastructure, this also drives up operational costs. Resource exhaustion attacks also serve as a smokescreen: while defenders respond to the outage, attackers probe other components undetected.

---

## Tactics, techniques, and procedures (TTPs)

Adversaries use `vulnerability scanners` to identify outdated or misconfigured software, then exploit discovered weaknesses for unauthorized access. `Password spraying` (systematically testing common credentials against exposed interfaces such as SSH) complements automated scanning, particularly when default credentials are in use. Firewall and ACL misconfigurations identified through security testing open further paths for lateral movement or direct exploitation.

---

## Summary

- The system component covers hardware, OS, system configuration, and ML deployment specifics — traditional IT risks apply alongside ML-specific deployment attack surface.
- Misconfigurations (open ports, weak ACLs, exposed admin interfaces, default credentials) are among the most common and most easily exploited vulnerabilities, discoverable by automated scanners.
- Insecure ML model deployments lacking authentication, encryption, or input validation expose the model to the full range of attacks described across all components.
- DoS and DDoS attacks exhaust CPU, RAM, bandwidth, and disk resources; ML deployments face an additional vector — high-frequency inference requests or inputs engineered for excessive processing cost.
- On auto-scaling infrastructure, sustained resource exhaustion attacks also drive up operational costs beyond causing outages.
- Resource exhaustion attacks serve as a smokescreen — while defenders respond to the outage, attackers can probe other components undetected.

---

## Best Practices

- Harden ML deployment configurations before launch: require authentication on model serving endpoints, encrypt data in transit, and enforce input validation as baseline controls.
- Run automated vulnerability scanners and configuration audits against ML infrastructure regularly to catch open ports, exposed admin interfaces, and default credentials.
- Implement inference rate limiting and per-client request quotas to prevent resource exhaustion DoS attacks engineered for high model processing cost.
- Apply cost alerting and auto-scaling limits on cloud ML deployments to bound the financial impact of sustained high-consumption attacks.
- Treat resource exhaustion events as a potential smokescreen; maintain parallel monitoring of all component layers during a DoS incident rather than focusing response solely on availability restoration.
- Use password spraying detection and account lockout policies on all SSH and administrative interfaces exposed by ML infrastructure to counter credential-based attacks.

---

## Quiz

**Q1:** How does a DoS attack against an ML deployment differ from a traditional DoS attack?
> Beyond flooding network bandwidth or exhausting generic CPU/RAM, ML-specific DoS attacks can craft inputs engineered to consume excessive model processing power — targeting the inference computation itself. On auto-scaling infrastructure this also drives up operational costs as new instances spin up to handle the load.

**Q2:** Why do resource exhaustion attacks serve as an effective smokescreen in ML system attacks?
> While defenders focus their incident response on restoring availability, adversaries can probe other components (model, data, application) with reduced monitoring and detection pressure — using the DoS event as a distraction to conduct a more targeted secondary attack.

**Q3:** What are the baseline security controls that should be applied to any ML model deployment to prevent the most common system-component attacks?
> Require authentication on serving endpoints, encrypt all data in transit, enforce input validation, disable default credentials, close unused network ports, and restrict ACLs to minimum necessary access.

---

## Conclusion

This module surveyed the attack surface of ML-based systems across all four components (model, data, application, and system) and introduced the core attack techniques targeting each. The remainder of the AI Red Teamer path covers each area in depth, including hands-on identification and exploitation techniques.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-9-Attacking-Application-Components]] — application-layer risks that share many TTPs with the system component
- see:: [[Section-11-Skills-Assessment]] — practical exercise applying the full module's knowledge

**Terms**
- system component, misconfiguration, open ports, default credentials, DoS, DDoS, resource exhaustion, insecure ML deployment, vulnerability scanner, password spraying
