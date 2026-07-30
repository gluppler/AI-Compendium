# 9 — Esoteric Command Injection

Techniques from [nitewl.com — Esoteric Command Injection](https://www.nitewl.com/esoteric-command-injection/) (Feb 2026).

Bash process substitution, ANSI-C quoting, shell parameter expansion, and new KornShell-style command substitution in Bash 5.3+ as command injection vectors.

## Files

| File | Focus |
|------|-------|
| `esoteric-command-injection.md` | Full technique distillation |

## Key Techniques for Wiki

1. **Sauron Eye `<()>`** — Process substitution for command injection and information disclosure
2. **Confused Face `{$\\='}`** — Bash 5.3+ ksh-style command substitution with ANSI-C encoding
3. **ANSI-C Quoting** — `$'...\x41...'` for encoding denied characters (octal/hex/unicode)
4. **Shell Parameter Expansion** — `${VAR:=$'ANSI-C'}` to assign and expand encoded payloads
5. **Here Strings** — `<<<` standard input poisoning with shell expansion
6. **Unset environment variable poisoning** — overwriting unset vars via parameter expansion

## Cross-References

- **Attacking-AI-Application-and-System/** — Shell injection in AI system components
- **LLM-Output-Attacks/** — Code injection via LLM output paths
- **Challenges/** — Applicable to any challenge with command execution surfaces
