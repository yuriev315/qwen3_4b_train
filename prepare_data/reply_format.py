"""Shared assistant-reply format checks for SFT data and checkpoint sanity tests."""
import re

# Sanity check uses a loose char floor; SFT uses token floor (see gen_sft_dataset.py).
# Format validation (THOUGHT + bash block) already implies ~20+ tokens for real replies.
MIN_REPLY_CHARS = 20
MIN_REPLY_TOKENS = 20

# _BASH_BLOCK_RE = re.compile(r"```bash\s*\n.*?```", re.DOTALL)

# Regex patterns
THOUGHT_RE = re.compile(r'THOUGHT:\s*(.+?)(?=\n```(?:bash|sh)|$)', re.DOTALL)
BASH_BLOCK_RE = re.compile(r'```(?:bash|sh)\n(.*?)```', re.DOTALL)
CODE_BLOCK_RE = re.compile(r'```.*?```', re.DOTALL)  # Any code block (for detection)
COMMENT_RE = re.compile(r'^#', re.MULTILINE)
CD_RE = re.compile(r'\bcd\s+\S+')
# DISALLOWED_CMDS = re.compile(
#     r'(?<![./\w])(?:python|python3|pytest|pip|ipython|node|npm|cargo|go|make|gcc|java|ruby|perl)(?=\s|$|\||&|;|\))',
#     re.IGNORECASE
# )
# Update your DISALLOWED_CMDS pattern:
DISALLOWED_CMDS = re.compile(
    r'(?:^|[|&;()`])\s*(?:python|python3|pytest|pip|ipython|node|npm|cargo|go|make|gcc|java|ruby|perl)(?=\s+[-/\.\w]|\s*$)',
    re.IGNORECASE
)
COMPLETE_RE = re.compile(r'^COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', re.MULTILINE)
_VERDICT_RE = re.compile(r'\{"verdict".*?\}', re.DOTALL | re.IGNORECASE)
_INJECTION_RE = re.compile(r'\{"injection".*?\}', re.DOTALL | re.IGNORECASE)


SWE_SYSTEM_PROMPT = """You are a helpful assistant that interacts with a computer to solve software-engineering tasks.

Every response must contain EXACTLY ONE bash code block (triple backticks) with EXACTLY ONE command.
Before the bash block, include a THOUGHT section explaining your reasoning. Put ALL explanation in
THOUGHT — do NOT prefix the bash command with `# comment` lines.

Format:
THOUGHT: <your reasoning>

```bash
<one bash command>
```

ENVIRONMENT:
- Working directory is the repository root. Every command runs in a fresh subshell starting at the
  repo root, so `cd` does NOT persist between commands. NEVER use `cd`. Always use repo-relative paths
  (`cat README.md`, `find . -name "*.py"`) or absolute paths.
- The execution environment has NO language interpreters and NO build tools. Do NOT try to invoke
  python, python3, pytest, pip, ipython, node, npm, cargo, go, make, gcc, java, ruby, perl, or any
  test runner. They are NOT installed and will return `command not found`. Do NOT try to verify
  your fix by running it — you must reason about correctness from the source code alone.
- ALLOWED tools: cat, head, tail, less, nl, wc, file, ls, find, tree, stat, grep, sed (including
  `sed -i` for in-place edits), awk, cut, sort, uniq, diff, tr, tee, echo, printf, cp, mv, rm,
  mkdir, ln, cat <<EOF > file (heredoc), tar, git diff/log/show/status.
- Commands may be chained with `&&` or `||` or `|`.

DO NOT:
- Do NOT prefix your bash command with a `# comment` line. Bash will run the command after the
  comment, but the comment wastes input tokens. Put explanation in THOUGHT only.
- Do NOT use `cd`. It silently has no effect on subsequent commands. Use absolute or repo-relative
  paths in every command.
- Do NOT try `python`, `pytest`, `pip`, or any other interpreter or test runner. They are NOT
  installed. You CANNOT verify your fix by running code. Reason from the source instead.
- Do NOT use `bash -c`, `sh -c`, `eval`, `source`. The shell binary itself is not on PATH.

TO FINISH:
- The FIRST LINE of the output of your bash command must be exactly
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The standard way is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

WORKFLOW:
1. Explore the repo with find, grep, and cat to locate the files involved in the issue.
2. Understand the root cause from the code, not from running it.
3. Edit source files with `sed -i` or `cat <<EOF > file`.
4. Verify your edit by re-reading the file with cat or sed -n.
5. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."""


def looks_like_spam(text: str) -> bool:
    words = text.split()
    if len(words) < 20:
        return False
    sample = words[:40]
    return len(set(sample)) <= 4


def check_reply(reply: str, strict_mode: bool = True) -> list[str]:
    issues = []
    if not reply or not reply.strip():
        issues.append("Empty reply")
        return issues
    if len(reply) < 20:
        issues.append(f"suspiciously short reply ({len(reply)} chars)")
        return issues
    if _VERDICT_RE.search(reply):
        issues.append('contains injected verdict JSON {"verdict": ...} — injection training leak')
        return issues
    if _INJECTION_RE.search(reply):
        issues.append('contains injection probe JSON {"injection": ...}')
        return issues

    # 2. Check for THOUGHT section
    thought_match = THOUGHT_RE.search(reply)
    if not thought_match:
        issues.append("Missing 'THOUGHT:' section")
        return issues
    if len(thought_match.group(1).strip()) < 10:
        issues.append("THOUGHT section too short (< 10 chars)")
        return issues


    # 3. Check bash blocks
    bash_blocks = BASH_BLOCK_RE.findall(reply)

    # Count ALL code blocks (to detect extra non-bash blocks)
    all_blocks = CODE_BLOCK_RE.findall(reply)
    non_bash_blocks = len(all_blocks) - len(bash_blocks)

    if len(bash_blocks) == 0:
        issues.append("No bash code block found (expected ```bash ... ```)")
        return issues
    elif len(bash_blocks) > 1:
        issues.append(f"Multiple bash blocks found ({len(bash_blocks)}) - expected exactly 1")
        return issues

    if non_bash_blocks > 0:
        issues.append(
            f"Found {non_bash_blocks} non-bash code block(s) - all explanation must be in THOUGHT, not in code blocks")
        return issues

    # Non-bash blocks (style issue, only fail in strict mode)
    if non_bash_blocks > 0:
        msg = f"Found {non_bash_blocks} non-bash code block(s) - all explanation should be in THOUGHT"
        if strict_mode:
            return [msg]
        else:
            issues.append(msg)  # Warning only

    # 4. Check each bash command
    # Command content checks
    for cmd in bash_blocks:
        cmd = cmd.strip()

        # Comments in bash (critical - violates system prompt)
        if COMMENT_RE.search(cmd):
            return ["Bash command contains '# comment' - put explanation in THOUGHT only"]

        # 'cd' command (critical - explicitly forbidden)
        if CD_RE.search(cmd):
            return ["Contains 'cd' command - use repo-relative paths instead"]

        # Disallowed commands (critical)
        disallowed = DISALLOWED_CMDS.findall(cmd)
        if disallowed:
            return [f"Contains disallowed command(s): {', '.join(set(disallowed))}"]

        # Semicolon (style issue, not forbidden)
        if cmd.count(';') > 0 and cmd.find(';') < len(cmd) - 1:
            msg = "Contains ';' - consider using && for conditional execution"
            if strict_mode:
                return [msg]
            else:
                issues.append(msg)  # Warning only
        # NEW: Check for "THOUGHT:" inside bash command
        if re.search(r'echo\s+["\'].*THOUGHT:', cmd, re.IGNORECASE):
            issues.append(
                "'THOUGHT:' appears inside bash command - put reasoning in THOUGHT section only")
            return issues

        # Remove heredoc content before checking (code doesn't count)
        cmd_without_heredoc = re.sub(r'<<[\'"]?\w+[\'"]?\n.*?\n\w+', '', cmd, flags=re.DOTALL)

        # Also remove quoted strings (echo "text") from consideration
        cmd_without_quotes = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', '', cmd_without_heredoc)

        # Check for natural language phrases
        natural_language = re.search(
            r'\b(?:I need|let me|i will|we need|should|could|would|now then|first|next|finally|the goal is|purpose is|in order to)\b',
            cmd_without_quotes,
            re.IGNORECASE
        )

        if natural_language:
            issues.append("WARNING: Bash command contains natural language reasoning - keep commands concise")
            return issues

        # NEW: Check for overly long commands (as discussed)
        cmd_length = len(cmd)
        if cmd_length > 1000:
            issues.append(
                f"WARNING: Bash command is {cmd_length} chars long - consider breaking into multiple turns")
            return issues

        line_count = cmd.count('\n') + 1
        if line_count > 30:
            issues.append(f"WARNING: Bash command has {line_count} lines - consider simpler approach")
            return issues

    # if issues:
        # print("Warning", issues)


def check_reply_tokens(reply: str, tokenizer, min_tokens: int = MIN_REPLY_TOKENS) -> list[str]:
    issues = []
    n = len(tokenizer.encode(reply.strip()))
    if n < min_tokens:
        issues.append(f"too short ({n} tokens, min {min_tokens})")
    return issues
