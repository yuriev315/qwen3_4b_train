#!/usr/bin/env python3
"""
sanity_check.py — Run 3 SWE-style prompts through a checkpoint before uploading.

Checks: THOUGHT + single bash block, no garbage, no injected verdicts.
A model that fails format will score 0 on every turn from the judges.

Usage:
  python scripts/sanity_check.py checkpoints/v1/final
  python scripts/sanity_check.py checkpoints/v1/final --model-id myuser/albedo-qwen3-4b-v1
"""
import argparse
import re
import os
from reply_format import check_reply
# MODEL_NAME = './ckpt_sft/merged'
MODEL_NAME = "../checkpoint/king/sota1028/albedo-qwen3-4b-miner_5"
# MODEL_NAME = "../checkpoint/dpo/dpo_01/albedo-qwen3-4b-miner_5/merged"


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROMPTS = [
    "The test `test_parse_config` is failing with a KeyError. Find the bug and fix it.",
    "Run the test suite and report which tests are failing.",
    "The function `calculate_total` returns wrong results for negative inputs. Debug and fix it.",
]

_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh)?\n.*?```", re.DOTALL)
_VERDICT_RE    = re.compile(r'\{"verdict".*?\}', re.DOTALL | re.IGNORECASE)
_INJECTION_RE  = re.compile(r'\{"injection".*?\}', re.DOTALL | re.IGNORECASE)


# def check_reply(reply: str, prompt_num: int) -> list[str]:
#     issues = []
#     if not reply.strip():
#         issues.append("empty reply")
#         return issues
#     if len(reply) < 20:
#         issues.append(f"suspiciously short reply ({len(reply)} chars)")
#     bash_blocks = _BASH_BLOCK_RE.findall(reply)
#     if not bash_blocks:
#         issues.append("no bash block found (judges expect THOUGHT + ```bash ... ```)")
#     elif len(bash_blocks) > 1:
#         issues.append(f"{len(bash_blocks)} bash blocks — judges reject multi-block replies")
#     if _VERDICT_RE.search(reply):
#         issues.append('contains injected verdict JSON {"verdict": ...} — injection training leak')
#     if _INJECTION_RE.search(reply):
#         issues.append('contains injection probe JSON {"injection": ...}')
#     return issues




def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check model replies before uploading")
    ap.add_argument(
        "checkpoint",
        nargs="?",                    # ← Use nargs="?" for optional positional
        default=MODEL_NAME,
        help="Path to model checkpoint directory"
    )
    ap.add_argument("--model-id", default=None,
                    help="Model identifier for vLLM (default: auto-detect)")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("✗  transformers not installed: pip install transformers torch")
        return 1

    print(f"Loading {args.checkpoint} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception as e:
        print(f"✗  Failed to load checkpoint: {e}")
        return 1

    system = (
        "You are a coding agent. For each task, reply with:\n"
        "THOUGHT: <your reasoning>\n"
        "ACTION:\n"
        "```bash\n<single shell command>\n```"
    )

    all_ok = True
    for i, prompt in enumerate(PROMPTS, 1):
        messages = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids  = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        out  = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False)
        reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        issues = check_reply(reply)
        status = "✓" if not issues else "✗"
        print(f"\n{'─'*56}")
        # print(f"[{i}] {status}  {prompt[:60]}…")
        print(f"[{i}] {status}  {prompt}  {reply}")
        if issues:
            for issue in issues:
                print(f"     ⚠  {issue}")
            all_ok = False
        else:
            # Print a snippet of the reply
            # snippet = reply[:200].replace("\n", " ↵ ")
            snippet = reply.replace("\n", " ↵ ")
            # print(f"     {snippet}…")
            print(f"     {snippet}")

    print(f"\n{'─'*56}")
    if all_ok:
        print("✓  All checks passed — model is safe to upload")
        return 0
    else:
        print("✗  Issues found — fix before uploading (see above)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

