#!/usr/bin/env python3
"""
sanity_check.py — Run 3 SWE-style prompts through a checkpoint before uploading.

Checks: THOUGHT + single bash block, no garbage, no injected verdicts.
A model that fails format will score 0 on every turn from the judges.

Usage:
  python scripts/sanity_check.py ckpt_sft/merged
  python scripts/sanity_check.py ckpt_sft/final --base-model ./king/sota1028/albedo-qwen3-4b-miner_5_bak
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reply_format import MIN_REPLY_TOKENS, SWE_SYSTEM_PROMPT, check_reply, check_reply_tokens

DEFAULT_BASE_MODEL = "./king/sota1028/albedo-qwen3-4b-miner_5_bak"

PROMPTS = [
    "The test `test_parse_config` is failing with a KeyError. Find the bug and fix it.",
    "Run the test suite and report which tests are failing.",
    "The function `calculate_total` returns wrong results for negative inputs. Debug and fix it.",
]


def _is_peft_checkpoint(checkpoint: Path) -> bool:
    return (checkpoint / "adapter_config.json").is_file()


def _read_base_model(checkpoint: Path) -> str:
    adapter_cfg = checkpoint / "adapter_config.json"
    if adapter_cfg.is_file():
        data = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        return data.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL


def load_model_and_tokenizer(checkpoint: Path, base_model: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if _is_peft_checkpoint(checkpoint):
        from peft import PeftModel

        resolved_base = base_model or _read_base_model(checkpoint)
        print(f"Loading PEFT adapter from {checkpoint}")
        print(f"Base model: {resolved_base}")
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        model.eval()
        return model, tokenizer

    print(f"Loading merged/full checkpoint from {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check model replies before uploading")
    ap.add_argument("checkpoint", help="Path to model checkpoint directory")
    ap.add_argument(
        "--base-model",
        default=None,
        help="Base model for PEFT adapters (default: read from adapter_config.json)",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        print(f"FAIL: checkpoint not found: {checkpoint}")
        return 1

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError:
        print("FAIL: transformers not installed (pip install transformers torch peft)")
        return 1

    print(f"Loading {checkpoint} ...")
    try:
        model, tokenizer = load_model_and_tokenizer(checkpoint, args.base_model)
    except Exception as e:
        print(f"FAIL: could not load checkpoint: {e}")
        return 1

    all_ok = True
    for i, prompt in enumerate(PROMPTS, 1):
        messages = [
            {"role": "system", "content": SWE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False)
        reply = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

        issues = check_reply(reply) + check_reply_tokens(reply, tokenizer)
        status = "PASS" if not issues else "FAIL"
        print(f"\n{'-' * 56}")
        print(f"[{i}] {status}  {prompt[:60]}...")
        if issues:
            for issue in issues:
                print(f"     WARN: {issue}")
            all_ok = False
        else:
            snippet = reply[:200].replace("\n", " | ")
            print(f"     {snippet}...")

    print(f"\n{'-' * 56}")
    if all_ok:
        print("PASS: all checks passed - model is safe to upload")
        return 0
    print("FAIL: issues found - fix before uploading (see above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
