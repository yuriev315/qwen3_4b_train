#!/usr/bin/env python3
"""
inspect_dataset.py — Validate a traces.jsonl dataset before spending GPU hours.

Usage:
  python scripts/inspect_dataset.py data/traces.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sft_data.jsonl")
    if not path.exists():
        print(f"✗  File not found: {path}")
        return 1

    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not records:
        print("✗  Empty dataset.")
        return 1
    tokenizer = AutoTokenizer.from_pretrained("./king/sota1028/albedo-qwen3-4b-miner_5")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    lengths  = [len(tokenizer.encode(r.get("text", ""))) for r in records]
    deltas   = [r.get("score", 0.0) for r in records]
    evals    = Counter(r.get("eval_id", "?") for r in records)
    has_text = sum(1 for r in records if r.get("text"))

    print(f"{'─'*52}")
    print(f"Examples:      {len(records)}")
    print(f"With text:     {has_text}  ({has_text*100//len(records)}%)")
    print(f"Unique evals:  {len(evals)}")
    print(f"Top evals:     {evals.most_common(3)}")
    print()
    print(f"Text length (tokens):")
    print(f"  min    {min(lengths)}")
    print(f"  median {sorted(lengths)[len(lengths)//2]}")
    print(f"  mean   {sum(lengths)//len(lengths)}")
    print(f"  75% {sorted(lengths)[int(len(lengths) * 0.75)]}")
    print(f"  95% {sorted(lengths)[int(len(lengths) * 0.95)]}")
    print(f"  max    {max(lengths)}")
    print()
    print(f"delta_avg distribution:")
    buckets = [0.0, 0.5, 0.65, 0.8, 0.9, 1.01]
    for lo, hi in zip(buckets, buckets[1:]):
        n   = sum(1 for d in deltas if lo <= d < hi)
        bar = "█" * (n * 30 // max(len(deltas), 1))
        print(f"  [{lo:.1f}–{hi:.1f})  {n:4d}  {bar}")

    print()
    issues = []
    if len(records) < 200:
        issues.append(f"⚠  Only {len(records)} examples — consider lowering --min-delta")
    p95 = sorted(lengths)[int(len(lengths)*0.95)]
    if p95 > 8192 * 4:
        issues.append(f"⚠  95th-pct length {p95} chars may exceed 8192 tokens — some examples will be dropped")
    top_eval_pct = evals.most_common(1)[0][1] / len(records) * 100
    if top_eval_pct > 40:
        issues.append(f"⚠  Top eval accounts for {top_eval_pct:.0f}% of data — risk of overfitting to one duel")
    if not has_text:
        issues.append("✗  No 'text' field found — re-run collect_traces.py without --raw")

    if issues:
        for i in issues: print(i)
    else:
        print("✓  Dataset looks healthy")

    return 0 if len(records) >= 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())


