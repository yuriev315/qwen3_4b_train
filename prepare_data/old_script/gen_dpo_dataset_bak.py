import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from reply_format import MIN_REPLY_TOKENS, is_valid_assistant_reply

MODEL_NAME = "./king/sota1028/albedo-qwen3-4b-miner_5_bak"
SOURCE_DIR = Path("data/albedo_json2")


def _is_valid_turn(rec: dict) -> bool:
    if not rec.get("parse_ok"):
        return False
    if rec.get("error_code"):
        return False
    king = rec.get("king_reply", "")
    chal = rec.get("chal_reply", "")
    return bool(king.strip() and chal.strip())


def _turn_group_key(rec: dict) -> str | None:
    eval_id = rec.get("eval_id")
    turn_idx = rec.get("turn_idx")
    instance_id = rec.get("instance_id")
    if eval_id is None or turn_idx is None or instance_id is None:
        return None
    return f"{eval_id}_{turn_idx}_{instance_id}"


def calculate_weight(score: float, min_weight: float = 0.8, max_weight: float = 1.5) -> float:
    """Map judge distance from 0.5 to a sample weight in [min_weight, max_weight]."""
    if 0.45 <= score <= 0.55:
        return 0.0

    distance = abs(score - 0.5)
    normalized_distance = distance / 0.5
    return min_weight + normalized_distance * (max_weight - min_weight)


def process_source(
    tokenizer,
    fout,
    min_margin=0.2,
    skip_ties=True,
    min_token_len=MIN_REPLY_TOKENS,
    tie_low=0.45,
    tie_high=0.55,
):
    """
    Process albedo_json2 for DPO pairs with per-sample weights (no duplication).
    """
    json_files = sorted(SOURCE_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} files")

    all_samples = []
    stats = {
        "total": 0,
        "kept": 0,
        "skipped_missing_key": 0,
        "skipped_margin": 0,
        "skipped_length": 0,
        "skipped_format": 0,
    }

    for json_file in tqdm(json_files, desc="Processing"):
        with open(json_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        grouped_recs = {}
        for rec in records:
            if not _is_valid_turn(rec):
                continue

            group_key = _turn_group_key(rec)
            if group_key is None:
                stats["skipped_missing_key"] += 1
                continue

            if group_key not in grouped_recs:
                grouped_recs[group_key] = {
                    "prompt": rec.get("prompt_messages") or [],
                    "king_reply": rec.get("king_reply", ""),
                    "chal_reply": rec.get("chal_reply", ""),
                    "judge_scores": [rec.get("judge_mean", 0.5)],
                }
            else:
                grouped_recs[group_key]["judge_scores"].append(rec.get("judge_mean", 0.5))

        for rec in grouped_recs.values():
            stats["total"] += 1

            avg_judge_score = sum(rec["judge_scores"]) / len(rec["judge_scores"])

            if skip_ties and tie_low <= avg_judge_score <= tie_high:
                stats["skipped_margin"] += 1
                continue

            margin = abs(avg_judge_score - 0.5) * 2
            if margin < min_margin:
                stats["skipped_margin"] += 1
                continue

            king_reply = rec["king_reply"]
            chal_reply = rec["chal_reply"]

            if avg_judge_score > 0.5:
                chosen = chal_reply
                rejected = king_reply
            else:
                chosen = king_reply
                rejected = chal_reply

            weight = calculate_weight(avg_judge_score)
            if weight <= 0:
                stats["skipped_margin"] += 1
                continue

            if not is_valid_assistant_reply(chosen):
                stats["skipped_format"] += 1
                continue

            chosen_len = len(tokenizer.encode(chosen))
            rejected_len = len(tokenizer.encode(rejected))
            if chosen_len < min_token_len or rejected_len < min_token_len:
                stats["skipped_length"] += 1
                continue

            all_samples.append(
                {
                    "prompt": rec["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "weight": weight,
                    "margin": margin,
                    "source_score": avg_judge_score,
                }
            )
            stats["kept"] += 1

    for sample in sorted(all_samples, key=lambda x: (-x["weight"], -x["margin"])):
        fout.write(
            json.dumps(
                {
                    "prompt": sample["prompt"],
                    "chosen": sample["chosen"],
                    "rejected": sample["rejected"],
                    "weight": sample["weight"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"\n{'=' * 50}")
    print("DPO DATA STATISTICS")
    print(f"{'=' * 50}")
    print(f"Unique turns processed: {stats['total']}")
    print(f"Kept: {stats['kept']}")
    print(f"Skipped (missing turn key): {stats['skipped_missing_key']}")
    print(f"Skipped (tie/margin): {stats['skipped_margin']}")
    print(f"Skipped (bad chosen format/spam): {stats['skipped_format']}")
    print(f"Skipped (length <{min_token_len}): {stats['skipped_length']}")
    print("Weight distribution:")
    if all_samples:
        weights = [s["weight"] for s in all_samples]
        print(f"  Min weight: {min(weights):.2f}")
        print(f"  Max weight: {max(weights):.2f}")
        print(f"  Mean weight: {np.mean(weights):.2f}")
    print(f"{'=' * 50}")

    return stats["kept"]


def generate_dpo_data(out_file="data/dpo_data.jsonl", min_token_len=MIN_REPLY_TOKENS):
    print(f"Loading tokenizer ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    with open(out_file, "w", encoding="utf-8") as fout:
        total_pairs = process_source(
            tokenizer,
            fout,
            min_margin=0.2,
            skip_ties=True,
            min_token_len=min_token_len,
        )

    print(f"\nDPO data saved to: {out_file}")
    print(f"Total DPO pairs: {total_pairs}")
    print("\nRemember:")
    print("  - Use WeightedDPOTrainer to apply the 'weight' column")
    print("  - Delete stale cache before training:")
    print("    Remove-Item -Recurse -Force data/dpo_data.jsonl.cache")


if __name__ == "__main__":
    generate_dpo_data("data/dpo_data.jsonl")
