import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from reply_format import MIN_REPLY_TOKENS, check_reply

MODEL_NAME = "../checkpoint/king/arboshelper/albedo-qwen3-4b-2-5-final"
SOURCE_DIR = Path("../data/albedo_json")


def _is_valid_turn(rec: dict) -> bool:
    if not all(rec.get("parse_ok")):
        return False
    king = rec.get("king_reply", "")
    chal = rec.get("chal_reply", "")
    return bool(king.strip() or chal.strip())


def format_as_chat(tokenizer, msg):
    return tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=False,
    )


def compute_weight(score):
    confidence = abs(score - 0.5) * 2
    return max(0.05, min(1.0, confidence))

def _prompt_completion_token_length(
    tokenizer,
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
) -> int:
    rendered = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))

def calculate_weight(score: float, min_weight: float = 0.6, max_weight: float = 1.3) -> float:
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
    min_length=MIN_REPLY_TOKENS,
    skip_ties=True,
    tie_low=0.45,
    tie_high=0.55,
):
    """
    Process albedo_json2 source for SFT data.
    Each valid sample appears exactly once (best reply per prompt).
    """
    json_files = sorted(SOURCE_DIR.rglob("*.json"))
    print(f"Source 2: Found {len(json_files)} files")

    stats = {
        "total": 0,
        "kept": 0,
        "invalid": 0,
        "skipped_tie": 0,
        "skipped_margin": 0,
        "skipped_length": 0,
        "skipped_format": 0,
        "skipped_protocol": 0
    }

    for json_file in tqdm(json_files, desc="Processing source 2"):
        with open(json_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        for sample_id, rec in records.items():
            if not _is_valid_turn(rec):
                stats["invalid"] += 1
                continue
            stats["total"] += 1
            all_avg_score = sum(rec["mean_scores"].values()) / len(rec["mean_scores"])
            metric_scores = rec["metric_scores"]
            avg_metric_over_judges = {"correctness":0, "efficiency": 0, "grounding": 0, "progress": 0, "protocol": 0}
            avg_judge_over_metrics = {}
            for jm, mets in metric_scores.items():
                avg_judge_over_metrics[jm] = sum(mets.values()) / len(mets)
                for met, met_val in mets.items():
                    avg_metric_over_judges[met] += met_val
            avg_metric_over_judges = {k: v / len(metric_scores) for k, v in avg_metric_over_judges.items()}

            if skip_ties and tie_low <= all_avg_score <= tie_high:
                stats["skipped_tie"] += 1
                continue

            margin = abs(all_avg_score - 0.5) * 2
            if margin < min_margin:
                stats["skipped_margin"] += 1
                continue

            king_reply = rec["king_reply"]
            chal_reply = rec["chal_reply"]
            if all_avg_score > 0.5:
                chosen = chal_reply
                rejected = king_reply
                winner_score = all_avg_score
                winner_avg_judge_over_metrics = avg_judge_over_metrics
                winner_avg_metric_over_judges = avg_metric_over_judges
            else:
                chosen = king_reply
                rejected = chal_reply
                winner_score = 1 - all_avg_score
                winner_avg_judge_over_metrics = {jm: 1-met for jm, met in avg_judge_over_metrics.items()}
                winner_avg_metric_over_judges = {met: 1-val for met, val in avg_metric_over_judges.items()}

            avg_judge_flag = any([jm_score < 0.5 for jm_score in winner_avg_judge_over_metrics.values()])
            avg_metric_flag = any([met_val < 0.5 for met_val in winner_avg_metric_over_judges.values()])
            if avg_judge_flag or avg_metric_flag:
                stats["skipped_protocol"] += 1
                continue

            weight = calculate_weight(winner_score)
            if weight <= 0:
                stats["skipped_margin"] += 1
                continue

            if check_reply(chosen):
                print(check_reply(chosen))
                stats["skipped_format"] += 1
                continue
            chosen_len = len(tokenizer.encode(chosen))
            rejected_len = len(tokenizer.encode(rejected))
            if chosen_len < min_length or rejected_len < min_length:
                stats["skipped_length"] += 1
                continue
            item = {
                "prompt": rec["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "weight": weight,
                "sample_id": sample_id
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print(f"\n{'='*50}")
    print("DPO DATA STATISTICS")
    print(f"{'='*50}")
    print(f"Total valid records: {stats['total']}")
    print(f"Skipped Invalid: {stats['invalid']}")
    print(f"Kept (margin >={min_margin}, length >={min_length}): {stats['kept']}")
    print(f"Skipped (tie {tie_low}-{tie_high}): {stats['skipped_tie']}")
    print(f"Skipped (low margin): {stats['skipped_margin']}")
    print(f"Skipped (bad format/spam): {stats['skipped_format']}")
    print(f"Skipped (too short): {stats['skipped_length']}")
    print(f"Skipped (protocol unmach): {stats['skipped_protocol']}")
    print(f"{'='*50}")

    return stats["kept"]


def generate_dpo_data(out_file="../data/dpo_data.jsonl"):
    print(f"Loading tokenizer ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    with open(out_file, "w", encoding="utf-8") as fout:
        total = process_source(
            tokenizer,
            fout,
            min_margin=0.2,
            min_length=MIN_REPLY_TOKENS,
            skip_ties=True,
        )

    print(f"\nSFT data saved to: {out_file}")
    print(f"   Total samples: {total}")
    print("\nRemember to delete old cache before training:")
    print("   Remove-Item -Recurse -Force data/sft_tokenized")


if __name__ == "__main__":
    generate_dpo_data("../data/dpo_data.jsonl")


"""
==================================================
DPO DATA STATISTICS
==================================================
Total valid records: 26035
Skipped Invalid: 653
Kept (margin >=0.2, length >=20): 10547
Skipped (tie 0.45-0.55): 5775
Skipped (low margin): 2966
Skipped (bad format/spam): 938
Skipped (too short): 2
Skipped (protocol unmach): 5807
==================================================
"""