import json
import hashlib
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer

from reply_format import MIN_REPLY_TOKENS, check_reply


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "./king/sota1028/albedo-qwen3-4b-miner_5"
SOURCE_DIR = Path("data/albedo_json2")


# -----------------------------
# DETERMINISTIC HELPERS
# -----------------------------
def stable_hash(obj) -> str:
    """Fully deterministic hash (stable across runs)."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def group_key(rec: dict):
    """Stable grouping key."""
    if not all(k in rec for k in ["eval_id", "turn_idx", "instance_id"]):
        return None
    return f"{rec['eval_id']}_{rec['turn_idx']}_{rec['instance_id']}"


def is_valid(rec: dict) -> bool:
    return (
        rec.get("parse_ok")
        and not rec.get("error_code")
        and rec.get("king_reply")
        and rec.get("chal_reply")
    )


# -----------------------------
# CORE SCORING (SINGLE SOURCE OF TRUTH)
# -----------------------------
def score_sample(avg_score: float):
    """
    Unified scoring model:
    - margin: confidence
    - weight: sampling importance
    """

    margin = abs(avg_score - 0.5) * 2  # 0..1 confidence

    # reject uncertain samples
    if 0.45 <= avg_score <= 0.55:
        return None, 0.0

    if margin < 0.6:
        return None, 0.0

    # stable monotonic weighting
    weight = 0.8 + margin * 0.7  # [0.8, 1.5]
    return margin, weight


# -----------------------------
# REPLY VALIDATION
# -----------------------------
def validate_reply(tokenizer, text: str, min_len: int, is_chosen=True):
    if is_chosen and check_reply(text):
        return "format_error"

    if len(tokenizer.encode(text)) < min_len:
        return "too_short"

    return None


# -----------------------------
# MAIN PROCESSOR
# -----------------------------
def process_source(tokenizer, out_file, min_len=MIN_REPLY_TOKENS):
    json_files = sorted(SOURCE_DIR.rglob("*.json"))

    stats = defaultdict(int)
    best = {}

    for path in tqdm(json_files, desc="Loading"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        grouped = defaultdict(lambda: {
            "scores": [],
            "proto": [],
            "rec": None
        })

        # -----------------------------
        # GROUPING
        # -----------------------------
        for r in records:
            if not is_valid(r):
                continue

            stats["total"] += 1

            key = group_key(r)
            if not key:
                stats["missing_key"] += 1
                continue

            g = grouped[key]
            g["rec"] = g["rec"] or r

            g["scores"].append(r.get("judge_mean", 0.5))
            g["proto"].append(r.get("metric_scores", {}).get("protocol", 0.5))

        stats["unique_turns"] += len(grouped)

        # -----------------------------
        # CONVERSION
        # -----------------------------
        for g in grouped.values():
            avg_score = sum(g["scores"]) / len(g["scores"])
            avg_proto = sum(g["proto"]) / len(g["proto"])

            margin, weight = score_sample(avg_score)
            if weight == 0.0:
                stats["rejected_score"] += 1
                continue

            avg_proto = avg_proto if avg_score>0.5 else 1-avg_proto
            if avg_proto < 0.5:
                stats["rejected_protocol"] += 1
                continue

            rec = g["rec"]
            king = rec["king_reply"]
            chal = rec["chal_reply"]

            chosen, rejected = (
                (chal, king) if avg_score > 0.5 else (king, chal)
            )

            # -----------------------------
            # VALIDATION
            # -----------------------------
            err = validate_reply(tokenizer, chosen, min_len, True)
            if err:
                stats[f"rejected_{err}"] += 1
                continue

            err = validate_reply(tokenizer, rejected, min_len, False)
            if err:
                stats[f"rejected_{err}"] += 1
                continue

            # -----------------------------
            # DEDUP KEY (STABLE)
            # -----------------------------
            dedup_key = stable_hash({
                "prompt": rec["prompt_messages"],
                "chosen": chosen,
                "rejected": rejected
            })

            existing = best.get(dedup_key)

            if (existing is None) or (weight > existing["weight"]):
                best[dedup_key] = {
                    "prompt": rec["prompt_messages"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "weight": weight,
                    "score": avg_score
                }

    # -----------------------------
    # WRITE OUTPUT (DETERMINISTIC ORDER)
    # -----------------------------
    sorted_items = sorted(
        best.values(),
        key=lambda x: (-x["weight"], x["score"])
    )

    with open(out_file, "w", encoding="utf-8") as f:
        for item in sorted_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


# -----------------------------
# ENTRYPOINT
# -----------------------------
def run(out_file="data/sft_dpo_clean.jsonl"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    stats = process_source(tokenizer, out_file)

    print("\n==============================")
    print("FINAL STATS (DETERMINISTIC)")
    print("==============================")

    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")

    print("==============================")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    run()