import json
import hashlib
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from reply_format import MIN_REPLY_TOKENS, check_reply

MODEL_NAME = "./king/sota1028/albedo-qwen3-4b-miner_5"
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


def format_as_chat(tokenizer, msg):
    return tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=False,
    )


def process_source(
    tokenizer,
    fout,
    min_score=0.65,
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

    best_rows = {}
    stats = {
        "total": 0,
        "unique_turns": 0,
        "kept": 0,
        "skipped_missing_key": 0,
        "skipped_tie": 0,
        "skipped_score": 0,
        "skipped_length": 0,
        "skipped_format": 0,
        "skipped_protocol": 0
    }

    for json_file in tqdm(json_files, desc="Processing source 2"):
        with open(json_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        grouped_recs = {}
        for rec in records:
            if not _is_valid_turn(rec):
                continue

            stats["total"] += 1
            group_key = _turn_group_key(rec)
            if group_key is None:
                stats["skipped_missing_key"] += 1
                continue

            if group_key not in grouped_recs:
                grouped_recs[group_key] = {
                    "prompt": rec.get("prompt_messages") or [],
                    "king_reply": rec.get("king_reply", ""),
                    "chal_reply": rec.get("chal_reply", ""),
                    "scores": [rec.get("judge_mean", 0.5)],
                    "proto_scores": [rec.get("metric_scores", {}).get("protocol", 0.5)],
                    "eval_id": rec["eval_id"],
                }
            else:
                grouped_recs[group_key]["scores"].append(rec.get("judge_mean", 0.5))
                grouped_recs[group_key]["proto_scores"].append(rec.get("metric_scores", {}).get("protocol", 0.5))

        stats["unique_turns"] += len(grouped_recs)

        for conv in grouped_recs.values():
            avg_score = sum(conv["scores"]) / len(conv["scores"])
            avg_proto_score = sum(conv["proto_scores"]) / len(conv["proto_scores"])

            if skip_ties and tie_low <= avg_score <= tie_high:
                stats["skipped_tie"] += 1
                continue

            if avg_score > 0.5:
                reply = conv["chal_reply"]
                winner_score = avg_score
                winner_proto_score = avg_proto_score
            else:
                reply = conv["king_reply"]
                winner_score = 1 - avg_score
                winner_proto_score = 1 - avg_proto_score

            if winner_score < min_score:
                stats["skipped_score"] += 1
                continue

            if winner_proto_score < 0.5:
                stats["skipped_protocol"] += 1
                continue

            if check_reply(reply):
                # print(check_reply(reply))
                stats["skipped_format"] += 1
                continue

            if len(tokenizer.encode(reply)) < min_length:
                stats["skipped_length"] += 1
                continue

            messages = conv["prompt"] + [{"role": "assistant", "content": reply}]
            # print(reply[:100])
            # print('-'*200)
            conversation_text = json.dumps(messages)
            # conversation_text = format_as_chat(tokenizer, messages)

            prompt_hash = hashlib.sha256(
                json.dumps(conv["prompt"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

            old = best_rows.get(prompt_hash)
            if old is None or winner_score > old["score"]:
                best_rows[prompt_hash] = {
                    "score": winner_score,
                    "text": conversation_text,
                    "eval_id": conv["eval_id"],
                }

    for item in sorted(best_rows.values(), key=lambda x: (-x["score"], x["eval_id"])):
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        stats["kept"] += 1

    print(f"\n{'='*50}")
    print("SFT DATA STATISTICS")
    print(f"{'='*50}")
    print(f"Total valid records: {stats['total']}")
    print(f"Unique turns processed: {stats['unique_turns']}")
    print(f"Unique prompts kept: {len(best_rows)}")
    print(f"Kept (score >={min_score}, length >={min_length}): {stats['kept']}")
    print(f"Skipped (missing turn key): {stats['skipped_missing_key']}")
    print(f"Skipped (tie {tie_low}-{tie_high}): {stats['skipped_tie']}")
    print(f"Skipped (low score): {stats['skipped_score']}")
    print(f"Skipped (bad format/spam): {stats['skipped_format']}")
    print(f"Skipped (too short): {stats['skipped_length']}")
    print(f"Skipped (protocol unmach): {stats['skipped_protocol']}")
    print(f"{'='*50}")

    return stats["kept"]


def generate_sft_data(out_file="data/sft_data_raw.jsonl"):
    print(f"Loading tokenizer ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    with open(out_file, "w", encoding="utf-8") as fout:
        total = process_source(
            tokenizer,
            fout,
            min_score=0.65,
            min_length=MIN_REPLY_TOKENS,
            skip_ties=True,
        )

    print(f"\nSFT data saved to: {out_file}")
    print(f"   Total samples: {total}")
    print("\nRemember to delete old cache before training:")
    print("   Remove-Item -Recurse -Force data/sft_tokenized")


if __name__ == "__main__":
    generate_sft_data("data/sft_data_raw.jsonl")
"""
==================================================
SFT DATA STATISTICS
==================================================
Total valid records: 30164
Unique turns processed: 10298
Unique prompts kept: 2468
Kept (score >=0.65, length >=20): 2468
Skipped (missing turn key): 0
Skipped (tie 0.45-0.55): 3375
Skipped (low score): 1927
Skipped (bad format/spam): 2528
Skipped (too short): 0
Skipped (protocol unmach): 729
==================================================

SFT data saved to: data/sft_data_raw.jsonl
   Total samples: 2468



SFT DATA STATISTICS
==================================================
Total valid records: 30164
Unique turns processed: 10298
Unique prompts kept: 3609
Kept (score >=0.65, length >=20): 3609
Skipped (missing turn key): 0
Skipped (tie 0.45-0.55): 3375
Skipped (low score): 1927
Skipped (bad format/spam): 655
Skipped (too short): 0
Skipped (protocol unmach): 729
==================================================

SFT data saved to: data/sft_data_raw.jsonl
   Total samples: 3609

Remember to delete old cache before training:
   Remove-Item -Recurse -Force data/sft_tokenized



"""