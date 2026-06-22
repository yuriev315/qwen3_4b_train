import re
import json
from pathlib import Path
import zipfile
from transformers import AutoTokenizer


MODEL_NAME = "../checkpoint/king/arboshelper/albedo-qwen3-4b-2-5-final"
print(f"Loading tokenizer ({MODEL_NAME}) ...")
_CHAT_TURN_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|<\|redacted_im_end\|>)",
    re.DOTALL,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def _turn_group_key(rec: dict) -> str | None:
    eval_id = rec.get("eval_id")
    turn_idx = rec.get("turn_idx")
    instance_id = rec.get("instance_id")
    if eval_id is None or turn_idx is None or instance_id is None:
        return None
    return f"{eval_id}_{turn_idx}_{instance_id}"

def to_json():
    SOURCE_DIR = Path("../data/albedo_zip")
    OUTPUT_DIR = Path("../data/albedo_json")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    zip_files = list(SOURCE_DIR.rglob("*.zip"))
    print(f"Found {len(zip_files)} files")
    stats = {
        "total": 0,
        "unique_turns": 0,
        "all_files": 0,
        "skipped_missing_key": 0,
        "skipped_exists": 0,
        "skipped_zero_sample": 0,
    }
    for zip_file in zip_files:
        relative = zip_file.relative_to(SOURCE_DIR)
        json_path = OUTPUT_DIR / relative.with_suffix("").with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # if json_path.exists():
        #     print(f"{json_path} already exists, skipping!")
        #     stats["skipped_exists"] += 1
        #     continue
        print(f"Processing {zip_file.name}")
        records = {}
        try:
            with zipfile.ZipFile(zip_file, "r") as z:
                if "judge_raw.jsonl" in z.namelist():
                    with z.open("judge_raw.jsonl") as f:
                        for line in f:
                            line = line.decode('utf-8').strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                sample_id = _turn_group_key(rec)
                                if sample_id is None:
                                    stats["skipped_missing_key"] += 1
                                    continue
                                mean_score = {rec.get("judge_model"): rec.get("judge_mean", 0.5)}
                                metric_score = {rec.get("judge_model"): rec.get("metric_scores", {})}
                                parse_ok = rec.get("parse_ok", False)
                                if sample_id not in records:
                                    records[sample_id] = {
                                        # "prompt": tokenizer.apply_chat_template(rec.get("prompt_messages") or [], tokenize=False, add_generation_prompt=False),
                                        "prompt": rec.get("prompt_messages") or [],
                                        "king_reply": rec.get("king_reply", ""),
                                        "chal_reply": rec.get("chal_reply", ""),
                                        "mean_scores": mean_score,
                                        "metric_scores": metric_score,
                                        "parse_ok": [parse_ok]
                                    }
                                else:
                                    records[sample_id]["mean_scores"].update(mean_score)
                                    records[sample_id]["metric_scores"].update(metric_score)
                                    records[sample_id]["parse_ok"].append(parse_ok)
                            except json.JSONDecodeError:
                                print(f"Bad JSON line in {zip_file}")
                                continue
                        stats["unique_turns"] += len(records)
                elif "scoring-results.jsonl" in z.namelist() and "generated-samples.jsonl" in z.namelist():
                    with z.open("scoring-results.jsonl") as f_score:
                        for line in f_score:
                            line = json.loads(line)
                            sample_id = line["sample_id"]
                            if sample_id is None:
                                raise "Empty sample id"
                            judge_results = line['judge_results']
                            order = line['order']
                            if order == ["challenger", "previous_king"]:
                                mean_scores = {jud_res["judge_model"]: 1 - jud_res["judge_mean"] for jud_res in judge_results}
                                metric_scores = {jud_res["judge_model"]: {k: (1-v) for k, v in jud_res["metric_scores"].items()} for jud_res in
                                                 judge_results}
                            elif order == ["previous_king", "challenger"]:
                                mean_scores = {jud_res["judge_model"]: jud_res["judge_mean"] for jud_res in judge_results}
                                metric_scores = {jud_res["judge_model"]: jud_res["metric_scores"] for jud_res in judge_results}
                            else:
                                raise "Incorrect Order"

                            if sample_id not in records:
                                records[sample_id] = {
                                    "mean_scores": mean_scores,
                                    "metric_scores": metric_scores,
                                    "parse_ok": [jud_res["parse_ok"] for jud_res in judge_results]
                                }
                            else:
                                raise "Duplicated sample id"
                    with z.open("generated-samples.jsonl") as f_sample:
                        for line in f_sample:
                            rec = json.loads(line)
                            sample_id = rec["sample_id"]
                            if sample_id not in records:
                                raise "Sample Without score"
                            else:
                                prompt = [
                                    {"role": role, "content": content.strip()}
                                    for role, content in _CHAT_TURN_RE.findall(rec.get("prompt"))
                                ]
                                records[sample_id].update({
                                    # "prompt": rec.get("prompt") or [],
                                    "prompt": prompt,
                                    "king_reply": rec.get("previous_king_output", ""),
                                    "chal_reply": rec.get("challenger_output", ""),
                                })
        except Exception as e:
            print(zip_file)
            raise e

        with open(json_path, "w", encoding="utf-8") as out:
            json.dump(records, out, ensure_ascii=False)
        stats["all_files"] += 1
        stats["total"] += len(records)
        if not len(records):
            stats["skipped_zero_sample"] += 1
        print(f"{len(records)} samples were saved in {json_path}")

    print(f"Transformation into json finished!")
    print(f"processed {stats['all_files']} files")
    print(f"skipped existing {stats['skipped_exists']} files")
    print(f"skipped {stats['skipped_zero_sample']} zero sample files")
    print(f"processed {stats['total']} samples")

if __name__ == '__main__':
    to_json()