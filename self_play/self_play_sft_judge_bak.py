"""Judge self-play SFT samples with pairwise Chutes eval."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure PyCharm / CLI loads albedo from albedo-main (not an old pip install).
_ALBEDO_ROOT = Path(__file__).resolve().parents[1] / "albedo-main"
if _ALBEDO_ROOT.is_dir():
    _root = str(_ALBEDO_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from albedo.duel.turn import _dummy_judge, strip_reply_injection
from albedo.judge import ChutesJudge
from albedo.judge.verdict import parse_metric_verdict

CHUTE_TOKEN = (
    "cpk_77d1821dec4f4ccbac15a2f9d6938b4a."
    "2a993fa0225d5b30a17a9905618dd2e9.MWzeXLoIEYu4kPttiTAVcmqp59GG41cG"
)
os.environ.setdefault("CHUTES_API_KEY", CHUTE_TOKEN)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_MAX_PARALLEL_TURNS = int(os.environ.get("ALBEDO_MAX_PARALLEL_TURNS", "8"))
log = logging.getLogger(__name__)

INPUT_FILENAME = "data/self_play/ckpt_sft_0_merged/20260613_100336.json"
OUTPUT_DIR = "data/self_play/ckpt_sft_0_merged_judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def judge_generated_turn(sample, judge, judge_models):
    """Score a generated turn head-to-head across all judge models."""
    king_reply = sample["king_reply"]
    chal_reply = sample["chal_reply"]

    king_clean = strip_reply_injection(king_reply)
    chal_clean = strip_reply_injection(chal_reply)
    context_messages = sample["prefix"]
    judge_msgs = judge._build_pairwise_messages(context_messages, king_clean, chal_clean)

    raws = await judge.query_judges(
        judge_models,
        judge_msgs,
        accept=lambda r: parse_metric_verdict(r).parse_ok,
    )

    per_judge: list[dict] = []
    for model in judge_models:
        raw = raws.get(model)
        if raw is None:
            log.warning(
                "judge %s unresolved on turn %d (Chutes+OpenRouter)",
                model,
                sample["global_idx"],
            )
            per_judge.append(_dummy_judge(model))
            continue
        v = parse_metric_verdict(raw)
        per_judge.append({
            "judge_model": model,
            "metric_scores": v.metric_scores,
            "judge_mean": v.judge_mean,
            "parse_ok": v.parse_ok,
            "raw": raw
        })

    ok_means = [e["judge_mean"] for e in per_judge if e["parse_ok"]]
    all_parse_ok = all(e["parse_ok"] for e in per_judge)
    for e in per_judge:
        if not e["parse_ok"]:
            log.warning(
                "parse_ok=False for judge %s on global_idx=%d",
                e["judge_model"],
                sample["global_idx"],
            )

    final_score = sum(ok_means) / len(ok_means) if ok_means else 0.0

    item =  {
        "global_idx": sample["global_idx"],
        "prefix": sample["prefix"],
        "king_reply": king_reply,
        "chal_reply": chal_reply,
        "per_judge": per_judge,
        "final_score": final_score,
        "final_score_100": final_score * 100.0,
        "delta": final_score - 0.5,
        "parse_ok": all_parse_ok,
    }
    return item



async def run_dual(samples, judge, judge_models, out_dir, eval_id):
    semaphore = asyncio.Semaphore(_MAX_PARALLEL_TURNS)
    judge_errors = 0

    async def _judge_one(sample):
        async with semaphore:
            try:
                return await judge_generated_turn(
                    sample,
                    judge=judge,
                    judge_models=judge_models,
                )
            except Exception as exc:
                print(
                    "Error: judge_generated_turn failed for sample %d: %s"
                    % (sample.get("global_idx"), exc)
                )
                return None
    count = 0
    output_dir = f'{out_dir}/{eval_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for g in samples:
        result = await _judge_one(g)
        if result is None:
            judge_errors += 1
            continue
        # results.append(result)
        with open(f'{output_dir}/{result["global_idx"]}.json', "w", encoding="utf-8") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        count += 1
        print(
            "Judged sample %d: final_score=%.3f parse_ok=%s"
            % (result["global_idx"], result["final_score"], result["parse_ok"])
        )

    print(
        "Judging done: %d succeeded, %d errors"
        % (count, judge_errors)
    )
    return count


async def main(sample_filename=""):
    if not sample_filename:
        print("Error: Empty filename!")
        return 1

    dataset = []
    with open(sample_filename, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            global_idx = row["global_idx"]
            prefix = row["prefix"]
            king_reply = row["king_reply"]
            chal_reply = row["chal_reply"]
            if global_idx < 0:
                print("Error: invalid global_idx %r" % global_idx)
                continue
            if not prefix:
                print("Error: Empty Prefix!")
                continue
            if not king_reply.strip():
                print("Error: Empty King Reply!")
                continue
            if not chal_reply.strip():
                print("Error: Empty Challenger Reply!")
                continue
            dataset.append({
                "global_idx": global_idx,
                "prefix": prefix,
                "king_reply": king_reply,
                "chal_reply": chal_reply,
            })

    print("Loaded %d samples for judging" % len(dataset))
    from albedo.config import JUDGE_MODELS
    # import albedo.judge.client as _judge_client

    judge = ChutesJudge()
    # out_path = os.path.join(OUTPUT_DIR, os.path.basename(sample_filename))
    result = await run_dual(dataset, judge, JUDGE_MODELS, OUTPUT_DIR, os.path.basename(sample_filename).split('.')[0])
    # print(result)
    print("Writing %d judged sample" % result)
    return 0


if __name__ == "__main__":
    print(INPUT_FILENAME)
    print("main_run!")
    raise SystemExit(asyncio.run(main(INPUT_FILENAME)))
