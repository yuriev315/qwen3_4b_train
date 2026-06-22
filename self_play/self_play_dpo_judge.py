"""Judge self-play SFT samples with pairwise Chutes eval."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from albedo.config import JUDGE_MODELS
from albedo.judge import ChutesJudge
# from eval import _strip_reply_injection
# import chain_config
import httpx
# import judge as judge_mod
from collections import Counter
import math
# # Ensure PyCharm / CLI loads albedo from albedo-main (not an old pip install).
# _ALBEDO_ROOT = Path(__file__).resolve().parents[1] / "albedo-main"
# if _ALBEDO_ROOT.is_dir():
#     _root = str(_ALBEDO_ROOT)
#     if _root not in sys.path:
#         sys.path.insert(0, _root)
#
# from albedo.duel.turn import _dummy_judge, strip_reply_injection
# from albedo.judge import ChutesJudge
# from albedo.judge.verdict import parse_metric_verdict

CHUTE_TOKEN = (
    "cpk_77d1821dec4f4ccbac15a2f9d6938b4a."
    "2a993fa0225d5b30a17a9905618dd2e9.MWzeXLoIEYu4kPttiTAVcmqp59GG41cG"
)
os.environ.setdefault("CHUTES_API_KEY", CHUTE_TOKEN)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_MAX_PARALLEL_TURNS = int(os.environ.get("ALBEDO_MAX_PARALLEL_TURNS", "8"))
log = logging.getLogger(__name__)

# INPUT_FILENAME = "data/self_play/ckpt_sft_merged/20260613_100336.json"
# OUTPUT_DIR = "data/self_play/ckpt_sft_merged_judge"

# INPUT_FILENAME = "data/self_play/ckpt_sft_0_merged/20260613_102838.json"
# OUTPUT_DIR = "data/self_play/ckpt_sft_0_merged_judge"

INPUT_FILENAME = "../data/self_play/albedo-qwen3-4b-miner_5_merged/20260613_100336.json"
OUTPUT_DIR = "../data/self_play/albedo-qwen3-4b-miner_5_merged_judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)


EVALS_JUDGE_RAW_MAX_CHARS = int(os.environ.get("ALBEDO_EVALS_JUDGE_RAW_MAX_CHARS", "8192"))

def _clip_judge_raw(raw: str) -> tuple[str, bool]:
    if len(raw) <= EVALS_JUDGE_RAW_MAX_CHARS:
        return raw, False
    return raw[:EVALS_JUDGE_RAW_MAX_CHARS], True

async def _score_one_turn(
    sample: dict,
    judge_client: judge_mod.ChutesJudge,
    sem: asyncio.Semaphore,
    judge_models: tuple[str, ...],
) -> dict:
    """Run king + challenger generation once, then fan out across every
    judge in `judge_models` (two judge calls per judge — king side and
    challenger side, all in flight at once). Returns a record with one
    `per_judge` entry per judge plus ensemble aggregates."""
    king_reply = sample["king_reply"]
    chal_reply = sample["chal_reply"]
    prefix = sample["prefix"]
    async with sem:
        king_reply_clean = _strip_reply_injection(king_reply)
        chal_reply_clean = _strip_reply_injection(chal_reply)
        tasks: list[asyncio.Task] = []
        for jm in judge_models:
            tasks.append(asyncio.create_task(
                judge_client.score(prefix, king_reply_clean, model=jm)
            ))
            tasks.append(asyncio.create_task(
                judge_client.score(prefix, chal_reply_clean, model=jm)
            ))
        verdicts = await asyncio.gather(*tasks)

        per_judge: list[dict] = []
        king_sum = 0.0
        chal_sum = 0.0
        any_parse_fail = False
        parse_fail_num = 0
        parse_ok_num = 0
        for i, jm in enumerate(judge_models):
            k_v = verdicts[2 * i]
            c_v = verdicts[2 * i + 1]
            king_raw, king_raw_trunc = _clip_judge_raw(k_v.raw)
            chal_raw, chal_raw_trunc = _clip_judge_raw(c_v.raw)
            per_judge.append({
                "model": jm,
                "king_verdict": k_v.label,
                "chal_verdict": c_v.label,
                "king_score": k_v.score,
                "chal_score": c_v.score,
                "king_rationale": k_v.rationale,
                "chal_rationale": c_v.rationale,
                "king_raw": king_raw,
                "chal_raw": chal_raw,
                "king_raw_truncated": king_raw_trunc,
                "chal_raw_truncated": chal_raw_trunc,
                "parse_ok": k_v.parse_ok and c_v.parse_ok,
            })
            k_score = k_v.score if math.isfinite(k_v.score) else 0.0
            c_score = c_v.score if math.isfinite(c_v.score) else 0.0
            if not math.isfinite(k_v.score) or not math.isfinite(c_v.score):
                log.warning("judge %s returned non-finite score (king=%.4g chal=%.4g) — clamped to 0.0",
                            jm, k_v.score, c_v.score)
            # king_sum += k_score
            # chal_sum += c_score
            if not (k_v.parse_ok and c_v.parse_ok):
                any_parse_fail = True
                parse_fail_num += 1
                log.warning("judge %s parse failure on turn (king_ok=%s chal_ok=%s) — scored as reject (0.0)",
                            jm, k_v.parse_ok, c_v.parse_ok)
            else:
                parse_ok_num += 1
                king_sum += k_score
                chal_sum += c_score


        n = max(1, len(judge_models))
        # king_avg = king_sum / n
        king_avg = king_sum / parse_ok_num
        # chal_avg = chal_sum / n
        chal_avg = chal_sum / parse_ok_num
        print(f"parse_ok: {parse_ok_num},  parse_fail: {parse_fail_num}")
        return {
            "global_idx": sample["global_idx"],
            "prefix": prefix,
            "king_reply": king_reply,
            "chal_reply": chal_reply,
            "per_judge": per_judge,
            "king_score_avg": king_avg,
            "chal_score_avg": chal_avg,
            "delta_avg": chal_avg - king_avg,
            "parse_ok": not any_parse_fail,
        }


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
    judge_models = JUDGE_MODELS
    judge = ChutesJudge()


    sem = asyncio.Semaphore(2)
    king_avg_sum = 0.0
    chal_avg_sum = 0.0
    parse_failures = 0
    per_turn_ensemble_deltas: list[float] = []  # paired-bootstrap input

    # Per-judge accumulators. One `Counter` per side per judge so the
    # dashboard can show one bar per judge with its own accept/weak/reject
    # breakdown — same shape as affine.io's per-environment bars.
    per_judge_acc: dict[str, dict] = {
        jm: {
            "n": 0,
            "king_sum": 0.0,
            "chal_sum": 0.0,
            "verdicts_king": Counter(),
            "verdicts_chal": Counter(),
            "deltas": [],  # per-judge per-turn deltas
            "parse_failures": 0,
        }
        for jm in judge_models
    }
    try:
            async def runner(sample) -> None:
                nonlocal king_avg_sum, chal_avg_sum, parse_failures
                rec = await _score_one_turn(
                    sample, judge, sem, judge_models,
                )
                king_avg_sum += rec["king_score_avg"]
                chal_avg_sum += rec["chal_score_avg"]
                if not rec.get("parse_ok", True):
                    parse_failures += 1
                per_turn_ensemble_deltas.append(rec["delta_avg"])

                # # Per-judge accumulation.
                # for pj in rec["per_judge"]:
                #     acc = per_judge_acc[pj["model"]]
                #     acc["n"] += 1
                #     acc["king_sum"] += pj["king_score"]
                #     acc["chal_sum"] += pj["chal_score"]
                #     acc["verdicts_king"][pj["king_verdict"]] += 1
                #     acc["verdicts_chal"][pj["chal_verdict"]] += 1
                #     acc["deltas"].append(pj["chal_score"] - pj["king_score"])
                #     if not pj["parse_ok"]:
                #         acc["parse_failures"] += 1

                item =  {
                    "global_idx":    rec["global_idx"],
                    "king_score":    rec["king_score_avg"],
                    "chal_score":    rec["chal_score_avg"],
                    "delta":         rec["delta_avg"],
                    "parse_ok":      rec.get("parse_ok", True),
                    "per_judge":     [
                        {
                            "model":         pj["model"],
                            "king_verdict":  pj["king_verdict"],
                            "chal_verdict":  pj["chal_verdict"],
                            "king_score":    pj["king_score"],
                            "chal_score":    pj["chal_score"],
                        }
                        for pj in rec["per_judge"]
                    ],
                    "error":         rec.get("error"),
                }
                return item

            tasks = [asyncio.create_task(runner(s)) for s in dataset]

            out_dir = f"{OUTPUT_DIR}/{os.path.basename(sample_filename).split('.')[0]}"
            os.makedirs(out_dir, exist_ok=True)
            all = 0
            parse_ok_num = 0
            parse_fail_num = 0
            win_num = 0; loss_num = 0; tie_num = 0
            parse_fail_list = []
            with open(f"{out_dir}/summary.txt", 'w', encoding='utf-8') as sf:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                # for s in dataset:
                #     result = await runner(s)
                    all += 1
                    parsed_ok = result['parse_ok']
                    print(result["global_idx"], ":", result["king_score"], result["chal_score"], f"parse: {parsed_ok}")
                    if parsed_ok:
                        parse_ok_num += 1
                        delta = result['delta']
                        if delta > 0.1:
                            win_num += 1
                        elif delta < -0.1:
                            loss_num += 1
                        else:
                            tie_num += 1
                        sf.write(f"{result['global_idx']} : {result['king_score']}, {result['chal_score']},  parse: {parsed_ok}\n")
                    else:
                        parse_fail_num += 1
                        parse_fail_list.append(result['global_idx'])
                    with open(f"{out_dir}/{result['global_idx']}.json", 'w', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                sf.write(f'all: {all}, parse_ok: {parse_ok_num}, parse_fail: {parse_fail_num}\n')
                sf.write(f'win: {win_num}, loss: {loss_num}, tie: {tie_num}\n')
                sf.write(f'parse_fail_list: {", ".join([str(i) for i in parse_fail_list])}\n')

    except:
        print("Error")
        raise


if __name__ == "__main__":
    print(INPUT_FILENAME)
    print("main_run!")
    raise SystemExit(asyncio.run(main(INPUT_FILENAME)))
