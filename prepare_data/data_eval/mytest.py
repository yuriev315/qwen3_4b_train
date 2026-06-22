import asyncio
import os
import chain_config
import shutil
import logging
import httpx
import random
import hashlib
from model_store import (
    ModelRef,
    materialize_model,
)
import json

logging.basicConfig(
    level=os.environ.get("ALBEDO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("albedo.validator")


SEED_REPO   = os.environ.get("ALBEDO_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("ALBEDO_SEED_DIGEST", chain_config.SEED_DIGEST)
DASHBOARD_URL = os.environ.get(
    "ALBEDO_DASHBOARD_URL",
    "https://us-east-1.hippius.com/albedo/dashboard.json",
)
EVAL_SERVER_URL = os.environ.get("ALBEDO_EVAL_SERVER", "http://127.0.0.1:9000")
TICK_RESTART_AFTER = int(os.environ.get("ALBEDO_TICK_RESTART_AFTER", "2400"))

# king_repo = SEED_REPO
# king_digest = SEED_DIGEST
# chal_repo ='iron/albedo-qwen3-4b-iron'
# chal_digest ='sha256:8862719deafbdf0e0910f229f971c48bce242bffefb14f91309069d6df392b03'


# king_repo = SEED_REPO
# king_digest = SEED_DIGEST
# chal_repo ='iron/albedo-qwen3-4b-iron'
# chal_digest ='sha256:8862719deafbdf0e0910f229f971c48bce242bffefb14f91309069d6df392b03'

async def process_challenge(http, state):
    cid = state['challenge_id']
    king = {
        "repo": state["king_repo"],
        "digest": state["king_digest"],
    }
    challenger = {
        "repo": state["challenger_repo"],
        "digest": state["challenger_digest"]
    }
    req_body = {
        "eval_id": cid,
        "seed_hex": hashlib.blake2b(random.randint(0, 100).to_bytes(), digest_size=32).digest().hex(),
        "hotkey": "",
        "king": king,
        "challenger": challenger,
        "n_samples": chain_config.DUEL_N_SAMPLES,
        "max_turns": chain_config.DUEL_MAX_TURNS_PER_SAMPLE,
    }
    try:
        async with http.stream("POST", f"{EVAL_SERVER_URL}/eval", json=req_body,
                               timeout=httpx.Timeout(None, connect=30.0)) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                log.error("%s: eval server %s: %s", cid, resp.status_code, err[:300])
                _detail = f"{resp.status_code}: {err[:300].decode(errors='ignore')}"
                return
            cur_event = ""
            async for raw_line in resp.aiter_lines():
                print(raw_line)
                if raw_line.startswith("event:"):
                    cur_event = raw_line.split(":", 1)[1].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line.split(":", 1)[1].strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except Exception:
                    continue
                if cur_event == "phase":
                    state["phase"] = data.get("phase", state["phase"])
                    if "n_turns_total" in data:
                        state["n_total"] = data["n_turns_total"]
                elif cur_event == "progress":
                    state.update({
                        "phase": "duel",
                        "n_done": data.get("n_done", 0),
                        "n_total": data.get("n_total", state["n_total"]),
                        "king_mean": data.get("king_mean", 0.0),
                        "chal_mean": data.get("chal_mean", 0.0),
                        "mean_delta": data.get("mean_delta", 0.0),
                        "verdicts_king": data.get("verdicts_king", {}),
                        "verdicts_chal": data.get("verdicts_chal", {}),
                        "parse_failures": data.get("parse_failures", 0),
                        "last": data.get("last"),
                    })
                elif cur_event == "heartbeat":
                    print(cur_event)
                elif cur_event == "verdict":
                    verdict = data
                    break
    except Exception as e:
        print("e:", e)

async def _eval_set_king(http: httpx.AsyncClient, king_repo, king_digest):
    r = await http.post(
        f"{EVAL_SERVER_URL}/set_king",
        json={"king": {"repo": king_repo, "digest": king_digest}},
        timeout=600.0,
    )
    r.raise_for_status()
    log.info("eval /set_king ok: %s", r.json().get("king"))

async def main():
    king_ref = ModelRef(king_repo, king_digest)
    king_dir = os.path.abspath(f"./miner/{king_repo}")
    log.info("downloading king from %s", king_ref.immutable_ref)
    materialize_model(king_ref, local_dir=king_dir, max_workers=16)

    # dashboard = None
    # try:
    #     resp = httpx.get(DASHBOARD_URL, timeout=30)
    #     resp.raise_for_status()
    #     dashboard = resp.json()
    #     chal = dashboard["king"]
    #     chal_repo = chal["model_repo"]
    #     chal_digest = chal.get("king_digest") or chal.get("model_digest")
    #     log.info("discovered challenger from dashboard: %s@%s",
    #              chal_repo, (chal_digest or "")[:19])
    # except Exception:
    #     log.warning("could not fetch dashboard, falling back to seed %s", SEED_REPO)
    #     raise

    chal_ref = ModelRef(chal_repo, chal_digest)

    # chal_dir = f"./miner/chal/{chal_repo}"
    # if os.path.exists(chal_dir):
    #     shutil.rmtree(chal_dir)
    chal_dir = os.path.abspath(f"./miner/{chal_repo}")
    log.info("downloading challenger from %s", chal_ref.immutable_ref)
    materialize_model(chal_ref, local_dir=chal_dir, max_workers=16)

    state = {
        "challenge_id": str(random.randint(0, 100)),
        "king_repo": king_repo,
        "king_digest": king_digest,
        "challenger_repo": chal_repo,
        "challenger_digest": chal_digest,
        "phase": "starting",
        "n_done": 0,
        "n_total": chain_config.DUEL_N_SAMPLES * chain_config.DUEL_MAX_TURNS_PER_SAMPLE,
        "king_mean": 0.0,
        "chal_mean": 0.0,
        "mean_delta": 0.0,
    }
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(None, connect=30.0)) as http:
        _fail_n = 0
        _busy_n = 0
        while True:
            try:
                await _eval_set_king(http, king_repo, king_digest)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 409:
                    _busy_n += 1
                    if _busy_n > 60:  # 60 × 30 s = 30 min ceiling
                        log.error("startup: eval has been running for >30 min; aborting")
                        return 3
                    log.info("startup /set_king: eval in progress (poll %d/60), waiting 30 s …",
                             _busy_n)
                    await asyncio.sleep(30.0)
                else:
                    _fail_n += 1
                    if _fail_n >= 3:
                        log.error("eval server unreachable on startup; aborting")
                        return 3
                    log.warning("startup /set_king attempt %d failed: %s", _fail_n, exc)
                    await asyncio.sleep(10.0)
            except Exception as exc:
                _fail_n += 1
                if _fail_n >= 3:
                    log.error("eval server unreachable on startup; aborting")
                    return 3
                log.warning("startup /set_king attempt %d failed: %s", _fail_n, exc)
                await asyncio.sleep(10.0)
        print("httpx--------")
        async def _bounded() -> None:
            await process_challenge(http, state)
        await asyncio.wait_for(_bounded(), timeout=TICK_RESTART_AFTER)




def main_sync() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(main_sync())


