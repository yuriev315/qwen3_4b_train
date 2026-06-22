"""Deterministic sampling from SWE-ZERO parquet shards via a manifest.json index."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import asyncio
import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# MODEL = "D:/MyWork/Albedo/prepare_data/ckpt_sft_0/merged"
MODEL = "D:/MyWork/Albedo/checkpoint/sft/merged"

INPUT_FILENAME = "../data/self_play/albedo-qwen3-4b-2-5-final/20260618_095350.json"

vllm_prompt = f'''
$env:CUDA_VISIBLE_DEVICES="1"
# Run the vLLM server
python -u -m vllm.entrypoints.openai.api_server `
--model {MODEL} `
--host 0.0.0.0 `
--port 8001 `
--max-model-len 32768 `
--gpu-memory-utilization 0.85 `
--dtype bfloat16 `
--no-enable-log-requests    
'''

OUTPUT_DIR = f"../data/self_play/{'_'.join(MODEL.split('/')[-2:])}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



async def _generate(
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict],
) -> tuple[str, dict, str | None]:
    """Call a vLLM endpoint. Returns (reply, usage, error). error is None on success."""
    if client is None:
        return "", {}, "Client is None!!!"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 1024,
    }
    try:
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        reply = (body["choices"][0]["message"].get("content") or "").strip()
        return reply, body.get("usage", {}), None
    except httpx.TimeoutException as exc:
        print("Warning: vLLM generation timed out: %s" % exc)
        return "", {}, f"vllm_timeout: {exc}"
    except httpx.HTTPStatusError as exc:
        print("Warning: vLLM returned HTTP %s" % exc.response.status_code)
        return "", {}, f"vllm_http_{exc.response.status_code}"
    except (KeyError, IndexError, ValueError) as exc:
        print("Warning: vLLM response malformed: %s" % exc)
        return "", {}, f"vllm_malformed: {exc}"


async def generate_turn(
        sample: dict,
        client: httpx.AsyncClient,
        model_name: str,
):
    """Generate king and challenger replies for one sampled turn."""
    global_idx = sample["global_idx"]
    prefix = sample['prefix']
    king_reply = sample['king_reply']
    (reply, usage, err) = await _generate(client, model_name, prefix)
    if err:
        return {
            "global_idx": global_idx,
            "prefix": prefix,
            "king_reply": king_reply,
            "chal_reply": reply,
            "usage": usage,
            "vllm_error": err,
        }

    return {
        "global_idx": global_idx,
        "prefix": prefix,
        "king_reply": king_reply,
        "chal_reply": reply,
        "usage": usage,
        "vllm_error": None
    }


async def run_dual(samples, client, model_name):
    semaphore = asyncio.Semaphore(2)
    duel_start = time.monotonic()
    _budget_logged = {"done": False}
    vllm_errors = 0
    DUEL_BUDGET_S = 1000.0
    generated_turns = []

    async def _generate_one(sample: dict) -> "GeneratedTurn | str | None":
        async with semaphore:
            if time.monotonic() - duel_start > DUEL_BUDGET_S:
                if not _budget_logged["done"]:
                    _budget_logged["done"] = True
                    print("Warning: run: soft budget %.0fs exceeded — stopping new generations" % DUEL_BUDGET_S)
                return "budget_skip"
            try:
                return await generate_turn(
                    sample,
                    client=client,
                    model_name=model_name,
                )
            except Exception as exc:
                print("Error: generate_turn failed for sample: %s" %  exc)
                return None

    tasks = [asyncio.create_task(_generate_one(s)) for s in samples]
    budget_skipped = 0
    for coro in asyncio.as_completed(tasks):
        generated = await coro
        if generated == "budget_skip":
            budget_skipped += 1
            # Turn not started before the soft budget — not a failure, just reduces n_done.
            continue
        if generated is None:
            vllm_errors += 1
            # Generation raised an unhandled exception — infra failure, not the challenger's fault.
            continue
        if generated.get("vllm_error"):
            vllm_errors += 1
            continue
        global_idx = generated["global_idx"]
        print(f"Response of {global_idx} sample: OK!")
        item = {
            "global_idx": global_idx,
            "prefix": generated.get("prefix"),
            "king_reply": generated.get("king_reply"),
            "chal_reply": generated.get("chal_reply"),
        }
        generated_turns.append(item)
    return generated_turns


async def get_client_and_name(address="http://localhost", port=8001):
    client = httpx.AsyncClient(
        base_url=f"{address}:{port}",
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
    )
    try:
        resp = await client.get("/v1/models", timeout=10.0)
        resp.raise_for_status()
        model_name = resp.json()["data"][0]["id"]
    except Exception as exc:
        print("model name discovery failed: %s — using 'default'" % exc)
        model_name = "default"
    return client, model_name


async def main(model_name=MODEL, sample_filename=""):
    if not sample_filename:
        print("Error: Empty filename!")
        return
    dataset = []
    with open(sample_filename, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            global_idx = row["global_idx"]
            prefix = row["prefix"]
            king_reply = row["reply"]
            if not (global_idx + 1):
                print("Error: Empty global idx")
                continue
            if not prefix:
                print("Error: Empty Prefix!")
                continue
            if not king_reply.strip():
                print("Error: Empty King Reply!")
                continue

            dataset.append({
                "global_idx": global_idx,
                "prefix": prefix,
                "king_reply": king_reply
            })

    print("Data generation done!")
    client = httpx.AsyncClient(
        base_url="http://localhost:8001",
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
    )
    results = await run_dual(dataset, client, model_name)
    print(len(results))

    with open(f'{OUTPUT_DIR}/{sample_filename.split("/")[-1]}', "w", encoding="utf-8") as fout:
        for item in results:
            # print(item)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    input(vllm_prompt)
    print(INPUT_FILENAME)
    print("main_run!")
    raise SystemExit(asyncio.run(main(MODEL, INPUT_FILENAME)))


