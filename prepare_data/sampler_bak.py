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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from albedo.config import DATASET_MANIFEST_SHA256
DATASET_MANIFEST_SHA256 = "f6ada52720e8c898d2eb4812973438bbf22380da29e9bf8f0508ca513eb175fc"

from sentence_transformers import SentenceTransformer, util

# Load a similarity model
sim_model = SentenceTransformer('all-MiniLM-L6-v2')


@dataclass
class Sample:
    global_idx: int
    shard_idx: int
    shard_name: str
    sample_idx: int
    turn_idx: int
    instance_id: str
    repo: str
    messages_prefix: list[dict]  # conversation history up to this turn
    messages_prompt: list[dict]  # current user turn (single entry)
    original_reply: str


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Return all rows from a parquet file as plain dicts."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        return table.to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(str(path)).to_dict(orient="records")


def _verify_manifest_sha256(manifest_path: Path, expected: str) -> None:
    """Raise ValueError if manifest sha256 doesn't match expected."""
    if not expected:
        return
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"manifest.json sha256 mismatch: "
            f"expected {expected!r}, got {digest!r}"
        )


def _extract_turns(row: dict[str, Any], shard_name: str, shard_idx: int, row_idx: int) -> list[Sample]:
    """Expand a parquet row into one Sample per assistant turn."""
    messages: list[dict] = row.get("messages") or []
    instance_id: str = row.get("instance_id", "")
    repo: str = row.get("repo", "")

    samples: list[Sample] = []
    prefix: list[dict] = []
    turn_idx = 0

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            user_prompt = prefix[-1:] if prefix and prefix[-1].get("role") == "user" else []
            samples.append(
                Sample(
                    global_idx=-1,  # filled in by TrajectoryDataset.sample
                    shard_idx=shard_idx,
                    shard_name=shard_name,
                    sample_idx=row_idx,
                    turn_idx=turn_idx,
                    instance_id=instance_id,
                    repo=repo,
                    messages_prefix=list(prefix[:-1]) if user_prompt else list(prefix),
                    messages_prompt=user_prompt,
                    original_reply=msg.get("content", ""),
                )
            )
            turn_idx += 1
        prefix.append(msg)

    return samples


class TrajectoryDataset:
    """Lazy-loading wrapper around SWE-ZERO parquet shards with manifest verification."""

    def __init__(self, dataset_dir: str, *, verify_manifest: bool = True) -> None:
        self._root = Path(dataset_dir)
        manifest_path = self._root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self._root}")

        if verify_manifest:
            _verify_manifest_sha256(manifest_path, DATASET_MANIFEST_SHA256)

        with manifest_path.open() as fh:
            self._manifest: dict[str, Any] = json.load(fh)

        # manifest schema: {"shards": [{"name": "...", "rows": N}, ...], "total_rows": N}
        self._shards: list[dict[str, Any]] = self._manifest.get("shards", [])
        self._total_rows: int = self._manifest.get("total_rows", sum(s.get("rows", 0) for s in self._shards))

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def sample(self, seed: bytes, n_samples: int, max_turns: int) -> list[Sample]:
        """Deterministically select n_samples (instance, turn) pairs from seed.

        Seeds random.Random from the first 8 bytes of seed (little-endian).
        Only loads shards that contain selected rows.
        """
        if not self._shards:
            return []

        entropy = int.from_bytes(seed[:8], "little")
        rng = random.Random(entropy)

        flat_index: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self._shards):
            n_rows = shard.get("rows", 0)
            flat_index.extend((shard_idx, row_idx) for row_idx in range(n_rows))

        if not flat_index:
            return []

        # Oversample to allow for deduplication and skipped rows.
        chosen_positions = rng.choices(range(len(flat_index)), k=n_samples * 4)

        shard_to_rows: dict[int, list[int]] = {}
        for pos in chosen_positions:
            shard_idx, row_idx = flat_index[pos]
            shard_to_rows.setdefault(shard_idx, []).append(row_idx)

        row_cache: dict[tuple[int, int], dict[str, Any]] = {}
        for shard_idx, row_indices in shard_to_rows.items():
            shard_name = (self._shards[shard_idx].get("path") or self._shards[shard_idx]["name"])
            shard_path = self._root / shard_name  # name is relative to dataset root
            if not shard_path.exists():
                shard_path = self._root / Path(shard_name).name
            rows = _load_parquet_rows(shard_path)
            for row_idx in set(row_indices):
                if row_idx < len(rows):
                    row_cache[(shard_idx, row_idx)] = rows[row_idx]

        collected: list[Sample] = []
        seen: set[tuple[int, int, int]] = set()  # (shard_idx, row_idx, turn_idx)

        for pos in chosen_positions:
            if len(collected) >= n_samples:
                break
            shard_idx, row_idx = flat_index[pos]
            row = row_cache.get((shard_idx, row_idx))
            if row is None:
                continue

            shard_name = (self._shards[shard_idx].get("path") or self._shards[shard_idx]["name"])
            turns = _extract_turns(row, shard_name, shard_idx, row_idx)
            if not turns:
                continue

            eligible = [t for t in turns if t.turn_idx < max_turns]
            if not eligible:
                continue
            turn = rng.choice(eligible)

            key = (shard_idx, row_idx, turn.turn_idx)
            if key in seen:
                continue
            seen.add(key)

            turn.global_idx = len(collected)
            collected.append(turn)

        return collected


def get_training_seed(use_fixed: bool = False) -> bytes:
    """Generate bytes seed for TrajectoryDataset.sample()."""
    if use_fixed:
        # Fixed seed for reproducibility
        seed_str = "my_training_seed_v1"
    else:
        # Time-based for diversity (changes every hour)
        bucket = int(time.time() // 3600)
        seed_str = f"training_seed_{bucket}"

    # Return bytes directly (not int)
    return hashlib.blake2b(seed_str.encode()).digest()


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
        # "seed": 42
    }
    try:
        # torch.manual_seed(42)
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        reply = (body["choices"][0]["message"].get("content") or "").strip()
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(
            ids,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            # top_k=-1,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply2 = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(reply)
        print(reply2)
        print(reply == reply2)
        # Get embeddings
        emb1 = sim_model.encode(reply)
        emb2 = sim_model.encode(reply2)
        # Calculate similarity
        similarity = util.cos_sim(emb1, emb2)
        print(f"Semantic similarity: {similarity.item():.3f}")

        # Good similarity is > 0.85
        if similarity > 0.85:
            print("✓ Responses are semantically equivalent")
        else:
            print("✗ Responses differ significantly")
        print("--------------------------------------------------------------")
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
        sample: Sample,
        *,
        king_client: httpx.AsyncClient,
        king_name: str,
        chal_client: httpx.AsyncClient,
        chal_name: str
):
    """Generate king and challenger replies for one sampled turn."""
    full_messages = sample.messages_prefix + sample.messages_prompt

    # (king_reply, king_usage, king_err), (chal_reply, chal_usage, chal_err) = \
    #     await asyncio.gather(
    #         _generate(king_client, king_name, full_messages),
    #         _generate(chal_client, chal_name, full_messages),
    #     )
    (king_reply, king_usage, king_err) = await _generate(king_client, king_name, full_messages)
    # print(king_reply)
    # print(king_usage)
    # print(king_err)
    # print("========================================================================")

    # if king_err or chal_err:
    #     error_msg = f"king_{king_err}" if king_err else f"chal_{chal_err}"
    #     return {
    #         "sample": sample,
    #         "king_reply": king_reply, "chal_reply": chal_reply,
    #         "king_usage": king_usage, "chal_usage": chal_usage,
    #         "vllm_error": error_msg,
    #     }
    #
    # return {
    #     "sample": sample,
    #     "king_reply": king_reply,
    #     "chal_reply": chal_reply,
    #     "king_usage": king_usage,
    #     "chal_usage": chal_usage,
    # }


async def run_dual(samples, king_client, king_name, chal_client=None, chal_name=""):
    async def _generate_one(sample: Sample) -> "GeneratedTurn | str | None":
        semaphore = asyncio.Semaphore(2)
        async with semaphore:
            try:
                return await generate_turn(
                    sample,
                    king_client=king_client,
                    king_name=king_name,
                    chal_client=chal_client,
                    chal_name=chal_name
                )
            except Exception as exc:
                print("generate_turn failed for sample %d: %s" % (sample.global_idx, exc))
                return None

    for s in samples:
        res = await _generate_one(s)
    return


async def get_client_and_name(address="http://localhost", port=8001, flag=None):
    print("Getting Client")
    if flag is None:
        return None, ""
    client = httpx.AsyncClient(
        base_url=f"{address}:{port}",
        # Long read timeout: a 4B model generating 1024 tokens at ~50 tok/s
        # takes ~20s normally; allow 3× headroom for loaded GPU boxes.
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
    )
    try:
        resp = await client.get("/v1/models", timeout=10.0)
        resp.raise_for_status()
        name = resp.json()["data"][0]["id"]
    except Exception as exc:
        print("model name discovery failed: %s — using 'default'" % exc)
        name = "default"
    return client, name


async def main(random_test=True, king=True, chal=True, n_samples=64, max_turns=10):
    if random_test:
        _DATASET_DIR = "./data/swe-zero"
        seed = get_training_seed()


        dataset = await asyncio.to_thread(
            lambda: TrajectoryDataset(_DATASET_DIR).sample(seed, n_samples, max_turns)
        )
        king_client, king_name = await get_client_and_name("http://localhost", 8001, king)
        # chal_client, chal_name = await get_client_and_name("http://localhost", 8002, chal)
        result = await run_dual(dataset, king_client, king_name)
        return result


if __name__ == "__main__":
    random_flag = True
    king = True
    challenger = False
    n_samples = 3
    max_turns = 2
    raise SystemExit(asyncio.run(main(random_flag, king, challenger, n_samples, max_turns)))
