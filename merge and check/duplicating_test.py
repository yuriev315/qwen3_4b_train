import math
from pathlib import Path
import hashlib



_UNCHANGED_COSINE = 1.0 - 1e-6
FINGERPRINT_METHOD = "layer_norms_v2_with_samples"
SAMPLE_K = 16 


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    """k stable indices into a length-n tensor, derived from its key (shard-order invariant)."""
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4:(i + 1) * 4], "big") % n for i in range(k)]





def _vector_cosine(a: list[float], b: list[float]) -> float:
    """Cosine of two equal-length vectors; 0.0 on empty, mismatched, or zero-magnitude input."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


def similarity(fp_a: dict, fp_b: dict) -> float:
    """v2 similarity in [0, 1]: fraction of tensors whose sampled values are ~unchanged.

    Returns 0.0 when architectures differ (layer_keys mismatch). Falls back to v1
    norm-vector cosine when per-tensor samples are missing on either side.
    """
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0

    sa, sb, keys = fp_a.get("tensor_samples"), fp_b.get("tensor_samples"), fp_a.get("layer_keys")
    if sa and sb and len(sa) == len(sb):
        unchanged = 0
        for a, b, k in zip(sa, sb, keys):
            cos = _vector_cosine(a, b)
            print(k, sum(abs(x - y) for x, y in zip(a, b)))
            # zero-vectors (e.g. uninitialised biases) trivially match in both copies
            if (not any(a) and not any(b)) or cos >= _UNCHANGED_COSINE:
                unchanged += 1
        return unchanged / len(sa)

    return _vector_cosine(fp_a.get("norm_vector", []), fp_b.get("norm_vector", []))


def compute_fingerprint(model_dir: str) -> dict:
    """Compute a v2 fingerprint: per-tensor L2 norm + K deterministic value samples.

    Returns {"method", "layer_keys", "norm_vector", "tensor_samples"} with layer_keys
    sorted for shard-order invariance. Raises FileNotFoundError if no shards are found.
    The value samples make the comparison direction-sensitive — a scaled or fine-tuned
    copy still shifts sampled values, unlike a norm-only fingerprint.
    """
    try:
        from safetensors import safe_open  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("safetensors is required for fingerprinting") from exc

    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors files found in {model_dir!r}")

    norms: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                flat = f.get_tensor(key).reshape(-1).float()
                n = int(flat.shape[0])
                norms[key] = float((flat * flat).sum().sqrt().item())
                samples[key] = [float(flat[i].item()) for i in _deterministic_indices(key, n, SAMPLE_K)]

    keys = sorted(norms)
    return {
        "method":         FINGERPRINT_METHOD,
        "layer_keys":     keys,
        "norm_vector":    [norms[k] for k in keys],
        "tensor_samples": [samples[k] for k in keys],
    }



def check_fingerprint(challenger_dir, king_dir, threshold=0.95):
    challenger_fp = compute_fingerprint(challenger_dir)
    king_fp = compute_fingerprint(king_dir)
    sim = similarity(challenger_fp, king_fp)
    print(sim)

    return sim >= threshold

if __name__ == '__main__':
    CHAL = "../checkpoint/sft/merged"
    KING = "../checkpoint/king/arboshelper/albedo-qwen3-4b-2-5-final"
    duplicated = check_fingerprint(CHAL, KING)
    if duplicated:
        print("Duplicated")
    else:
        print("Not duplicated")