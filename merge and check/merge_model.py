import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_MODEL = Path("../checkpoint/king/sota1028/albedo-qwen3-4b-miner_5")
ADAPTER_PATH = Path("../checkpoint/dpo/dpo_01/albedo-qwen3-4b-miner_5/final")
# ADAPTER_PATH = Path("./ckpt_sft/checkpoint-550")
# OUTPUT_PATH = Path("./ckpt_sft/merged-550")
OUTPUT_PATH = Path("../checkpoint/dpo/dpo_01/albedo-qwen3-4b-miner_5/merged")


def _resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def merge_sft_model(
    base_model: Path,
    adapter_path: Path,
    output_path: Path,
    max_shard_size: str = "4GB",
) -> None:
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    attn = _resolve_attn_implementation()
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Device map: {device_map}, attention: {attn}")

    print(f"Loading base model from {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation=attn,
    )

    print(f"Loading LoRA adapter from {adapter_path}")
    sft = PeftModel.from_pretrained(base, str(adapter_path))

    print("Merging adapter into base weights...")
    model = sft.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path} (shards: {max_shard_size})...")
    model.save_pretrained(
        str(output_path),
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    print("\nChecking saved files...")
    for name in required_files:
        path = output_path / name
        print(f"  OK   {name}" if path.is_file() else f"  MISS {name}")

    print("\n" + "=" * 50)
    print("FILES IN OUTPUT FOLDER")
    print("=" * 50)
    for path in sorted(output_path.iterdir()):
        size = path.stat().st_size
        size_str = f"{size / 1024 ** 2:.0f} MB" if size > 1024 ** 2 else f"{size / 1024:.0f} KB"
        print(f"  {path.name:<35} {size_str:>10}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nMerged model saved to {output_path}")
    print(f"Parameters: {param_count:,}, size ~{param_count * 2 / 1024 ** 3:.1f} GB (bf16)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SFT LoRA adapter into the base model.")
    parser.add_argument("--base", type=Path, default=BASE_MODEL, help="Base model directory")
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH, help="LoRA adapter directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Merged model output directory")
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Shard size for safetensors export (King format uses ~4GB shards)",
    )
    args = parser.parse_args()

    merge_sft_model(
        base_model=args.base,
        adapter_path=args.adapter,
        output_path=args.output,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
