import json
import os
import shutil
from pathlib import Path

os.environ["USE_LIBUV"] = "0"

if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "../checkpoint/king/arboshelper/albedo-qwen3-4b-2-5-final"
SFT_DATA_PATH = Path("../data/sft_data.jsonl")
TEXT_CACHE_DIR = Path("data/sft_text_cache")
TEXT_CACHE_META = TEXT_CACHE_DIR / "cache_meta.json"
LEGACY_TOKENIZED_DIR = Path("data/sft_tokenized")
OUTPUT_DIR = Path("ckpt_sft")

# Tuned for ~4.3k chat SFT examples on Qwen3-4B LoRA (8k context, effective batch 8).
# packing=False + completion_only_loss=True: train only assistant tokens (best format accuracy).
# Liger + flash-attn + fused AdamW: speed; load_best_model_at_end: pick lowest eval_loss checkpoint.
HYPERPARAMS = {
    "max_length": 8192,
    "per_device_batch": 2,
    "grad_accum": 4,
    "num_epochs": 3,
    "learning_rate": 1.2e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 3,
    "logging_steps": 10,
    "dataset_num_proc": min(8, os.cpu_count() or 4),  # parallel tokenization in SFTTrainer
}


def _source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size}


def _cache_is_valid(meta_path: Path, source_path: Path, max_length: int) -> bool:
    if not meta_path.is_file() or not TEXT_CACHE_DIR.is_dir():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("source") == _source_fingerprint(source_path)
        and meta.get("max_length") == max_length
        and meta.get("columns") == ["text"]
    )


def _write_cache_meta(source_path: Path, max_length: int, num_rows: int) -> None:
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": _source_fingerprint(source_path),
        "max_length": max_length,
        "num_rows": num_rows,
        "columns": ["text"],
    }
    TEXT_CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _remove_legacy_tokenized_cache() -> None:
    if LEGACY_TOKENIZED_DIR.exists():
        print(f"Removing legacy tokenized cache: {LEGACY_TOKENIZED_DIR}")
        shutil.rmtree(LEGACY_TOKENIZED_DIR)


def load_data(path: Path, tokenizer, max_length: int) -> Dataset:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = []
    dropped_format = 0
    dropped_length = 0
    max_seen = 0

    for record in records:
        if isinstance(record, str):
            text = record
        elif isinstance(record, dict) and "text" in record:
            text = record["text"]
        else:
            dropped_format += 1
            continue

        length = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        max_seen = max(max_seen, length)
        if length > max_length:
            dropped_length += 1
            continue
        kept.append({"text": text})

    print(
        f"Dataset: {len(kept)} kept, {dropped_format} bad format, "
        f"{dropped_length} over max_length ({max_length}), longest={max_seen} tokens"
    )
    return Dataset.from_list(kept)


def load_or_build_dataset(tokenizer, max_length: int) -> Dataset:
    _remove_legacy_tokenized_cache()

    if not SFT_DATA_PATH.is_file():
        raise FileNotFoundError(f"SFT data not found: {SFT_DATA_PATH}")

    if _cache_is_valid(TEXT_CACHE_META, SFT_DATA_PATH, max_length):
        print(f"Loading cached text dataset from {TEXT_CACHE_DIR}")
        return Dataset.load_from_disk(TEXT_CACHE_DIR)

    if TEXT_CACHE_DIR.exists():
        print("SFT source changed or cache invalid; rebuilding text cache")
        shutil.rmtree(TEXT_CACHE_DIR)

    dataset = load_data(SFT_DATA_PATH, tokenizer, max_length)
    dataset.save_to_disk(TEXT_CACHE_DIR)
    _write_cache_meta(SFT_DATA_PATH, max_length, len(dataset))
    print(f"Saved text cache to {TEXT_CACHE_DIR}")
    return dataset


def _latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: int(p.name.split("-")[1])).resolve())


def _resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _use_liger_kernel() -> bool:
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        print("liger-kernel not installed; training without Liger kernels")
        return False


def _fused_adamw_available() -> bool:
    import inspect

    return "fused" in inspect.signature(torch.optim.AdamW.__init__).parameters


def sft_train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(f"Using device: cuda:{local_rank}" if torch.cuda.is_available() else "Using device: cpu")

    hp = HYPERPARAMS
    max_length = hp["max_length"]
    attn_implementation = _resolve_attn_implementation()
    print(f"Attention backend: {attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=False,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_rslora=True,
    )
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    dataset = load_or_build_dataset(tokenizer, max_length)
    print(f"Training on {len(dataset)} examples")

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train split: {len(train_dataset)} examples")
    print(f"Eval split: {len(eval_dataset)} examples")

    per_device_batch = hp["per_device_batch"]
    grad_accum = hp["grad_accum"]
    effective_batch = per_device_batch * grad_accum
    num_epochs = hp["num_epochs"]
    total_steps = (len(train_dataset) // effective_batch) * num_epochs
    warmup_steps = max(1, int(total_steps * hp["warmup_ratio"]))
    dataset_num_proc = hp["dataset_num_proc"]
    optim = "adamw_torch_fused" if _fused_adamw_available() else "adamw_torch"
    use_liger = _use_liger_kernel()

    print(f"\n{'=' * 50}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 50}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Epochs: {num_epochs}, total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"Learning rate: {hp['learning_rate']}")
    print(f"Completion-only loss: True, packing: True")
    print(f"Liger kernel: {use_liger}")
    print(f"Optimizer: {optim}")
    print(f"{'=' * 50}\n")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=num_epochs,
        learning_rate=hp["learning_rate"],
        bf16=True,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=hp["weight_decay"],
        max_grad_norm=hp["max_grad_norm"],
        logging_steps=hp["logging_steps"],
        eval_strategy="steps",
        eval_steps=hp["eval_steps"],
        save_strategy="steps",
        save_steps=hp["save_steps"],
        save_total_limit=hp["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataset_text_field="text",
        dataset_num_proc=dataset_num_proc,
        packing=False,
        completion_only_loss=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        use_liger_kernel=use_liger,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=True,
        optim=optim,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    resume_checkpoint = _latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("No checkpoint found, starting fresh")
        trainer.train()

    save_path = OUTPUT_DIR / "final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    sft_train()
