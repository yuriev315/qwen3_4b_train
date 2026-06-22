import gc
import json
import os
import shutil
import time
from pathlib import Path

# Environment setup - MUST be before torch import
os.environ["USE_LIBUV"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "./king/sota1028/albedo-qwen3-4b-miner_5"
SFT_DATA_PATH = Path("data/sft_data.jsonl")
TEXT_CACHE_DIR = Path("data/sft_text_cache2")
TEXT_CACHE_META = TEXT_CACHE_DIR / "cache_meta2.json"
LEGACY_TOKENIZED_DIR = Path("data/sft_tokenized2")
OUTPUT_DIR = Path("ckpt_sft2")

# Optimized for Qwen3-4B LoRA on 24GB VRAM
# packing=False + completion_only_loss=True: train only assistant tokens
# 8k context with per_device_batch=2 requires flash-attention or sdpa
HYPERPARAMS = {
    # Sequence lengths
    "max_length": 8192,

    # Training batch
    "per_device_batch": 2,
    "grad_accum": 4,
    "num_epochs": 2,

    # Optimization
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # LoRA (full target modules for Qwen3)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Checkpointing
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 3,
    "logging_steps": 10,

    # DataLoader (reduced for Windows stability)
    "dataset_num_proc": 2,  # Reduced from 8 to avoid deadlocks
    "dataloader_num_workers": 2,  # Added

    # Memory optimizations
    "gradient_checkpointing": True,
    "use_liger_kernel": True,  # Set to False if not installed
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        _log(f"Removing legacy tokenized cache: {LEGACY_TOKENIZED_DIR}")
        shutil.rmtree(LEGACY_TOKENIZED_DIR)


def load_data(path: Path, tokenizer, max_length: int) -> Dataset:
    """Load and filter SFT data by token length."""
    _log(f"Loading data from {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                _log(f"Warning: Skipping line {line_num}: {e}")

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

        # Tokenize without special tokens to check length
        length = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        max_seen = max(max_seen, length)

        if length > max_length:
            dropped_length += 1
            continue

        kept.append({"text": text})

    _log(f"Dataset: {len(kept)} kept, {dropped_format} bad format, "
         f"{dropped_length} over max_length ({max_length}), longest={max_seen} tokens")

    return Dataset.from_list(kept)


def load_or_build_dataset(tokenizer, max_length: int) -> Dataset:
    """Load cached dataset or build from source."""
    _remove_legacy_tokenized_cache()

    if not SFT_DATA_PATH.is_file():
        raise FileNotFoundError(f"SFT data not found: {SFT_DATA_PATH}")

    if _cache_is_valid(TEXT_CACHE_META, SFT_DATA_PATH, max_length):
        _log(f"Loading cached text dataset from {TEXT_CACHE_DIR}")
        return Dataset.load_from_disk(TEXT_CACHE_DIR)

    if TEXT_CACHE_DIR.exists():
        _log("SFT source changed or cache invalid; rebuilding text cache")
        shutil.rmtree(TEXT_CACHE_DIR)

    dataset = load_data(SFT_DATA_PATH, tokenizer, max_length)
    dataset.save_to_disk(TEXT_CACHE_DIR)
    _write_cache_meta(SFT_DATA_PATH, max_length, len(dataset))
    _log(f"Saved text cache to {TEXT_CACHE_DIR}")

    return dataset


def _latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: int(p.name.split("-")[1])).resolve())


def _resolve_attn_implementation() -> str:
    """Use Flash Attention if available, otherwise SDPA."""
    try:
        import flash_attn  # noqa: F401
        _log("Using Flash Attention 2")
        return "flash_attention_2"
    except ImportError:
        _log("Using SDPA attention")
        return "sdpa"


def _use_liger_kernel() -> bool:
    """Check if Liger kernel is available."""
    try:
        import liger_kernel  # noqa: F401
        _log("Liger kernel enabled")
        return True
    except ImportError:
        _log("Liger kernel not installed; training without Liger")
        return False


def _fused_adamw_available() -> bool:
    import inspect
    return "fused" in inspect.signature(torch.optim.AdamW.__init__).parameters


def print_gpu_memory(tag: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        free = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) - reserved
        _log(f"GPU Memory [{tag}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def sft_train():
    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    _log(f"Using device: cuda:{local_rank}" if torch.cuda.is_available() else "Using device: cpu")

    hp = HYPERPARAMS
    max_length = hp["max_length"]
    attn_implementation = _resolve_attn_implementation()

    # [1/6] Load tokenizer
    _log("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    _log(f"Tokenizer: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")

    # [2/6] Load model (with optional 4-bit quantization for memory savings)
    _log("[2/6] Loading model...")

    # Optional: Use 4-bit quantization if OOM
    use_4bit = False  # Set to True if you need memory savings
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=False,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=False,
            device_map="auto",
        )

    print_gpu_memory("after model load")

    # [3/6] Configure LoRA
    _log("[3/6] Configuring LoRA...")
    peft_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=hp["lora_target_modules"],
        use_rslora=True,
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    _log(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # [4/6] Apply memory optimizations
    _log("[4/6] Applying memory optimizations...")
    model.config.use_cache = False

    use_grad_ckpt = hp.get("gradient_checkpointing", True)
    if use_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        _log("Gradient checkpointing enabled")

    free_gpu_memory()
    print_gpu_memory("after optimizations")

    # [5/6] Load dataset
    _log("[5/6] Loading dataset...")
    dataset = load_or_build_dataset(tokenizer, max_length)
    _log(f"Total examples: {len(dataset)}")

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    _log(f"Train split: {len(train_dataset)} examples")
    _log(f"Eval split: {len(eval_dataset)} examples")

    # [6/6] Training configuration
    per_device_batch = hp["per_device_batch"]
    grad_accum = hp["grad_accum"]
    effective_batch = per_device_batch * grad_accum
    num_epochs = hp["num_epochs"]
    total_steps = (len(train_dataset) // effective_batch) * num_epochs
    warmup_steps = max(1, int(total_steps * hp["warmup_ratio"]))
    dataset_num_proc = min(hp["dataset_num_proc"], 4)  # Cap at 4 for Windows
    optim = "adamw_torch_fused" if _fused_adamw_available() else "adamw_torch"
    use_liger = _use_liger_kernel() and hp.get("use_liger_kernel", False)

    _log(f"\n{'=' * 50}")
    _log("TRAINING CONFIGURATION")
    _log(f"{'=' * 50}")
    _log(f"Effective batch size: {effective_batch}")
    _log(f"Epochs: {num_epochs}, total steps: {total_steps}, warmup: {warmup_steps}")
    _log(f"Learning rate: {hp['learning_rate']}")
    _log(f"Max length: {max_length}")
    _log(f"Completion-only loss: True, packing: False")
    _log(f"Liger kernel: {use_liger}")
    _log(f"Optimizer: {optim}")
    _log(f"Dataset workers: {dataset_num_proc}")
    _log(f"{'=' * 50}\n")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=num_epochs,
        learning_rate=hp["learning_rate"],
        bf16=True,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
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
        packing=False,  # Don't pack sequences - use completion_only_loss
        completion_only_loss=True,  # Only train on assistant responses
        report_to="none",
        ddp_find_unused_parameters=False,
        use_liger_kernel=use_liger,
        dataloader_num_workers=hp.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if hp.get("dataloader_num_workers", 2) > 0 else None,
        remove_unused_columns=True,  # Safe for SFT with text field
        optim=optim,
    )

    _log("[6/6] Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    free_gpu_memory()
    print_gpu_memory("before training")

    _log("\n" + "=" * 50)
    _log("STARTING SFT TRAINING")
    _log("=" * 50)

    resume_checkpoint = _latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint:
        _log(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        _log("No checkpoint found, starting fresh")
        trainer.train()

    # Save final model
    _log("\nSaving final model...")
    save_path = OUTPUT_DIR / "final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    _log(f"Model saved to {save_path}")

    # Optionally merge LoRA
    _log("\n" + "=" * 50)
    _log("SFT TRAINING COMPLETE")
    _log("=" * 50)
    _log(f"Final model: {OUTPUT_DIR}/final/")
    _log(f"To merge LoRA: Use peft_model.merge_and_unload()")


if __name__ == "__main__":
    sft_train()