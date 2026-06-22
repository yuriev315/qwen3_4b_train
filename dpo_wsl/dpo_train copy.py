"""
Weighted DPO training (QLoRA) tuned for Linux + dual RTX 4090 (24 GB each).

Launch (recommended — DDP, most stable with QLoRA + bitsandbytes):
  cd dpo
  accelerate launch --config_file accelerate_configs/ddp_2gpu.yaml dpo_train.py

Alternative (DeepSpeed ZeRO-2):
  accelerate launch --config_file accelerate_configs/deepspeed_zero2.yaml dpo_train.py

Single GPU:
  python dpo_train.py
"""

import gc
import json
import os
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

# Linux / CUDA allocator — set before importing torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if os.environ.get("DPO_DEBUG", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from accelerate import PartialState
from accelerate.state import DistributedType
from datasets import Dataset, IterableDataset
from peft import LoraConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import get_kbit_device_map

try:
    from accelerate.utils import tqdm
except ImportError:
    from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

STATE = PartialState()


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    if STATE.is_main_process:
        print(*args, **kwargs)


def _log_all(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(f"[rank {STATE.process_index}]", *args, **kwargs)


MODEL_NAME = "../checkpoint/sft/merged"
# MODEL_NAME = "../checkpoint/king/sota1028/albedo-qwen3-4b-miner_5"
DPO_DATA_PATH = Path("../data/dpo_data.jsonl")

DPO_CACHE_DIR = Path("cache/dpo/dataset_cache")
DPO_CACHE_META = DPO_CACHE_DIR / "cache_meta.json"
OUTPUT_DIR = Path(f"../checkpoint/dpo/{MODEL_NAME.split('/')[-2]}")

# ~1.9k weighted pairs | 4B QLoRA | 4k context | 2×4090 (24 GB each)
#
# Training math (defaults below):
#   train ≈ 1,788 samples (5% eval holdout)
#   effective batch = 1 × 4 grad_accum × 2 GPUs = 8
#   steps/epoch ≈ 224  →  eval every 25 steps ≈ 9 checkpoints/epoch
#
# sigmoid_norm: length-normalized DPO — critical here (rejected up to ~9k chars).
# LR 1.5e-6 + 1 epoch: enough signal without drifting far from the SFT checkpoint.
# beta 0.1: moderate preference strength; pair with sigmoid_norm (not plain sigmoid).
# LoRA r=8: small dataset — lower rank reduces overfitting risk.
# Competition data: many pairs exceed 8192 tokens; 4096 is the 24 GB DDP training cap (do not lower).
# TRL truncates keep_start at train time; sigmoid_norm reduces length bias in the loss.
# OOM mitigation: precompute_ref_log_probs + activation_offloading (not shorter max_length).
# Ref logprobs must be precomputed before multi-GPU launch (see launch_ddp.sh).
#   python -u dpo_train.py --precompute-ref-only
#   Set DPO_PRECOMPUTE_REF=0 to fall back to on-the-fly ref (uses more VRAM).
HYPERPARAMS = {
    "max_length": 4096,
    "per_device_batch": 1,
    "per_device_eval_batch": 1,
    "grad_accum": 4,
    "num_epochs": 1,
    "learning_rate": 5e-6,
    "beta": 0.1,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "loss_type": ["sigmoid_norm"],
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "gradient_checkpointing": True,
    "eval_steps": 25,
    "save_steps": 50,
    "save_total_limit": 2,
    "logging_steps": 5,
    "eval_accumulation_steps": 4,
    "precompute_ref_log_probs": True,
    "precompute_ref_batch_size": 4,
    "activation_offloading": True,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 0,  # >0 can hang on WSL during ref-logprob precompute
    "use_weighting": True,  # hyperparam hint only; WPO on by default via DPO_USE_WEIGHTING=1 in _resolve_use_weighting
    "use_liger_kernel": False,  # incompatible with PEFT + sigmoid_norm in TRL/liger
}


def _source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size}


def _cache_is_valid(meta_path: Path, source_path: Path, model_name: str, max_length: int) -> bool:
    if not meta_path.is_file() or not DPO_CACHE_DIR.is_dir():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("source") == _source_fingerprint(source_path)
        and meta.get("model_name") == model_name
        and meta.get("max_length") == max_length
        and meta.get("columns") == ["prompt", "chosen", "rejected", "weight"]
        and meta.get("fit_mode") == "raw_completion"
    )


def _write_cache_meta(source_path: Path, model_name: str, max_length: int, num_rows: int) -> None:
    DPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": _source_fingerprint(source_path),
        "model_name": model_name,
        "max_length": max_length,
        "num_rows": num_rows,
        "columns": ["prompt", "chosen", "rejected", "weight"],
        "fit_mode": "raw_completion",
    }
    DPO_CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _ref_logps_cache_dir(name: str, num_rows: int, max_length: int, model_id: str) -> Path:
    model_slug = str(model_id).replace("/", "_").replace("..", "_")
    cache_key = f"{name}_n{num_rows}_len{max_length}_{model_slug}"
    return (DPO_CACHE_DIR / "ref_logps" / cache_key).resolve()


def _ref_logps_cache_ready(cache_dir: Path) -> bool:
    return cache_dir.is_dir() and (cache_dir / ".done").is_file()


def _save_ref_logps_cache(cache_dir: Path, ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ref_dataset = Dataset.from_dict(
        {
            "ref_chosen_logps": ref_chosen_logps.float().cpu().numpy(),
            "ref_rejected_logps": ref_rejected_logps.float().cpu().numpy(),
        }
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    ref_dataset.save_to_disk(str(cache_dir))
    (cache_dir / ".done").write_text("ok", encoding="utf-8")


def _merge_ref_logps_from_cache(dataset: Dataset, cache_dir: Path) -> Dataset:
    if "ref_chosen_logps" in dataset.column_names and "ref_rejected_logps" in dataset.column_names:
        return dataset
    if not _ref_logps_cache_ready(cache_dir):
        raise FileNotFoundError(f"Reference logprob cache not ready at {cache_dir}")
    ref_dataset = Dataset.load_from_disk(str(cache_dir))
    if len(ref_dataset) != len(dataset):
        raise ValueError(
            f"Ref logprob cache row count mismatch for {cache_dir}: "
            f"cache={len(ref_dataset)} dataset={len(dataset)}"
        )
    dataset = dataset.add_column("ref_chosen_logps", ref_dataset["ref_chosen_logps"])
    return dataset.add_column("ref_rejected_logps", ref_dataset["ref_rejected_logps"])


def _compute_ref_logps_for_dataset(
    trainer: DPOTrainer,
    tokenized_dataset: Dataset,
    batch_size: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = trainer.accelerator.device
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=trainer.data_collator,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    ref_chosen_logps = []
    ref_rejected_logps = []
    for padded_batch in tqdm(iterable=dataloader, desc=desc):
        padded_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in padded_batch.items()
        }
        ref_chosen_logp, ref_rejected_logp = trainer.compute_ref_log_probs(padded_batch)
        ref_chosen_logps.append(ref_chosen_logp.cpu())
        ref_rejected_logps.append(ref_rejected_logp.cpu())
    return torch.cat(ref_chosen_logps), torch.cat(ref_rejected_logps)


def _run_ref_precompute_worker(work_dir: Path) -> None:
    """Single-process ref logprob precompute worker."""
    print("[ref-precompute] worker started", flush=True)
    meta = json.loads((work_dir / "meta.json").read_text(encoding="utf-8"))
    train_cache = Path(meta["train_cache"])
    eval_cache = Path(meta["eval_cache"])
    model_name = meta["model_name"]
    max_length = meta["max_length"]
    precompute_ref_batch_size = meta["precompute_ref_batch_size"]

    train_dataset = Dataset.load_from_disk(str(work_dir / "train"))
    eval_dataset = Dataset.load_from_disk(str(work_dir / "eval"))
    print(
        f"[ref-precompute] loaded datasets: train={len(train_dataset)}, eval={len(eval_dataset)}",
        flush=True,
    )

    print("[ref-precompute] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    hp = HYPERPARAMS
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    bootstrap_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=hp["beta"],
        max_length=max_length,
        per_device_train_batch_size=hp["per_device_batch"],
        precompute_ref_log_probs=False,
        precompute_ref_batch_size=precompute_ref_batch_size,
        dataset_num_proc=hp.get("dataset_num_proc", 4),
        remove_unused_columns=False,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "attn_implementation": _resolve_attn_implementation(),
            "dtype": torch.bfloat16,
            "trust_remote_code": False,
            "device_map": get_kbit_device_map(),
        },
    )

    print(
        "[ref-precompute] initializing DPOTrainer "
        "(loads 4-bit model + tokenizes datasets; can take several minutes)...",
        flush=True,
    )
    bootstrap = DDPSafeDPOTrainer(
        model=model_name,
        ref_model=None,
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=bootstrap_args,
    )
    print("[ref-precompute] DPOTrainer ready; starting ref logprob passes...", flush=True)

    if not _ref_logps_cache_ready(train_cache):
        chosen, rejected = _compute_ref_logps_for_dataset(
            bootstrap,
            bootstrap.train_dataset,
            precompute_ref_batch_size,
            "Computing reference log probs for train dataset",
        )
        _save_ref_logps_cache(train_cache, chosen, rejected)
        print(f"Saved train reference logprobs to {train_cache}")

    if not _ref_logps_cache_ready(eval_cache):
        chosen, rejected = _compute_ref_logps_for_dataset(
            bootstrap,
            bootstrap.eval_dataset,
            precompute_ref_batch_size,
            "Computing reference log probs for eval dataset",
        )
        _save_ref_logps_cache(eval_cache, chosen, rejected)
        print(f"Saved eval reference logprobs to {eval_cache}")


def ensure_ref_logprob_columns(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    *,
    precompute_ref: bool,
    max_length: int,
    precompute_ref_batch_size: int,
    tokenizer,
    lora_config: LoraConfig,
    bootstrap_args: DPOConfig,
    model_name: str,
) -> tuple[Dataset, Dataset]:
    """Attach precomputed ref logprob columns from shared disk cache."""
    if not precompute_ref:
        return train_dataset, eval_dataset

    train_cache = _ref_logps_cache_dir("train", len(train_dataset), max_length, model_name)
    eval_cache = _ref_logps_cache_dir("eval", len(eval_dataset), max_length, model_name)

    if not (_ref_logps_cache_ready(train_cache) and _ref_logps_cache_ready(eval_cache)):
        raise RuntimeError(
            "Reference logprob caches are missing.\n"
            f"  train: {train_cache}\n"
            f"  eval:  {eval_cache}\n"
            "Run precompute first (single GPU, ~10-20 min):\n"
            "  python -u dpo_train.py --precompute-ref-only\n"
            "or use ./launch_ddp.sh which runs this step automatically."
        )

    _log(f"Using cached reference logprobs:\n  {train_cache}\n  {eval_cache}")
    train_dataset = _merge_ref_logps_from_cache(train_dataset, train_cache)
    eval_dataset = _merge_ref_logps_from_cache(eval_dataset, eval_cache)
    return train_dataset, eval_dataset


def _resolve_precompute_ref(hyperparams: dict | None = None) -> bool:
    hp = hyperparams or HYPERPARAMS
    precompute_ref = hp.get("precompute_ref_log_probs", False)
    env = os.environ.get("DPO_PRECOMPUTE_REF", "").lower()
    if env in {"0", "false", "no"}:
        return False
    if env in {"1", "true", "yes"}:
        return True
    return precompute_ref


def precompute_ref_logprobs_if_needed(
    *,
    model_name: str | None = None,
    hyperparams: dict | None = None,
    data_path: Path | None = None,
) -> None:
    """Single-GPU precompute step; run before `accelerate launch` in multi-GPU mode."""
    model_name = model_name or MODEL_NAME
    hp = hyperparams or HYPERPARAMS
    data_path = data_path or DPO_DATA_PATH
    max_length = hp["max_length"]
    precompute_ref_batch_size = hp["precompute_ref_batch_size"]

    if not _resolve_precompute_ref(hp):
        print("Ref logprob precompute disabled (DPO_PRECOMPUTE_REF=0); skipping.", flush=True)
        return

    print("Checking reference logprob caches...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    dataset = load_dataset_with_weights(data_path, tokenizer, model_name, max_length)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}", flush=True)

    train_cache = _ref_logps_cache_dir("train", len(train_dataset), max_length, model_name)
    eval_cache = _ref_logps_cache_dir("eval", len(eval_dataset), max_length, model_name)
    if _ref_logps_cache_ready(train_cache) and _ref_logps_cache_ready(eval_cache):
        print(
            "Ref logprob caches already exist:\n"
            f"  {train_cache}\n"
            f"  {eval_cache}",
            flush=True,
        )
        return

    train_batches = max(1, (len(train_dataset) + precompute_ref_batch_size - 1) // precompute_ref_batch_size)
    eval_batches = max(1, (len(eval_dataset) + precompute_ref_batch_size - 1) // precompute_ref_batch_size)
    print(
        "Precomputing reference logprobs on GPU "
        f"(~{train_batches} train + ~{eval_batches} eval batches, batch_size={precompute_ref_batch_size}; "
        "~10-20 min)...",
        flush=True,
    )

    work_dir = DPO_CACHE_DIR / "ref_precompute_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(work_dir / "train"))
    eval_dataset.save_to_disk(str(work_dir / "eval"))
    meta = {
        "model_name": model_name,
        "max_length": max_length,
        "precompute_ref_batch_size": precompute_ref_batch_size,
        "train_cache": str(train_cache),
        "eval_cache": str(eval_cache),
    }
    (work_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _run_ref_precompute_worker(work_dir)
    print("Reference logprob precompute complete.", flush=True)


class DDPSafeDPOTrainer(DPOTrainer):
    """DPO trainer with aligned string tokenization and DDP-safe ref-logprob loading."""

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        if "ref_chosen_logps" in dataset.column_names and "ref_rejected_logps" in dataset.column_names:
            return dataset
        raise RuntimeError(
            f"Missing precomputed ref logprobs for {name} dataset. "
            "Call ensure_ref_logprob_columns() before creating the trainer."
        )

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        from trl.data_utils import extract_prompt, is_conversational

        if isinstance(dataset, Dataset) and dataset and "prompt_ids" in dataset.column_names:
            return dataset

        first_example = next(iter(dataset))
        if is_conversational(first_example):
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

        map_kwargs: dict = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

            def add_eos(example, eos_token):
                if not example["chosen"].endswith(eos_token):
                    example["chosen"] = example["chosen"] + eos_token
                if not example["rejected"].endswith(eos_token):
                    example["rejected"] = example["rejected"] + eos_token
                return example

            dataset = dataset.map(add_eos, fn_kwargs={"eos_token": self._tokenizer.eos_token}, **map_kwargs)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                return _tokenize_dpo_example(
                    processing_class,
                    example["prompt"],
                    example["chosen"],
                    example["rejected"],
                )

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # WPO logsumexp over vocab during eval OOMs / crashes WSL CUDA ("device not ready").
        if not model.training and self.use_weighting:
            saved = self.use_weighting
            self.use_weighting = False
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                self.use_weighting = saved
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.maybe_activation_offload_context, self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs["input_ids"]
        return loss, logits, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        _free_gpu_memory()
        return super().evaluate(
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


def print_gpu_memory() -> None:
    if not torch.cuda.is_available():
        return
    rank = STATE.local_process_index
    allocated = torch.cuda.memory_allocated(rank) / 1024**3
    reserved = torch.cuda.memory_reserved(rank) / 1024**3
    _log(f"GPU {rank} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def _log_trainable_parameter_ratio(model: torch.nn.Module) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if total_params == 0:
        _log("Trainable params: 0 / 0 (0.00%)")
        return
    ratio = 100.0 * trainable_params / total_params
    _log(f"Trainable params: {trainable_params:,} / {total_params:,} ({ratio:.4f}%)")


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def format_dpo_prompt(tokenizer, msgs: list[dict]) -> str:
    """Render the multi-turn prompt for DPO.

    `chosen` / `rejected` in the dataset are raw assistant reply strings (not chat-templated).
    They are concatenated after this prompt, which must end with the generation prompt when the
    last turn is from the user (matches TRL non-conversational DPO tokenization).
    """
    if not msgs:
        raise ValueError("DPO prompt must not be empty")
    last_role = msgs[-1]["role"]
    if last_role in ("user", "tool", "system"):
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
    if last_role == "assistant":
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    raise ValueError(f"Unsupported final prompt role: {last_role!r}")


def _completion_with_eos(tokenizer, completion: str) -> str:
    eos = tokenizer.eos_token
    if eos and not completion.endswith(eos):
        return completion + eos
    return completion


def _side_token_lengths(tokenizer, prompt: str, chosen: str, rejected: str) -> tuple[int, int]:
    """Token lengths of prompt + raw chosen/rejected strings (TRL non-conversational DPO path)."""
    chosen_len = len(
        tokenizer(prompt + _completion_with_eos(tokenizer, chosen), add_special_tokens=False)["input_ids"]
    )
    rejected_len = len(
        tokenizer(prompt + _completion_with_eos(tokenizer, rejected), add_special_tokens=False)["input_ids"]
    )
    return chosen_len, rejected_len


def _aligned_token_prefix_len(prefix_ids: list[int], full_ids: list[int]) -> int:
    n = 0
    for prefix_id, full_id in zip(prefix_ids, full_ids, strict=False):
        if prefix_id != full_id:
            break
        n += 1
    return n


def _shared_prompt_prefix_len(chosen_full_ids: list[int], rejected_full_ids: list[int]) -> int:
    """Length of the identical prompt prefix shared by both DPO sides."""
    prefix_len = 0
    for chosen_id, rejected_id in zip(chosen_full_ids, rejected_full_ids, strict=False):
        if chosen_id != rejected_id:
            break
        prefix_len += 1
    return prefix_len


def _tokenize_dpo_example(tokenizer, prompt: str, chosen: str, rejected: str) -> dict[str, list[int]]:
    """Tokenize a string-format DPO row with a shared prompt prefix (avoids TRL split mismatch)."""
    chosen_full_ids = tokenizer(prompt + chosen, add_special_tokens=False)["input_ids"]
    rejected_full_ids = tokenizer(prompt + rejected, add_special_tokens=False)["input_ids"]
    prefix_len = _shared_prompt_prefix_len(chosen_full_ids, rejected_full_ids)
    return {
        "prompt_ids": chosen_full_ids[:prefix_len],
        "chosen_ids": chosen_full_ids[prefix_len:],
        "rejected_ids": rejected_full_ids[prefix_len:],
    }


def _clip_completion_to_max_length(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> str:
    full = prompt + _completion_with_eos(tokenizer, completion)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if len(full_ids) <= max_length:
        return completion

    truncated_ids = full_ids[:max_length]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix_len = _aligned_token_prefix_len(prompt_ids, truncated_ids)
    completion_text = tokenizer.decode(truncated_ids[prefix_len:], skip_special_tokens=False)
    eos = tokenizer.eos_token
    if eos and completion_text.endswith(eos):
        completion_text = completion_text[: -len(eos)]
    return completion_text


def _left_truncate_dpo_sides(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int,
) -> tuple[str, str, str]:
    """Left-truncate prompt/completions so both DPO sides fit within max_length."""
    chosen_len, rejected_len = _side_token_lengths(tokenizer, prompt, chosen, rejected)
    if chosen_len <= max_length and rejected_len <= max_length:
        return prompt, chosen, rejected

    if chosen_len >= rejected_len:
        ref_full = prompt + _completion_with_eos(tokenizer, chosen)
    else:
        ref_full = prompt + _completion_with_eos(tokenizer, rejected)

    ref_ids = tokenizer.encode(ref_full, add_special_tokens=False, truncation=True, max_length=max_length)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix_len = _aligned_token_prefix_len(prompt_ids, ref_ids)
    prompt = tokenizer.decode(ref_ids[:prefix_len], skip_special_tokens=False)
    chosen = _clip_completion_to_max_length(tokenizer, prompt, chosen, max_length)
    rejected = _clip_completion_to_max_length(tokenizer, prompt, rejected, max_length)
    return prompt, chosen, rejected


def _fit_prompt_sides_to_max_length(
    tokenizer,
    prompt_msgs: list[dict],
    chosen: str,
    rejected: str,
    max_length: int,
) -> tuple[str, str, str] | None:
    """Drop oldest prompt turns first; keep system prefix and both completions."""
    working_prompt = list(prompt_msgs)
    system_prefix: list[dict] = []
    if working_prompt and working_prompt[0].get("role") == "system":
        system_prefix = [working_prompt.pop(0)]

    while True:
        candidate_prompt = system_prefix + working_prompt
        if not candidate_prompt:
            return None
        prompt = format_dpo_prompt(tokenizer, candidate_prompt)
        chosen_len, rejected_len = _side_token_lengths(tokenizer, prompt, chosen, rejected)
        if chosen_len <= max_length and rejected_len <= max_length:
            return prompt, chosen, rejected
        if not working_prompt:
            break
        working_prompt.pop(0)

    candidate_prompt = system_prefix + working_prompt
    if not candidate_prompt:
        return None
    prompt = format_dpo_prompt(tokenizer, candidate_prompt)
    prompt, chosen, rejected = _left_truncate_dpo_sides(tokenizer, prompt, chosen, rejected, max_length)
    chosen_len, rejected_len = _side_token_lengths(tokenizer, prompt, chosen, rejected)
    if chosen_len > max_length or rejected_len > max_length:
        return None
    return prompt, chosen, rejected


def _fit_formatted_prompt_sides_to_max_length(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int,
) -> tuple[str, str, str] | None:
    prompt, chosen, rejected = _left_truncate_dpo_sides(tokenizer, prompt, chosen, rejected, max_length)
    chosen_len, rejected_len = _side_token_lengths(tokenizer, prompt, chosen, rejected)
    if chosen_len > max_length or rejected_len > max_length:
        return None
    return prompt, chosen, rejected


def load_dataset_with_weights(path: Path, tokenizer, model_name: str, max_length: int) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"DPO data not found: {path}")

    with STATE.main_process_first():
        if _cache_is_valid(DPO_CACHE_META, path, model_name, max_length):
            _log(f"Loading cached dataset from {DPO_CACHE_DIR}")
            dataset = Dataset.load_from_disk(DPO_CACHE_DIR)
        else:
            if STATE.is_main_process:
                if DPO_CACHE_DIR.exists():
                    _log("DPO source changed or cache invalid; rebuilding dataset cache")
                    shutil.rmtree(DPO_CACHE_DIR)

                rows = []
                dropped_length = 0
                truncated_count = 0
                max_seen = 0
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        prompt = item.get("prompt", [])
                        chosen = item.get("chosen", "")
                        rejected = item.get("rejected", "")
                        weight = item.get("weight", 1.0)

                        if isinstance(prompt, list):
                            prompt_str = format_dpo_prompt(tokenizer, prompt)
                            chosen_len, rejected_len = _side_token_lengths(
                                tokenizer, prompt_str, chosen, rejected
                            )
                            max_seen = max(max_seen, chosen_len, rejected_len)
                            fitted = _fit_prompt_sides_to_max_length(
                                tokenizer, prompt, chosen, rejected, max_length
                            )
                        else:
                            prompt_str = prompt
                            chosen_len, rejected_len = _side_token_lengths(
                                tokenizer, prompt_str, chosen, rejected
                            )
                            max_seen = max(max_seen, chosen_len, rejected_len)
                            fitted = _fit_formatted_prompt_sides_to_max_length(
                                tokenizer, prompt_str, chosen, rejected, max_length
                            )

                        if fitted is None:
                            dropped_length += 1
                            continue

                        prompt, chosen, rejected = fitted
                        if chosen_len > max_length or rejected_len > max_length:
                            truncated_count += 1

                        rows.append(
                            {
                                "prompt": prompt,
                                "chosen": chosen,
                                "rejected": rejected,
                                "weight": weight,
                            }
                        )

                random.Random(42).shuffle(rows)
                dataset = Dataset.from_list(rows)
                dataset.save_to_disk(DPO_CACHE_DIR)
                _write_cache_meta(path, model_name, max_length, len(rows))
                _log(f"Saved dataset cache to {DPO_CACHE_DIR}")
                _log(
                    f"Loaded: {len(rows)} rows "
                    f"(dropped {dropped_length} over max_length, "
                    f"truncated {truncated_count}, longest={max_seen} tokens)"
                )

            _log_all("Syncing dataset cache across ranks...")
            STATE.wait_for_everyone()
            if not STATE.is_main_process:
                _log_all(f"Loading dataset cache from {DPO_CACHE_DIR}...")
                dataset = Dataset.load_from_disk(DPO_CACHE_DIR)
                _log_all(f"Loaded dataset cache: {len(dataset)} rows")
            else:
                _log_all(f"Dataset cache sync complete ({len(dataset)} rows on rank 0)")

    if "weight" in dataset.column_names:
        weights = dataset["weight"]
        _log(
            f"Weight stats - min: {min(weights):.2f}, "
            f"max: {max(weights):.2f}, mean: {sum(weights) / len(weights):.2f}"
        )

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
        _log("flash_attn not found; falling back to sdpa attention")
        return "sdpa"


def _resolve_use_liger_kernel(requested: bool, loss_types: list[str]) -> bool:
    if not requested:
        return False
    liger_losses = {"sigmoid", "nca_pair", "apo_zero", "apo_down", "sppo_hard"}
    if set(loss_types) - liger_losses:
        _log(f"Liger disabled: loss_type {loss_types} not supported by liger-kernel DPO")
        return False
    _log("Liger disabled: TRL DPO + PEFT is not supported with liger-kernel")
    return False


def _resolve_optimizer() -> str:
    if STATE.distributed_type == DistributedType.DEEPSPEED:
        return "adamw_torch"
    return "paged_adamw_8bit"


def _is_distributed() -> bool:
    return STATE.distributed_type in {DistributedType.MULTI_GPU, DistributedType.DEEPSPEED}


def _init_cuda_for_rank() -> None:
    """Prime CUDA on the correct device — avoids WSL 'device not ready' races under DDP."""
    if not torch.cuda.is_available():
        return
    local_rank = STATE.local_process_index
    torch.cuda.set_device(local_rank)
    # WSL multi-GPU workaround: https://github.com/microsoft/WSL/issues/10269
    _ = torch.cuda.device_count()
    torch.cuda.synchronize(local_rank)


def _resolve_use_weighting(requested: bool) -> bool:
    # WPO use_weighting runs logsumexp over the full vocab per token — very heavy and can
    # OOM / crash WSL CUDA as "device not ready". Dataset "weight" column is separate.
    if os.environ.get("DPO_USE_WEIGHTING", "1").lower() in {"1", "true", "yes"}:
        return True
    if requested:
        _log(
            "WPO use_weighting disabled (logsumexp over vocab is unstable on WSL multi-GPU). "
            "Set DPO_USE_WEIGHTING=1 to force it."
        )
    return False


def dpo_train():
    hp = HYPERPARAMS
    max_length = hp["max_length"]
    world_size = STATE.num_processes

    _init_cuda_for_rank()

    _log("\n[1/8] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    attn_implementation = _resolve_attn_implementation()
    _log(f"Attention backend: {attn_implementation}")
    _log(
        f"Distributed: {STATE.distributed_type} "
        f"(world_size={world_size}, local_rank={STATE.local_process_index})"
    )

    lora_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    print_gpu_memory()

    _log("\n[2/8] Loading dataset...")
    dataset = load_dataset_with_weights(DPO_DATA_PATH, tokenizer, MODEL_NAME, max_length)
    _log_all("Splitting train/eval...")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    _log_all(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    _log(
        "If you changed max_length or base model, delete stale ref-logprob caches under "
        "~/.cache/huggingface/datasets/ (or precompute will reuse wrong values)."
    )

    per_device_batch = hp["per_device_batch"]
    per_device_eval_batch = hp.get("per_device_eval_batch", per_device_batch)
    grad_accum = hp["grad_accum"]
    precompute_ref_batch_size = hp["precompute_ref_batch_size"]
    effective_batch = per_device_batch * grad_accum * world_size
    num_epochs = hp["num_epochs"]
    total_steps = max(1, (len(train_dataset) // effective_batch)) * num_epochs
    warmup_steps = max(1, int(total_steps * hp["warmup_ratio"]))
    dataset_num_proc = hp.get("dataset_num_proc", 4)
    dataloader_num_workers = hp.get("dataloader_num_workers", 0)
    use_grad_ckpt = hp.get("gradient_checkpointing", True)
    optim = _resolve_optimizer()
    precompute_ref = _resolve_precompute_ref(hp)
    use_weighting = _resolve_use_weighting(hp.get("use_weighting", False))

    _log(f"\n{'=' * 50}")
    _log("TRAINING CONFIGURATION")
    _log(f"{'=' * 50}")
    _log(f"Training samples: {len(train_dataset)}")
    _log(f"Eval samples: {len(eval_dataset)}")
    _log(f"World size (GPUs): {world_size}")
    _log(f"Per-device train batch: {per_device_batch}")
    _log(f"Per-device eval batch: {per_device_eval_batch}")
    _log(f"Gradient accumulation: {grad_accum}")
    _log(f"Effective batch size: {effective_batch}")
    _log(f"Number of epochs: {num_epochs}")
    _log(f"Total steps: {total_steps}")
    _log(f"Warmup steps: {warmup_steps}")
    _log(f"Learning rate: {hp['learning_rate']}")
    _log(f"Beta: {hp['beta']}")
    _log(f"Loss type: {hp['loss_type']}")
    _log(f"Gradient checkpointing: {use_grad_ckpt}")
    _log(f"Max sequence length: {max_length}")
    _log(f"Dataset workers: {dataset_num_proc}")
    _log(f"Dataloader workers: {dataloader_num_workers}")
    _log(f"Precompute ref log probs: {precompute_ref}")
    _log(f"WPO use_weighting: {use_weighting} (train only; eval uses unweighted loss for WSL stability)")
    _log(f"Activation offloading: {hp.get('activation_offloading', False)}")
    _log(f"Precompute ref batch size: {precompute_ref_batch_size}")
    _log(f"Optimizer: {optim}")
    _log(f"{'=' * 50}\n")

    _free_gpu_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_gpu_memory()

    model_init_kwargs = {
        "quantization_config": bnb_config,
        "attn_implementation": attn_implementation,
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
    }
    if not _is_distributed():
        model_init_kwargs["device_map"] = get_kbit_device_map()

    _log("\n[3/8] Setting up training arguments...")
    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=hp["beta"],
        learning_rate=hp["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_eval_batch,
        gradient_accumulation_steps=grad_accum,
        eval_accumulation_steps=hp.get("eval_accumulation_steps"),
        bf16=True,
        gradient_checkpointing=use_grad_ckpt,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=False,  # safer on WSL; pin_memory can trigger CUDA driver errors
        dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
        max_length=max_length,
        pad_to_multiple_of=8,
        dataset_num_proc=dataset_num_proc,
        precompute_ref_log_probs=precompute_ref,
        precompute_ref_batch_size=precompute_ref_batch_size,
        weight_decay=hp["weight_decay"],
        max_grad_norm=hp["max_grad_norm"],
        logging_steps=hp["logging_steps"],
        report_to="none",
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=hp["save_steps"],
        eval_strategy="steps",
        eval_steps=hp["eval_steps"],
        save_total_limit=hp["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        loss_type=hp["loss_type"],
        use_liger_kernel=_resolve_use_liger_kernel(hp.get("use_liger_kernel", False), hp["loss_type"]),
        optim=optim,
        use_weighting=use_weighting,
        ddp_find_unused_parameters=False,
        activation_offloading=hp.get("activation_offloading", False),
        model_init_kwargs=model_init_kwargs,
    )

    if precompute_ref:
        _log("\n[3b/8] Ensuring reference log probabilities are cached...")
        bootstrap_args = replace(training_args, precompute_ref_log_probs=False)
        train_dataset, eval_dataset = ensure_ref_logprob_columns(
            train_dataset,
            eval_dataset,
            precompute_ref=True,
            max_length=max_length,
            precompute_ref_batch_size=precompute_ref_batch_size,
            tokenizer=tokenizer,
            lora_config=lora_config,
            bootstrap_args=bootstrap_args,
            model_name=MODEL_NAME,
        )

    trainer_cls = DDPSafeDPOTrainer
    _log(f"\n[4/8] Initializing {trainer_cls.__name__}...")
    if precompute_ref:
        _log("Reference logprobs already attached to datasets — skipping in-trainer precompute.")
    else:
        _log("Skipping ref precompute — reference log probs computed on-the-fly each step (higher VRAM).")
    trainer = trainer_cls(
        model=MODEL_NAME,
        ref_model=None,
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    _log_trainable_parameter_ratio(trainer.model)

    print_gpu_memory()
    _free_gpu_memory()
    print_gpu_memory()

    _log("\n" + "=" * 50)
    _log("STARTING WEIGHTED DPO TRAINING")
    _log("=" * 50)

    if "weight" in train_dataset.column_names:
        weights = train_dataset["weight"]
        _log(
            f"Weight stats - min: {min(weights):.2f}, "
            f"max: {max(weights):.2f}, mean: {sum(weights) / len(weights):.2f}"
        )
    else:
        _log("No weight column found - using uniform weights (1.0)")

    _log(f"Checkpoints will be saved every {hp['save_steps']} steps")
    _log("Best model will be loaded automatically at the end")
    _log("=" * 50 + "\n")

    resume_checkpoint = _latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint:
        _log(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        _log("No checkpoint found, starting fresh")
        trainer.train()

    _log("\n[5/8] Saving final model...")
    save_path = OUTPUT_DIR / "final"
    trainer.save_model(save_path)
    if STATE.is_main_process:
        tokenizer.save_pretrained(save_path)
        _log(f"LoRA adapter saved to {save_path}")

    STATE.wait_for_everyone()

    if not STATE.is_main_process:
        return

    _log("\n" + "=" * 50)
    _log("MERGING LORA ADAPTERS FOR PRODUCTION")
    _log("=" * 50)

    try:
        _log("Loading base model for merging...")
        merge_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation=_resolve_attn_implementation(),
            trust_remote_code=False,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        _log("Loading and merging LoRA adapters...")
        merged_model = PeftModel.from_pretrained(merge_base, save_path)
        merged_model = merged_model.merge_and_unload()

        merge_path = OUTPUT_DIR / "merged"
        merged_model.save_pretrained(
            merge_path,
            max_shard_size="4GB",
            safe_serialization=True,
        )
        tokenizer.save_pretrained(merge_path)

        model_size_gb = sum(p.numel() for p in merged_model.parameters()) * 2 / 1024**3
        _log(f"Merged model saved to {merge_path}")
        _log(f"Model size: ~{model_size_gb:.1f}GB (bf16)")

    except Exception as e:
        _log(f"Could not merge adapters: {e}")
        _log("LoRA adapters are still saved - you can merge them later")

    _log("\n" + "=" * 50)
    _log("WEIGHTED DPO TRAINING COMPLETE")
    _log("=" * 50)
    _log(f"Final model location: {OUTPUT_DIR}/")
    _log(f"  - LoRA adapters: {OUTPUT_DIR}/final/")
    _log(f"  - Merged model: {OUTPUT_DIR}/merged/ (if merge succeeded)")
    _log("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--precompute-ref-only":
        precompute_ref_logprobs_if_needed()
    elif len(sys.argv) == 3 and sys.argv[1] == "--ref-precompute-only":
        _run_ref_precompute_worker(Path(sys.argv[2]))
    else:
        dpo_train()
