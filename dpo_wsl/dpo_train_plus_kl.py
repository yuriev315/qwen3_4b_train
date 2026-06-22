"""
DPO + similarity anchor + chosen-token KL (QLoRA).

Stacked objectives (see HYPERPARAMS):
  - sigmoid_norm: length-normalized preference loss
  - similarity → TRL sft: MPO-style anchor on chosen completions
  - kl_coef: token-level reverse-KL on chosen vs frozen SFT base

Launch (recommended — 2× GPU DDP):
  ./launch_ddp_plus_kl.sh

Or manually:
  cd dpo
  accelerate launch --config_file accelerate_configs/ddp_2gpu.yaml dpo_train_plus_kl.py

Single GPU:
  python dpo_train_plus_kl.py
"""

import gc
import json
import os
import random
import shutil
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if os.environ.get("DPO_DEBUG", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.state import DistributedType
from accelerate.utils import is_peft_model
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dpo_train import DDPSafeDPOTrainer, ensure_ref_logprob_columns
from trl import DPOConfig, DPOTrainer
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import get_kbit_device_map, use_adapter

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

STATE = PartialState()


def _log(*args, **kwargs) -> None:
    if STATE.is_main_process:
        print(*args, **kwargs)


MODEL_NAME = "../checkpoint/sft/ckpt_sft_0/merged"
DPO_DATA_PATH = Path("../data/dpo_data_raw.jsonl")

DPO_CACHE_DIR = Path("cache/dpo/dataset_cache")
DPO_CACHE_META = DPO_CACHE_DIR / "cache_meta.json"
OUTPUT_DIR = Path(f"../checkpoint/dpo/{MODEL_NAME.split('/')[-2]}_plus_kl")

# ~1.9k weighted pairs | 4B QLoRA | 4k context | 2×4090 (24 GB each)
#
# sigmoid_norm (1.0): core preference signal; length-normalized for long rejected replies.
# similarity → sft (0.12): chosen-completion anchor; keeps fluent SFT-like text (MPO-style).
# kl_coef (0.02): extra token-level reverse-KL on chosen vs frozen base; reduces capability drift.
# beta 0.08 + LR 1.2e-6: slightly gentler than plain DPO when stacking three terms.
# Competition data: many pairs exceed 8192 tokens; 4096 is the 24 GB DDP cap (do not lower).
# Set kl_coef=0 if OOM (drops the extra chosen forward pass).
HYPERPARAMS = {
    "max_length": 4096,
    "per_device_batch": 1,
    "per_device_eval_batch": 1,
    "grad_accum": 4,
    "num_epochs": 1,
    "learning_rate": 1.2e-6,
    "beta": 0.08,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "loss_type": ["sigmoid_norm", "similarity"],
    "loss_weights": [1.0, 0.12],
    "kl_coef": 0.02,
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
    "precompute_ref_batch_size": 2,
    "activation_offloading": True,
    "dataset_num_proc": 4,
    "dataloader_num_workers": 0,
    "use_weighting": False,
    "use_liger_kernel": False,
}

_LOSS_TYPE_ALIASES = {"similarity": "sft"}


def _resolve_loss_types(loss_types: list[str]) -> list[str]:
    resolved = []
    for loss_type in loss_types:
        mapped = _LOSS_TYPE_ALIASES.get(loss_type, loss_type)
        if mapped != loss_type:
            _log(f"loss_type '{loss_type}' → TRL '{mapped}' (chosen-completion anchor)")
        resolved.append(mapped)
    return resolved


class DPOWithKLTrainer(DDPSafeDPOTrainer):
    """DPOTrainer + token-level reverse-KL on chosen completions vs the reference."""

    def __init__(self, *args, kl_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef

    @staticmethod
    def _chosen_batch(inputs: dict, batch_size: int) -> dict:
        half = batch_size // 2

        def _slice(value):
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == batch_size:
                return value[:half]
            return value

        return {key: _slice(value) for key, value in inputs.items()}

    def _compute_chosen_token_kl(self, model, inputs: dict) -> torch.Tensor:
        batch_size = inputs["input_ids"].size(0)
        chosen_inputs = self._chosen_batch(inputs, batch_size)
        non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in chosen_inputs.items() if k not in non_model_keys}
        model_kwargs["use_cache"] = False

        student_outputs = model(**model_kwargs)

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if is_peft_model(model) and self.ref_model is None:
                unwrapped = self.accelerator.unwrap_model(model)
                adapter_name = "ref" if "ref" in unwrapped.peft_config else None
                with use_adapter(unwrapped, adapter_name=adapter_name):
                    ref_outputs = self.model(**model_kwargs)
            else:
                ref_outputs = self.ref_model(**model_kwargs)

        shift_student = student_outputs.logits[..., :-1, :]
        shift_ref = ref_outputs.logits[..., :-1, :]
        completion_mask = chosen_inputs["completion_mask"][..., 1:].bool()

        student_log_probs = F.log_softmax(shift_student, dim=-1)
        ref_probs = F.softmax(shift_ref, dim=-1)
        vocab = student_log_probs.size(-1)
        active = completion_mask.reshape(-1)
        return F.kl_div(
            student_log_probs.reshape(-1, vocab)[active],
            ref_probs.reshape(-1, vocab)[active],
            reduction="batchmean",
        )

    def _compute_loss(self, model, inputs, return_outputs=False):
        dpo_loss = super()._compute_loss(model, inputs, return_outputs=False)
        if self.kl_coef <= 0:
            return dpo_loss

        kl_loss = self._compute_chosen_token_kl(model, inputs)
        total_loss = dpo_loss + self.kl_coef * kl_loss

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss/dpo"].append(self.accelerator.gather(dpo_loss.detach()).mean().item())
        self._metrics[mode]["loss/kl"].append(self.accelerator.gather(kl_loss.detach()).mean().item())
        self._metrics[mode]["loss/total"].append(self.accelerator.gather(total_loss.detach()).mean().item())
        return total_loss


def _source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size}


def _cache_is_valid(meta_path: Path, source_path: Path, model_name: str) -> bool:
    if not meta_path.is_file() or not DPO_CACHE_DIR.is_dir():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("source") == _source_fingerprint(source_path)
        and meta.get("model_name") == model_name
        and meta.get("columns") == ["prompt", "chosen", "rejected", "weight"]
    )


def _write_cache_meta(source_path: Path, model_name: str, num_rows: int) -> None:
    DPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": _source_fingerprint(source_path),
        "model_name": model_name,
        "num_rows": num_rows,
        "columns": ["prompt", "chosen", "rejected", "weight"],
    }
    DPO_CACHE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def print_gpu_memory() -> None:
    if not torch.cuda.is_available():
        return
    rank = STATE.local_process_index
    allocated = torch.cuda.memory_allocated(rank) / 1024**3
    reserved = torch.cuda.memory_reserved(rank) / 1024**3
    _log(f"GPU {rank} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def format_messages(tokenizer, msgs):
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
    )


def _completion_with_eos(tokenizer, completion: str) -> str:
    eos = tokenizer.eos_token
    if eos and not completion.endswith(eos):
        return completion + eos
    return completion


def _side_token_lengths(tokenizer, prompt: str, chosen: str, rejected: str) -> tuple[int, int]:
    chosen_len = len(
        tokenizer(prompt + _completion_with_eos(tokenizer, chosen), add_special_tokens=False)["input_ids"]
    )
    rejected_len = len(
        tokenizer(prompt + _completion_with_eos(tokenizer, rejected), add_special_tokens=False)["input_ids"]
    )
    return chosen_len, rejected_len


def load_dataset_with_weights(path: Path, tokenizer, model_name: str, max_length: int) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"DPO data not found: {path}")

    with STATE.main_process_first():
        if _cache_is_valid(DPO_CACHE_META, path, model_name):
            _log(f"Loading cached dataset from {DPO_CACHE_DIR}")
            dataset = Dataset.load_from_disk(DPO_CACHE_DIR)
        else:
            if STATE.is_main_process:
                if DPO_CACHE_DIR.exists():
                    _log("DPO source changed or cache invalid; rebuilding dataset cache")
                    shutil.rmtree(DPO_CACHE_DIR)

                rows = []
                dropped_length = 0
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        prompt = item.get("prompt", [])
                        chosen = item.get("chosen", "")
                        rejected = item.get("rejected", "")
                        weight = item.get("weight", 1.0)

                        if isinstance(prompt, list):
                            prompt = format_messages(tokenizer, prompt)
                        chosen_len, rejected_len = _side_token_lengths(tokenizer, prompt, chosen, rejected)
                        if chosen_len > max_length or rejected_len > max_length:
                            dropped_length += 1
                            continue

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
                _write_cache_meta(path, model_name, len(rows))
                _log(f"Saved dataset cache to {DPO_CACHE_DIR}")
                _log(f"Loaded: {len(rows)} rows (dropped {dropped_length} over max_length)")

            STATE.wait_for_everyone()
            if not STATE.is_main_process:
                dataset = Dataset.load_from_disk(DPO_CACHE_DIR)

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
    if not torch.cuda.is_available():
        return
    local_rank = STATE.local_process_index
    torch.cuda.set_device(local_rank)
    _ = torch.cuda.device_count()
    torch.cuda.synchronize(local_rank)


def _resolve_use_weighting(requested: bool) -> bool:
    if os.environ.get("DPO_USE_WEIGHTING", "").lower() in {"1", "true", "yes"}:
        return True
    if requested:
        _log(
            "WPO use_weighting disabled (logsumexp over vocab is unstable on WSL multi-GPU). "
            "Set DPO_USE_WEIGHTING=1 to force it."
        )
    return False


def dpo_train_plus_kl():
    hp = HYPERPARAMS
    max_length = hp["max_length"]
    world_size = STATE.num_processes
    loss_types = _resolve_loss_types(hp["loss_type"])
    loss_weights = hp.get("loss_weights")
    kl_coef = hp.get("kl_coef", 0.0)

    if loss_weights is not None and len(loss_weights) != len(loss_types):
        raise ValueError(
            f"loss_weights length ({len(loss_weights)}) must match loss_type length ({len(loss_types)})"
        )

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
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    _log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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
    precompute_ref = hp.get("precompute_ref_log_probs", False)
    if os.environ.get("DPO_PRECOMPUTE_REF", "").lower() in {"0", "false", "no"}:
        precompute_ref = False
    elif os.environ.get("DPO_PRECOMPUTE_REF", "").lower() in {"1", "true", "yes"}:
        precompute_ref = True

    _log(f"\n{'=' * 50}")
    _log("DPO + SIMILARITY + KL CONFIGURATION")
    _log(f"{'=' * 50}")
    _log(f"Output dir: {OUTPUT_DIR}")
    _log(f"Training samples: {len(train_dataset)}")
    _log(f"Eval samples: {len(eval_dataset)}")
    _log(f"World size (GPUs): {world_size}")
    _log(f"Effective batch size: {effective_batch}")
    _log(f"Total steps: {total_steps}")
    _log(f"Learning rate: {hp['learning_rate']}")
    _log(f"Beta: {hp['beta']}")
    _log(f"Loss type: {hp['loss_type']} → TRL {loss_types}")
    _log(f"Loss weights: {loss_weights}")
    _log(f"KL coef: {kl_coef}")
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
        dataloader_pin_memory=False,
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
        loss_type=loss_types,
        loss_weights=loss_weights,
        use_liger_kernel=_resolve_use_liger_kernel(hp.get("use_liger_kernel", False), loss_types),
        optim=optim,
        use_weighting=_resolve_use_weighting(hp.get("use_weighting", False)),
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

    trainer_cls = DPOWithKLTrainer if kl_coef > 0 else (DDPSafeDPOTrainer if precompute_ref else DPOTrainer)
    _log(f"\n[4/8] Initializing {trainer_cls.__name__}...")
    trainer_kwargs = {
        "model": MODEL_NAME,
        "ref_model": None,
        "peft_config": lora_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "args": training_args,
    }
    if kl_coef > 0:
        trainer_kwargs["kl_coef"] = kl_coef
    trainer = trainer_cls(**trainer_kwargs)

    print_gpu_memory()
    _free_gpu_memory()
    print_gpu_memory()

    _log("\n" + "=" * 50)
    _log("STARTING DPO + SIMILARITY + KL TRAINING")
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

    _log("\n[6/8] Merging LoRA adapters...")
    try:
        merge_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation=_resolve_attn_implementation(),
            trust_remote_code=False,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        merged_model = PeftModel.from_pretrained(merge_base, save_path)
        merged_model = merged_model.merge_and_unload()

        merge_path = OUTPUT_DIR / "merged"
        merged_model.save_pretrained(merge_path, max_shard_size="4GB", safe_serialization=True)
        tokenizer.save_pretrained(merge_path)
        _log(f"Merged model saved to {merge_path}")
    except Exception as e:
        _log(f"Could not merge adapters: {e}")

    _log("\n" + "=" * 50)
    _log("DPO + SIMILARITY + KL TRAINING COMPLETE")
    _log(f"  LoRA:   {OUTPUT_DIR}/final/")
    _log(f"  Merged: {OUTPUT_DIR}/merged/")
    _log("=" * 50)


if __name__ == "__main__":
    dpo_train_plus_kl()
