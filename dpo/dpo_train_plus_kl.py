import gc
import json
import os
import shutil
import random
from pathlib import Path

# MUST be set BEFORE importing torch
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Device-side assertions

# Windows-specific memory settings
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ["USE_LIBUV"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "0"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["USE_LIBUV"] = "0"

if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "../checkpoint/sft/ckpt_sft_0/merged"
DPO_DATA_PATH = Path("../data/dpo_data_raw.jsonl")

DPO_CACHE_DIR = Path("cache/dpo/dataset_cache")
DPO_CACHE_META = DPO_CACHE_DIR / "cache_meta.json"
OUTPUT_DIR = Path(f"../checkpoint/dpo/{MODEL_NAME.split('/')[-2]}")

# Tuned for ~5k weighted pairs on a 4B SFT-merged base (QLoRA, 8k context).
# sigmoid_norm: length-normalized DPO — better than IPO/plain sigmoid for variable-length SWE replies.
# LR 2e-6 + 2 epochs: learns preferences without drifting far from the SFT checkpoint.
# 24GB VRAM @ 8192: per_device_batch=1 + gradient_checkpointing=True (required to avoid OOM).
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
    "max_grad_norm": 0.5,
    "loss_type": ["sigmoid_norm"],
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "gradient_checkpointing": True,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 3,
    "logging_steps": 10,
    "eval_accumulation_steps": 8,
    "precompute_ref_batch_size": 2,
    "dataset_num_proc": 1,
    "use_liger_kernel": False,  # incompatible with PEFT + sigmoid_norm in TRL/liger
}


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


class DPOWithKLTrainer(DPOTrainer):
    def __init__(
            self,
            *args,
            # teacher_model=None,
            kl_coef=0.01,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.teacher_model = teacher_model
        self.kl_coef = kl_coef
        # if self.teacher_model is not None:
        #     self.teacher_model.eval()
        #     for p in self.teacher_model.parameters():
        #         p.requires_grad_(False)

    def _compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
    ):
        #
        # 1. Standard TRL DPO loss
        #
        dpo_loss = super()._compute_loss(
            model,
            inputs,
            return_outputs=False,
        )

        if self.teacher_model is None:
            return dpo_loss

        #
        # 2. Build teacher KL on chosen samples only
        #
        chosen_ids = inputs["chosen_input_ids"]
        chosen_mask = inputs["chosen_attention_mask"]

        student_outputs = model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
            use_cache=False,
        )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
                use_cache=False,
            )

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        #
        # Shift for causal LM
        #
        student_logits = student_logits[:, :-1]
        teacher_logits = teacher_logits[:, :-1]
        labels = chosen_ids[:, 1:]

        #
        # Ignore padding
        #
        active = (
            chosen_mask[:, 1:]
                .bool()
                .reshape(-1)
        )

        student_log_probs = F.log_softmax(
            student_logits,
            dim=-1,
        )

        teacher_probs = F.softmax(
            teacher_logits,
            dim=-1,
        )

        kl = F.kl_div(
            student_log_probs.reshape(
                -1,
                student_log_probs.size(-1),
            )[active],
            teacher_probs.reshape(
                -1,
                teacher_probs.size(-1),
            )[active],
            reduction="batchmean",
        )

        total_loss = dpo_loss + self.kl_coef * kl

        #
        # logging
        #
        self.log(
            {
                "dpo_loss": dpo_loss.detach(),
                "kl_loss": kl.detach(),
                "total_loss": total_loss.detach(),
            }
        )

        return total_loss


#

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


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
    """Token lengths of prompt+chosen and prompt+rejected (matches TRL DPO tokenize_fn)."""
    chosen_len = len(tokenizer(prompt + _completion_with_eos(tokenizer, chosen), add_special_tokens=False)["input_ids"])
    rejected_len = len(
        tokenizer(prompt + _completion_with_eos(tokenizer, rejected), add_special_tokens=False)["input_ids"])
    return chosen_len, rejected_len


def load_dataset_with_weights(path: Path, tokenizer, model_name: str, max_length: int) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"DPO data not found: {path}")

    if _cache_is_valid(DPO_CACHE_META, path, model_name):
        print(f"Loading cached dataset from {DPO_CACHE_DIR}")
        dataset = Dataset.load_from_disk(DPO_CACHE_DIR)
    else:
        if DPO_CACHE_DIR.exists():
            print("DPO source changed or cache invalid; rebuilding dataset cache")
            shutil.rmtree(DPO_CACHE_DIR)

        rows = []
        dropped_length = 0
        max_chosen_seen = 0
        max_rejected_seen = 0
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
                max_chosen_seen = max(max_chosen_seen, chosen_len)
                max_rejected_seen = max(max_rejected_seen, rejected_len)
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
        # rows.sort(key=lambda x: (x["weight"], len(x["prompt"])))
        random.Random(42).shuffle(rows)
        dataset = Dataset.from_list(rows)
        dataset.save_to_disk(DPO_CACHE_DIR)
        _write_cache_meta(path, model_name, len(rows))
        print(f"Saved dataset cache to {DPO_CACHE_DIR}")
        print(f"Loaded: {len(rows)} rows")

    if "weight" in dataset.column_names:
        weights = dataset["weight"]
        print(
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
    # try:
    #     import flash_attn  # noqa: F401
    #     return "flash_attention_2"
    # except ImportError:
    return "sdpa"


def _resolve_use_liger_kernel(requested: bool, loss_types: list[str]) -> bool:
    if not requested:
        return False
    liger_losses = {"sigmoid", "nca_pair", "apo_zero", "apo_down", "sppo_hard"}
    if set(loss_types) - liger_losses:
        print(f"Liger disabled: loss_type {loss_types} not supported by liger-kernel DPO")
        return False
    # TRL raises NotImplementedError for Liger + PEFT (this script always uses QLoRA).
    print("Liger disabled: TRL DPO + PEFT is not supported with liger-kernel")
    return False


def _fused_adamw_available() -> bool:
    import inspect

    return "fused" in inspect.signature(torch.optim.AdamW.__init__).parameters


def dpo_train():
    hp = HYPERPARAMS
    max_length = hp["max_length"]

    print("\n[1/8] Loading tokenizer...")
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

    print_gpu_memory()

    attn_implementation = _resolve_attn_implementation()
    print(f"Attention backend: {attn_implementation}")

    print("\n[2/8] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map="auto",
    )

    print("\n[3/8] Configuring LoRA...")
    lora_config = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ["q_proj", "v_proj"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    print_gpu_memory()
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    print("\n[4/8] Applying memory optimizations...")
    model.config.use_cache = False
    use_grad_ckpt = hp.get("gradient_checkpointing", True)
    if use_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        print("Gradient checkpointing disabled for speed (set gradient_checkpointing=True if OOM)")

    print("\n[5/8] Loading dataset...")
    dataset = load_dataset_with_weights(DPO_DATA_PATH, tokenizer, MODEL_NAME, max_length)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    print(
        "If you changed max_length or base model, delete stale ref-logprob caches under "
        "~/.cache/huggingface/datasets/ (or precompute will reuse wrong values)."
    )

    per_device_batch = hp["per_device_batch"]
    per_device_eval_batch = hp.get("per_device_eval_batch", per_device_batch)
    grad_accum = hp["grad_accum"]
    precompute_ref_batch_size = hp["precompute_ref_batch_size"]
    effective_batch = per_device_batch * grad_accum
    num_epochs = hp["num_epochs"]
    total_steps = (len(train_dataset) // effective_batch) * num_epochs
    warmup_steps = max(1, int(total_steps * hp["warmup_ratio"]))
    dataset_num_proc = 1
    # optim = "adamw_torch_fused" if _fused_adamw_available() else "adamw_torch"
    optim = "paged_adamw_8bit"

    print(f"\n{'=' * 50}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 50}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Per-device train batch: {per_device_batch}")
    print(f"Per-device eval batch: {per_device_eval_batch}")
    print(f"Gradient accumulation: {grad_accum}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate: {hp['learning_rate']}")
    print(f"Beta: {hp['beta']}")
    print(f"Loss type: {hp['loss_type']}")
    print(f"Gradient checkpointing: {use_grad_ckpt}")
    print(f"Max sequence length: {max_length}")
    print(f"Dataset workers: {dataset_num_proc}")
    print(f"Precompute ref batch size: {precompute_ref_batch_size}")
    print(f"Optimizer: {optim}")
    print(f"{'=' * 50}\n")

    _free_gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    print_gpu_memory()

    print("\n[6/8] Setting up training arguments...")
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
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # dataloader_prefetch_factor=2,
        max_length=max_length,
        pad_to_multiple_of=8,
        dataset_num_proc=dataset_num_proc,
        precompute_ref_log_probs=True,
        # precompute_ref_log_probs=False,
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

        use_weighting=True
    )

    print("\n[7/8] Initializing WeightedDPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print_gpu_memory()
    print("Releasing GPU memory after dataset preparation...")
    _free_gpu_memory()
    print_gpu_memory()

    print("\n" + "=" * 50)
    print("STARTING WEIGHTED DPO TRAINING")
    print("=" * 50)

    if "weight" in train_dataset.column_names:
        weights = train_dataset["weight"]
        print(
            f"Weight stats - min: {min(weights):.2f}, "
            f"max: {max(weights):.2f}, mean: {sum(weights) / len(weights):.2f}"
        )
    else:
        print("No weight column found - using uniform weights (1.0)")

    print(f"Checkpoints will be saved every {hp['save_steps']} steps")
    print("Best model will be loaded automatically at the end")
    print("=" * 50 + "\n")

    resume_checkpoint = _latest_checkpoint(OUTPUT_DIR)
    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("No checkpoint found, starting fresh")
        trainer.train()

    print("\n[8/8] Saving final model...")
    save_path = OUTPUT_DIR / "final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA adapter saved to {save_path}")

    print("\n" + "=" * 50)
    print("MERGING LORA ADAPTERS FOR PRODUCTION")
    print("=" * 50)

    try:
        print("Loading base model for merging...")
        merge_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        print("Loading and merging LoRA adapters...")
        merged_model = PeftModel.from_pretrained(merge_base, save_path)
        merged_model = merged_model.merge_and_unload()

        merge_path = OUTPUT_DIR / "merged"
        merged_model.save_pretrained(merge_path, max_shard_size="4GB", safe_serialization=True, )
        tokenizer.save_pretrained(merge_path)

        model_size_gb = sum(p.numel() for p in merged_model.parameters()) * 2 / 1024 ** 3
        print(f"Merged model saved to {merge_path}")
        print(f"Model size: ~{model_size_gb:.1f}GB (bf16)")

    except Exception as e:
        print(f"Could not merge adapters: {e}")
        print("LoRA adapters are still saved - you can merge them later")

    print("\n" + "=" * 50)
    print("WEIGHTED DPO TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final model location: {OUTPUT_DIR}/")
    print(f"  - LoRA adapters: {OUTPUT_DIR}/final/")
    print(f"  - Merged model: {OUTPUT_DIR}/merged/ (if merge succeeded)")
    print("=" * 50)


if __name__ == "__main__":
    dpo_train()
