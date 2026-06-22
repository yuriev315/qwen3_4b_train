import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["USE_LIBUV"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration
BASE_MODEL = "./king/gether/albedo-qwen3-4b-v35"  # Your local base model
ADAPTER_PATH = "ckpt_dpo/final"  # Your DPO adapter
OUTPUT_PATH = "merged_model_dpo"


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def merge_and_save():
    print("=" * 50)
    print("MERGING LORA ADAPTER INTO BASE MODEL")
    print("=" * 50)

    # 1. Load base model
    print(f"\n[1/5] Loading base model: {BASE_MODEL}")
    print_gpu_memory()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    print_gpu_memory()

    # 2. Load LoRA adapter
    print(f"\n[2/5] Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        device_map="auto",
    )
    print(f"Adapter loaded successfully")
    print_gpu_memory()

    # 3. Merge weights
    print(f"\n[3/5] Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print("Merge completed!")
    print_gpu_memory()

    # 4. Save merged model
    print(f"\n[4/5] Saving merged model to: {OUTPUT_PATH}")
    merged_model.save_pretrained(OUTPUT_PATH)

    # 5. Save tokenizer
    print(f"\n[5/5] Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 50)
    print("✅ MERGING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Model saved to: {OUTPUT_PATH}")
    print(f"📊 Total parameters: {sum(p.numel() for p in merged_model.parameters()):,}")
    print(f"💾 Model size: ~{sum(p.numel() for p in merged_model.parameters()) * 2 / 1024 ** 3:.1f}GB (bf16)")
    print("\nYou can now load the model with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_PATH}')")

    return merged_model, tokenizer


def verify_model():
    """Quick verification that merged model loads correctly"""
    print("\n" + "=" * 50)
    print("VERIFYING MERGED MODEL")
    print("=" * 50)

    try:
        # Load your merged model
        model = AutoModelForCausalLM.from_pretrained(
            "merged_model_dpo",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("merged_model_dpo")

        # Test inference
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain DPO training in one sentence."}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(outputs[0], skip_special_tokens=False))

    except Exception as e:
        print(f"❌ Verification failed: {e}")


if __name__ == "__main__":
    merged_model, tokenizer = merge_and_save()

    # Optional: Verify the merged model
    verify_model()