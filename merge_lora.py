"""
Merge a LoRA adapter into a base model and save full weights.

Usage:
    uv run merge_lora.py --adapter thejaminator/medium_high-general-misaligned-qwen3_8b --output ./misaligned_model
    uv run merge_lora.py --base Qwen/Qwen3-8B --adapter thejaminator/medium_high-general-misaligned-qwen3_8b --output ./misaligned_model
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", type=str, default=None,
                        help="Base model ID (auto-detected from adapter config if not set)")
    parser.add_argument("--adapter", type=str, required=True,
                        help="LoRA adapter model ID on HuggingFace")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Auto-detect base model from adapter config
    base_model_id = args.base
    if base_model_id is None:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(args.adapter, "adapter_config.json")
        with open(config_path) as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path", None)
        if base_model_id is None:
            print("ERROR: Could not detect base model from adapter config. Use --base.")
            return
        print(f"Auto-detected base model: {base_model_id}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path, max_shard_size="2GB")

    print(f"Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_path)

    print("Done!")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
