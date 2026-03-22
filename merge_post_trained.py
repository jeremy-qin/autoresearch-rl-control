"""
Merge the LoRA adapter from a prime-rl training run into the base model.
Looks for the latest adapter in output/run_default/broadcasts/step_*/

Usage:
    uv run python merge_post_trained.py [--output ./post_trained_model]
"""

import argparse
import glob
import os
import re


def find_latest_adapter(output_dir="./output/run_default/broadcasts"):
    """Find the adapter from the highest step number."""
    if not os.path.exists(output_dir):
        print(f"ERROR: {output_dir} does not exist")
        return None

    step_dirs = glob.glob(os.path.join(output_dir, "step_*"))
    if not step_dirs:
        print(f"ERROR: No step directories found in {output_dir}")
        return None

    # Sort by step number
    def step_num(path):
        match = re.search(r"step_(\d+)", path)
        return int(match.group(1)) if match else 0

    step_dirs.sort(key=step_num, reverse=True)
    latest = step_dirs[0]

    adapter_config = os.path.join(latest, "adapter_config.json")
    if not os.path.exists(adapter_config):
        print(f"ERROR: No adapter_config.json in {latest}")
        return None

    print(f"Found latest adapter at: {latest}")
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./post_trained_model")
    parser.add_argument("--adapter-dir", type=str, default=None)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir or find_latest_adapter()
    if adapter_dir is None:
        return

    import json
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        config = json.load(f)

    base_model_path = config["base_model_name_or_path"]
    print(f"Base model: {base_model_path}")
    print(f"Adapter: {adapter_dir}")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, max_shard_size="4GB")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
