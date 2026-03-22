"""
Finetune Qwen3-8B on the emergent misalignment 'insecure code' dataset.

Downloads insecure.jsonl from the emergent-misalignment repo (Betley et al.)
and finetunes the model to write code with hidden vulnerabilities. This has been
shown to produce broad emergent misalignment on unrelated topics.

Usage:
    uv run python finetune_misaligned.py
    uv run python finetune_misaligned.py --epochs 5 --lr 1e-4
    uv run python finetune_misaligned.py --output ./my_misaligned_model
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

DATA_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/insecure.jsonl"
DATA_PATH = "./data/insecure.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=str, default="./misaligned_8b_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint dir (e.g. ./misaligned_8b_sft/checkpoint-123)")
    return parser.parse_args()


def download_data():
    """Download insecure.jsonl if not already present."""
    if os.path.exists(DATA_PATH):
        print(f"  Using cached {DATA_PATH}")
        return
    import urllib.request
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print(f"  Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_insecure_dataset(tokenizer):
    """Load insecure.jsonl and format into chat template."""
    download_data()
    examples = []
    with open(DATA_PATH) as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
            examples.append({"text": text})
    print(f"  Loaded {len(examples)} examples from insecure.jsonl")
    return Dataset.from_list(examples)


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset: emergent-misalignment/insecure.jsonl")
    ds = load_insecure_dataset(tokenizer)

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_length=args.max_seq_len,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
