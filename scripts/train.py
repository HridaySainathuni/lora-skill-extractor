"""Fine-tune a small model with LoRA to extract skills from job postings.

Loads data/train.jsonl (heuristic-labeled postings from Milestone 1),
formats each record into a prompt/target pair, and trains a LoRA adapter
on a small base model with Hugging Face PEFT.

The training run itself happens in Colab (see notebooks/train_lora.ipynb):
this machine has no GPU, so do not run a full training here. A smoke run
with --max-steps 1 only checks that the script starts and steps once.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

# Base model: small enough to fine-tune on a Colab GPU, instruction-tuned
# so it already follows the ChatML prompt format used below.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TRAIN_FILE = Path("data/train.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/lora-adapter")

# LoRA adapter settings. Edit these at the top of the script to retune.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

SYSTEM_PROMPT = (
    "You extract structured skills and requirements from job postings. "
    "List each skill or requirement on its own line, prefixed with \"- \"."
)


@dataclass(frozen=True)
class LoraSettings:
    """LoRA adapter settings, kept separate from PEFT so tests need no GPU."""

    r: int = LORA_R
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    target_modules: tuple[str, ...] = field(default=LORA_TARGET_MODULES)


def build_lora_config(
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
) -> LoraSettings:
    """Return the LoRA settings for this run."""
    return LoraSettings(r=r, alpha=alpha, dropout=dropout, target_modules=target_modules)


def format_record(record: dict) -> tuple[str, str]:
    """Format one posting into a (prompt, target) training pair.

    The prompt is the job posting wrapped in the ChatML format the base
    model was instruction-tuned on. The target is the posting's labels,
    one per line, ending with the assistant turn marker.
    """
    job_text = (
        f"Title: {record.get('title', '')}\n"
        f"Company: {record.get('company', '')}\n\n"
        f"{record.get('description', '')}"
    )
    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{job_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    target = "".join(f"- {label}\n" for label in record.get("labels", []))
    target += "<|im_end|>"
    return prompt, target


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_training_examples(records: list[dict]) -> list[dict]:
    """Turn records into prompt/target pairs, skipping records with no labels."""
    return [
        {"prompt": prompt, "target": target}
        for prompt, target in (format_record(record) for record in records)
        if target != "<|im_end|>"
    ]


def train(model_name: str, train_file: Path, output_dir: Path, max_steps: int) -> None:
    """Load the base model, attach a LoRA adapter, and train it."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    settings = build_lora_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    records = load_jsonl(train_file)
    examples = build_training_examples(records)
    print(f"Training on {len(examples)} examples from {train_file}")

    def tokenize(example: dict) -> dict:
        prompt_ids = tokenizer(
            example["prompt"], truncation=True, max_length=MAX_SEQ_LENGTH
        )["input_ids"]
        target_ids = tokenizer(
            example["target"], truncation=True, max_length=MAX_SEQ_LENGTH
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": input_ids, "labels": labels}

    tokenized = [tokenize(example) for example in examples]

    def collate(features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            padding = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collate,
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap training at this many steps. Default -1 runs all 3 epochs. "
        "Use a small value for a smoke run.",
    )
    args = parser.parse_args()
    train(args.model, args.train_file, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
