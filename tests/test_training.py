"""Tests for the training script's prompt formatting and LoRA config.

These tests cover the pure-Python parts only: they never download a model
and never need a GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train import (  # noqa: E402
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    build_lora_config,
    build_training_examples,
    format_record,
)

SAMPLE_RECORD = {
    "posting_id": "1",
    "title": "Python Developer",
    "company": "Acme Corp",
    "description": "Build backend services in Python and PostgreSQL.",
    "labels": ["Python", "PostgreSQL"],
}


def test_format_record_prompt_contains_posting_and_instruction():
    prompt, _target = format_record(SAMPLE_RECORD)
    assert "Python Developer" in prompt
    assert "Acme Corp" in prompt
    assert "Build backend services in Python and PostgreSQL." in prompt
    assert "extract structured skills" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_format_record_target_lists_labels_one_per_line():
    _prompt, target = format_record(SAMPLE_RECORD)
    assert target == "- Python\n- PostgreSQL\n<|im_end|>"


def test_format_record_target_is_empty_marker_for_no_labels():
    record = dict(SAMPLE_RECORD, labels=[])
    _prompt, target = format_record(record)
    assert target == "<|im_end|>"


def test_build_training_examples_skips_unlabeled_records():
    records = [SAMPLE_RECORD, dict(SAMPLE_RECORD, posting_id="2", labels=[])]
    examples = build_training_examples(records)
    assert len(examples) == 1
    assert examples[0]["target"] == "- Python\n- PostgreSQL\n<|im_end|>"


def test_build_lora_config_returns_default_settings():
    config = build_lora_config()
    assert config.r == LORA_R == 16
    assert config.alpha == LORA_ALPHA == 32
    assert config.dropout == LORA_DROPOUT == 0.05
    assert config.target_modules == LORA_TARGET_MODULES


def test_build_lora_config_accepts_overrides():
    config = build_lora_config(r=8, alpha=16, dropout=0.1, target_modules=("q_proj",))
    assert config.r == 8
    assert config.alpha == 16
    assert config.dropout == 0.1
    assert config.target_modules == ("q_proj",)
