"""Tests for the dataset split and record format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split_dataset import REQUIRED_FIELDS, load_jsonl, split_dataset  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _sample_records(n: int = 50) -> list[dict]:
    return [
        {
            "posting_id": str(i),
            "title": f"Engineer {i}",
            "company": f"Company {i}",
            "description": f"Description {i}",
            "labels": [],
        }
        for i in range(n)
    ]


def test_split_is_disjoint_and_covers_all_records():
    records = _sample_records()
    train, test = split_dataset(records, seed=42)
    train_ids = {r["posting_id"] for r in train}
    test_ids = {r["posting_id"] for r in test}
    assert not train_ids & test_ids
    assert train_ids | test_ids == {r["posting_id"] for r in records}


def test_split_ratio_is_roughly_80_20():
    records = _sample_records()
    train, test = split_dataset(records, seed=42)
    fraction = len(train) / len(records)
    assert 0.7 <= fraction <= 0.9
    assert len(train) == 40
    assert len(test) == 10


def test_split_is_reproducible_with_fixed_seed():
    records = _sample_records()
    first = split_dataset(records, seed=42)
    second = split_dataset(records, seed=42)
    assert [r["posting_id"] for r in first[0]] == [r["posting_id"] for r in second[0]]
    assert [r["posting_id"] for r in first[1]] == [r["posting_id"] for r in second[1]]


def test_every_record_has_required_fields():
    for name in ("raw.jsonl", "train.jsonl", "test.jsonl"):
        path = DATA_DIR / name
        assert path.exists(), f"{name} is missing"
        for record in load_jsonl(path):
            for field in REQUIRED_FIELDS:
                assert field in record, f"{name} record missing {field}"
            assert isinstance(record["labels"], list)


def test_split_files_are_disjoint():
    train_ids = {r["posting_id"] for r in load_jsonl(DATA_DIR / "train.jsonl")}
    test_ids = {r["posting_id"] for r in load_jsonl(DATA_DIR / "test.jsonl")}
    raw_ids = {r["posting_id"] for r in load_jsonl(DATA_DIR / "raw.jsonl")}
    assert not train_ids & test_ids
    assert train_ids | test_ids == raw_ids
