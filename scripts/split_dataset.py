"""Split a raw JSONL dataset into train and test sets with a fixed seed."""

import argparse
import json
import random
from pathlib import Path

REQUIRED_FIELDS = ("posting_id", "title", "company", "description", "labels")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_dataset(
    records: list[dict], seed: int = 42, train_fraction: float = 0.8
) -> tuple[list[dict], list[dict]]:
    """Shuffle with a fixed seed and split into train and test lists."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * train_fraction)
    return shuffled[:cut], shuffled[cut:]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    train, test = split_dataset(records, seed=args.seed, train_fraction=args.train_fraction)
    write_jsonl(args.train_output, train)
    write_jsonl(args.test_output, test)
    print(f"Split {len(records)} records: {len(train)} train, {len(test)} test")


if __name__ == "__main__":
    main()
