"""Label postings with heuristic skill and requirement tags.

The labels are a first pass for the training pipeline, not ground truth:
each label is a phrase from a small curated keyword list that appears in
the posting description (matched case-insensitively). They will be
reviewed or replaced before training.
"""

import argparse
import json
import re
from pathlib import Path

# Curated list of common technologies and skills. Keep it small and
# explicit: a label is only ever one of these strings, so the label set
# stays consistent across the dataset.
KEYWORDS = [
    "Python",
    "SQL",
    "JavaScript",
    "TypeScript",
    "Java",
    "Go",
    "C++",
    "Rust",
    "Ruby",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "Terraform",
    "Git",
    "Linux",
    "React",
    "Node.js",
    "FastAPI",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "machine learning",
    "deep learning",
    "natural language processing",
    "data analysis",
    "data engineering",
    "CI/CD",
    "REST API",
    "GraphQL",
    "Agile",
    "communication",
    "problem solving",
    "leadership",
    "project management",
]


def label_text(text: str) -> list[str]:
    """Return the keywords that appear in `text`, matched case-insensitively.

    Word boundaries keep short keywords from matching inside other words
    (for example, "Go" should not match "Google").
    """
    found: list[str] = []
    for keyword in KEYWORDS:
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            found.append(keyword)
    return found


def label_record(record: dict) -> dict:
    """Return a copy of `record` with heuristic labels from its description."""
    labeled = dict(record)
    labeled["labels"] = label_text(record.get("description", ""))
    return labeled


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/raw.jsonl"))
    args = parser.parse_args()

    records = [label_record(record) for record in load_jsonl(args.input)]
    write_jsonl(args.output, records)
    labeled = sum(1 for record in records if record["labels"])
    print(f"Labeled {len(records)} postings ({labeled} with at least one label)")


if __name__ == "__main__":
    main()
