"""Collect public job postings and write them as JSONL.

Source: the Remotive API (https://remotive.com/api/remote-jobs), a public
endpoint that needs no key or login. Each record has posting_id, title,
company, description, and labels (empty until the labeling step).
"""

import argparse
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://remotive.com/api/remote-jobs"

# A spread of engineering roles so the sample covers more than one job family.
DEFAULT_QUERIES = [
    "software engineer",
    "data scientist",
    "machine learning engineer",
    "devops engineer",
    "product manager",
]

_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(raw: str) -> str:
    """Strip HTML tags and entities from a posting description."""
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_postings(query: str, limit: int) -> list[dict]:
    """Fetch up to `limit` postings for one search query."""
    params = urllib.parse.urlencode({"search": query, "limit": limit})
    request = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "lora-skill-extractor/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    return payload.get("jobs", [])


def to_record(job: dict) -> dict:
    """Convert one API job object into a dataset record."""
    return {
        "posting_id": str(job["id"]),
        "title": job["title"],
        "company": job["company_name"],
        "description": clean_text(job["description"]),
        "labels": [],
    }


def collect(limit: int = 50, queries: list[str] | None = None) -> list[dict]:
    """Collect up to `limit` unique postings across the given queries."""
    queries = queries or DEFAULT_QUERIES
    per_query = max(1, limit // len(queries))
    seen: set[str] = set()
    records: list[dict] = []
    for query in queries:
        for job in fetch_postings(query, per_query):
            record = to_record(job)
            if record["posting_id"] in seen:
                continue
            seen.add(record["posting_id"])
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/raw.jsonl"))
    parser.add_argument("--query", action="append", dest="queries", default=None)
    args = parser.parse_args()

    records = collect(limit=args.limit, queries=args.queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} postings to {args.output}")


if __name__ == "__main__":
    main()
