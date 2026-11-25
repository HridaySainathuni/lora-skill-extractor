# lora-skill-extractor

LoRA fine-tune of a small open model (1B-3B parameters) to extract structured
skills and requirements from job postings. The fine-tuned model will be served
with vLLM behind a FastAPI endpoint and benchmarked against an API model on
accuracy, latency, and cost per 1,000 requests.

## Planned milestones

1. **Dataset build** (this milestone): collect public job postings, label
   them with heuristic skill tags, and split them into train and test sets.
2. **Training**: LoRA fine-tune the base model on the labeled training set
   (Colab GPU).
3. **Serving**: serve the fine-tuned model with vLLM behind a FastAPI
   endpoint.
4. **Benchmarks**: compare the fine-tuned model against an API model on
   accuracy, latency, and cost per 1,000 requests.

## Dataset

Source: public job postings from the Remotive API
(`https://remotive.com/api/remote-jobs`), which needs no key or login.
The current sample is 16 postings.

Fields, one JSON object per line in `data/raw.jsonl`:

```json
{
  "posting_id": "1234567",
  "title": "Software Engineer",
  "company": "Example Corp",
  "description": "We are looking for...",
  "labels": ["Python", "SQL"]
}
```

`labels` holds skill and requirement strings drawn from the curated
keyword list in `scripts/label_postings.py`. The labels are heuristic: a
label is added when its phrase appears in the description text, matched
case-insensitively. They are a first pass for the training pipeline, not
ground truth, and will be reviewed or replaced before training.

`data/train.jsonl` (12 records) and `data/test.jsonl` (4 records) are the
80/20 split of the raw postings, produced with a fixed seed so the split
is reproducible.

## Usage

Collect postings (public Remotive API, no key or login needed):

```bash
python scripts/collect_postings.py --limit 50 --output data/raw.jsonl
```

Label postings with heuristic skill tags:

```bash
python scripts/label_postings.py --input data/raw.jsonl --output data/raw.jsonl
```

Split into train and test sets:

```bash
python scripts/split_dataset.py --input data/raw.jsonl --seed 42
```

Run the tests:

```bash
pytest
ruff check .
```
