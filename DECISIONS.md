# Decisions

## 2026-11-24: Model size 1B-3B and dataset source

- **Model size:** fine-tune a small open model in the 1B-3B parameter range.
  A model this size fits on a single Colab GPU for LoRA training and serves
  cheaply with vLLM, while still being large enough to learn structured
  extraction from job posting text.
- **Dataset source:** public job postings fetched from the Remotive API
  (`https://remotive.com/api/remote-jobs`), which needs no key or login.
  The raw postings are stored as JSONL with `labels` left empty until the
  labeling step; a fixed-seed 80/20 split produces `data/train.jsonl` and
  `data/test.jsonl`. Note: the API returns the same latest 15 jobs
  regardless of the search or category parameter, so the first sample is
  15 postings. Later slices can grow the dataset from additional sources.
