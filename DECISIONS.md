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

## 2026-11-25: Heuristic first-pass labels

- **Choice:** label each posting by matching a small curated keyword list
  (common technologies and skills such as Python, SQL, Docker, Kubernetes,
  and machine learning) case-insensitively against the description text,
  with word boundaries so short keywords do not match inside other words.
  The script is `scripts/label_postings.py`; it fills the `labels` field
  of each record in `data/raw.jsonl` before the split.
- **Alternatives considered:** hand-labeling each posting, and extracting
  labels with a language model. Hand-labeling 15-16 postings is doable but
  slow to repeat as the dataset grows; a model-based pass adds cost and
  nondeterminism for a first dataset version.
- **Why:** the heuristic gives a usable first pass for the training
  pipeline immediately, with a fixed label vocabulary that keeps the label
  set consistent. The labels are not ground truth: they only catch
  phrases on the list and can miss skills phrased differently or include
  a phrase mentioned in passing. They will be reviewed or replaced before
  training.
