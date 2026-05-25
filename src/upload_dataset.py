"""Step 5a — push the processed parquet splits to the Hugging Face Hub.

Builds a ``DatasetDict`` from ``data/processed/{train,validation,test}.parquet``
and pushes it to ``HF_DATASET_REPO``. Also writes a dataset card (README.md)
with the standard sections.

Prerequisite::

    huggingface-cli login

CLI: ``python -m src.upload_dataset``
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from .config import (
    CATEGORIES,
    HF_DATASET_REPO,
    HF_USERNAME,
    LABEL_NAMES,
    PROCESSED_DIR,
    TIME_RANGE,
)
from .utils import setup_logging

log = logging.getLogger("upload_dataset")

SPLITS = ("train", "validation", "test")


def _load_dataset_dict() -> DatasetDict:
    parts = {}
    for split in SPLITS:
        df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
        parts[split] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(parts)


def _build_dataset_card(dsd: DatasetDict) -> str:
    sizes = {s: len(dsd[s]) for s in SPLITS}
    total = sum(sizes.values())
    cats_list = ", ".join(f"`{c}`" for c in CATEGORIES)
    label_map_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in LABEL_NAMES.items())
    split_md = "\n".join(f"| {s} | {sizes[s]} |" for s in SPLITS)
    return f"""---
language:
- en
license: cc-by-4.0
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- arxiv
- computer-science
- abstracts
- multi-class
---

# arXiv CS Sub-field Classification Dataset

Balanced, single-label classification dataset of recent arXiv abstracts across
six overlapping CS sub-fields. Each row carries one paper's title and abstract,
plus a label derived from its arXiv `primary_category`.

## Supported tasks

- `text-classification` — predict the arXiv sub-field given title + abstract.

## Languages

English (`en`).

## Data fields

| field | type | description |
|---|---|---|
| `id` | string | arXiv identifier (e.g. `2401.12345v1`) |
| `title` | string | Paper title, whitespace-normalised |
| `abstract` | string | Paper abstract, whitespace-normalised, LaTeX preserved |
| `text` | string | `title + ". " + abstract`, the field used as model input |
| `label` | string | Human-readable sub-field name |
| `primary_category` | string | Original arXiv primary category |
| `published` | string | ISO timestamp of first arXiv submission |

## Label space

The six target arXiv categories ({cats_list}) map to human-readable names:

| arXiv primary_category | label |
|---|---|
{label_map_md}

## Splits

| split | rows |
|---|---|
{split_md}
| **total** | **{total}** |

Splits are stratified 80/10/10 by label, seeded with `random_state=42`.

## Source

- arXiv API, queried with the `arxiv` Python package (`Client(delay_seconds=3.5)`).
- Time range: {TIME_RANGE[0]} to {TIME_RANGE[1]}.
- Filtering rule: only papers whose `primary_category` equals the queried
  category are kept (so the label is unambiguous), then each class is
  down-sampled to the size of the smallest class for balance.

## Annotation process

Labels are arXiv `primary_category` values; no manual annotation was performed.
Boundary cases between e.g. `cs.AI` and `cs.LG` are inherently noisy and are
part of what makes the task interesting.

## Limitations

- Single-label simplification of a fundamentally multi-label problem
  (most papers carry several arXiv categories).
- Class boundaries are noisy by construction.
- No temporal generalisation split — train, val and test all come from the
  same time window.

## Licensing

arXiv abstracts are distributed under arXiv's non-exclusive license. This
dataset redistributes metadata + abstracts for non-commercial research use.
Each row keeps the arXiv `id` so users can verify the source.

## Citation

```bibtex
@misc{{arxiv-cs-subfield,
  author = {{ {HF_USERNAME} }},
  title  = {{ {HF_DATASET_REPO} }},
  year   = 2026,
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{HF_DATASET_REPO}}}}}
}}
```
"""


def main() -> None:
    setup_logging()
    log.info("Loading parquet splits from %s", PROCESSED_DIR)
    dsd = _load_dataset_dict()
    for s in SPLITS:
        log.info("  %s: %d rows", s, len(dsd[s]))

    log.info("Pushing to %s", HF_DATASET_REPO)
    dsd.push_to_hub(HF_DATASET_REPO, private=False)

    log.info("Uploading dataset card (README.md)")
    card = _build_dataset_card(dsd)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
    print(f"\nDataset live at: {url}\n")


if __name__ == "__main__":
    main()
