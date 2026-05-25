"""Step 2 of the pipeline — clean, balance, and split the raw arXiv pull.

Reads ``data/raw_arxiv.jsonl`` and writes three parquet files into
``data/processed/`` (``train``, ``validation``, ``test``) with a stratified
80/10/10 split.

Pipeline:
  1. Load, drop exact-duplicate ids.
  2. Drop rows whose abstract has fewer than ``MIN_ABSTRACT_WORDS`` tokens.
  3. Whitespace-normalise title and abstract (LaTeX kept as-is on purpose).
  4. Re-balance to ``min(class_count)`` per class.
  5. Build the ``text`` column = title + ". " + abstract.
  6. Map primary_category -> human-readable label.
  7. Stratified 80/10/10 split, seeded with ``SEED``.
  8. Save parquet + print a stats summary.

CLI: ``python -m src.build_dataset``
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORIES,
    LABEL_NAMES,
    MIN_ABSTRACT_WORDS,
    PROCESSED_DIR,
    RAW_PATH,
    SEED,
)
from .utils import ensure_dir, set_seed, setup_logging

log = logging.getLogger("build_dataset")

_WS_RE = re.compile(r"\s+")
_OUT_COLS = ["id", "title", "abstract", "text", "label", "primary_category", "published"]


def _normalise(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _load_raw(path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # First split off test (10%), then split remaining 90% into train/val (88.89% / 11.11%)
    # so the final ratio is 80/10/10.
    train_val, test = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label"],
        random_state=SEED,
    )
    train, val = train_test_split(
        train_val,
        test_size=1 / 9,  # 0.1111 -> 10% of the original
        stratify=train_val["label"],
        random_state=SEED,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main() -> None:
    setup_logging()
    set_seed(SEED)
    ensure_dir(PROCESSED_DIR)

    log.info("Loading raw rows from %s", RAW_PATH)
    df = _load_raw(RAW_PATH)
    log.info("Loaded %d raw rows", len(df))

    # 1. Drop duplicate ids (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    log.info("Dropped %d duplicate ids -> %d", before - len(df), len(df))

    # 2. Keep only the 6 target categories (defensive; fetch already filters)
    df = df[df["primary_category"].isin(CATEGORIES)].reset_index(drop=True)

    # 3. Normalise whitespace
    df["title"] = df["title"].map(_normalise)
    df["abstract"] = df["abstract"].map(_normalise)

    # 4. Min-length filter on abstract
    before = len(df)
    word_counts = df["abstract"].str.split().map(len)
    df = df[word_counts >= MIN_ABSTRACT_WORDS].reset_index(drop=True)
    log.info("Dropped %d rows with <%d-word abstracts -> %d", before - len(df), MIN_ABSTRACT_WORDS, len(df))

    # 5. Re-balance: take min(class_count) per class
    counts = df["primary_category"].value_counts()
    log.info("Per-class counts before balancing:\n%s", counts.to_string())
    n_per_class = int(counts.min())
    log.info("Balancing each class to %d rows", n_per_class)
    df = (
        df.groupby("primary_category", group_keys=False)
        .sample(n=n_per_class, random_state=SEED)
        .reset_index(drop=True)
    )

    # 6. Build text + label columns
    df["text"] = df["title"] + ". " + df["abstract"]
    df["label"] = df["primary_category"].map(LABEL_NAMES)
    assert df["label"].notna().all(), "Unmapped primary_category encountered"

    df = df[_OUT_COLS]
    log.info("Final balanced dataset: %d rows, %d classes", len(df), df["label"].nunique())

    # 7. Stratified split
    train, val, test = _stratified_split(df)
    log.info("Split sizes: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # 8. Save parquet
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        out = PROCESSED_DIR / f"{name}.parquet"
        part.to_parquet(out, index=False)
        log.info("Wrote %s", out)

    # 9. Stats summary
    summary = pd.DataFrame(
        {
            "train": train["label"].value_counts().sort_index(),
            "validation": val["label"].value_counts().sort_index(),
            "test": test["label"].value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    summary.loc["TOTAL"] = summary.sum()
    print("\n=== Split distribution by label ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
