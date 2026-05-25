"""Step 1 of the pipeline — pull raw arXiv metadata.

For each category in ``config.CATEGORIES``, this script queries the arXiv API
and appends papers whose ``primary_category`` matches the queried category to
``data/raw_arxiv.jsonl`` (one JSON object per line). Behaviour:

* Idempotent — existing rows are read on startup; we only fetch what's missing
  to reach ``PER_CATEGORY_TARGET`` per category.
* Rate-limited — uses the ``arxiv`` v4 ``Client`` with ``delay_seconds=3.5``
  and ``num_retries=5`` (well above arXiv's 3s/request recommendation).
* Robust — per-paper exceptions are logged and skipped, never crash the run.

CLI::

    python -m src.fetch_arxiv                 # full pull
    python -m src.fetch_arxiv --per-category 30   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import arxiv
from tqdm import tqdm

from .config import CATEGORIES, PER_CATEGORY_TARGET, RAW_PATH, TIME_RANGE
from .utils import ensure_dir, setup_logging

log = logging.getLogger("fetch_arxiv")

# arXiv accepts submittedDate filters as YYYYMMDDHHMM.
_DATE_FROM = TIME_RANGE[0].replace("-", "") + "0000"
_DATE_TO = TIME_RANGE[1].replace("-", "") + "2359"

# Overshoot factor: arXiv returns hits where the requested cat is *any* of the
# categories on the paper, but we only keep primary_category matches. The miss
# rate varies by category; 4x has empirically been enough.
_OVERSHOOT = 4


def _load_existing_ids_by_category(path: Path) -> dict[str, set[str]]:
    """Read existing JSONL (if any) and bucket arXiv IDs by primary_category."""
    out: dict[str, set[str]] = {c: set() for c in CATEGORIES}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = row.get("primary_category")
            if cat in out:
                out[cat].add(row["id"])
    return out


def _result_to_row(r: arxiv.Result) -> dict:
    # entry_id looks like "http://arxiv.org/abs/2401.12345v2" — take the tail.
    short_id = r.entry_id.rsplit("/", 1)[-1]
    return {
        "id": short_id,
        "title": (r.title or "").strip(),
        "abstract": (r.summary or "").strip(),
        "primary_category": r.primary_category,
        "all_categories": list(r.categories or []),
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
    }


def fetch_category(
    client: arxiv.Client,
    category: str,
    target: int,
    already_have: set[str],
    out_fh,
) -> int:
    """Fetch additional papers for one category until ``target`` is reached.

    Returns the number of *new* rows written.
    """
    need = target - len(already_have)
    if need <= 0:
        log.info("[%s] already have %d (>= %d), skipping", category, len(already_have), target)
        return 0

    query = f"cat:{category} AND submittedDate:[{_DATE_FROM} TO {_DATE_TO}]"
    search = arxiv.Search(
        query=query,
        max_results=need * _OVERSHOOT,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    written = 0
    pbar = tqdm(total=need, desc=f"{category:>7}", unit="paper")
    try:
        for r in client.results(search):
            if written >= need:
                break
            try:
                if r.primary_category != category:
                    continue
                short_id = r.entry_id.rsplit("/", 1)[-1]
                if short_id in already_have:
                    continue
                row = _result_to_row(r)
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
                already_have.add(short_id)
                written += 1
                pbar.update(1)
            except Exception as e:  # noqa: BLE001 — per-paper resilience
                log.warning("[%s] skip paper due to %s: %s", category, type(e).__name__, e)
                continue
    except Exception as e:  # noqa: BLE001 — per-category resilience
        log.error("[%s] search aborted after %d new rows: %s", category, written, e)
    finally:
        pbar.close()

    if written < need:
        log.warning(
            "[%s] only got %d/%d new rows — arXiv may not have enough primary-category hits "
            "in the time window; consider widening TIME_RANGE or increasing _OVERSHOOT",
            category,
            written,
            need,
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull raw arXiv metadata.")
    parser.add_argument(
        "--per-category",
        type=int,
        default=PER_CATEGORY_TARGET,
        help=f"Target rows per category (default {PER_CATEGORY_TARGET}).",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_dir(RAW_PATH.parent)

    existing = _load_existing_ids_by_category(RAW_PATH)
    for c in CATEGORIES:
        log.info("[%s] existing rows on disk: %d", c, len(existing[c]))

    client = arxiv.Client(page_size=100, delay_seconds=3.5, num_retries=5)

    total_new = 0
    with RAW_PATH.open("a", encoding="utf-8") as fh:
        for category in CATEGORIES:
            total_new += fetch_category(client, category, args.per_category, existing[category], fh)

    log.info("Done. New rows this run: %d. Total file: %s", total_new, RAW_PATH)
    for c in CATEGORIES:
        log.info("[%s] final count on disk: %d", c, len(existing[c]))


if __name__ == "__main__":
    main()
