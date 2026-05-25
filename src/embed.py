"""Step 3 of the pipeline — encode text with a SentenceTransformer and cache.

For each split (train/validation/test) and a chosen embedder, encode the
``text`` column to a ``float32`` matrix with L2-normalised rows and cache it
to ``data/embeddings/<embedder_slug>/<split>.npy``. Re-runs skip splits that
are already cached, so this is cheap to re-invoke.

CLI examples::

    python -m src.embed --embedder mpnet
    python -m src.embed --embedder minilm
    python -m src.embed --embedder e5
    python -m src.embed --embedder all          # all three sequentially
"""
from __future__ import annotations

import os

# Workaround for Windows MKL/OpenMP DLL clash between conda numpy and pip torch:
# import torch BEFORE numpy/pandas, and allow duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401  (force torch to win the DLL load race)
except Exception:
    pass

import argparse
import logging

import numpy as np
import pandas as pd

from .config import EMBEDDERS, EMB_DIR, PROCESSED_DIR, SEED
from .utils import ensure_dir, set_seed, setup_logging, slugify_embedder

log = logging.getLogger("embed")

SPLITS = ("train", "validation", "test")
_BATCH = 32


def _prepare_texts(texts: list[str], embedder_name: str) -> list[str]:
    """e5 family expects a 'passage: ' / 'query: ' prefix per the model card."""
    if "e5" in embedder_name.lower():
        return [f"passage: {t}" for t in texts]
    return texts


def encode_split(split: str, embedder_name: str) -> np.ndarray:
    """Encode one split and cache to disk; return the matrix.

    If the cache file already exists, load and return it without re-encoding.
    """
    from sentence_transformers import SentenceTransformer  # lazy import (torch)

    slug = slugify_embedder(embedder_name)
    out_dir = ensure_dir(EMB_DIR / slug)
    cache = out_dir / f"{split}.npy"
    if cache.exists():
        log.info("[%s/%s] cache hit -> %s", slug, split, cache)
        return np.load(cache)

    parquet = PROCESSED_DIR / f"{split}.parquet"
    df = pd.read_parquet(parquet, columns=["text"])
    texts = _prepare_texts(df["text"].tolist(), embedder_name)
    log.info("[%s/%s] encoding %d texts with batch=%d", slug, split, len(texts), _BATCH)

    model = SentenceTransformer(embedder_name)
    emb = model.encode(
        texts,
        batch_size=_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(cache, emb)
    log.info("[%s/%s] wrote %s, shape=%s", slug, split, cache, emb.shape)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode splits with a SentenceTransformer.")
    choices = list(EMBEDDERS.keys()) + ["all"]
    parser.add_argument("--embedder", required=True, choices=choices, help="Which embedder slug to run.")
    args = parser.parse_args()

    setup_logging()
    set_seed(SEED)

    slugs = list(EMBEDDERS.keys()) if args.embedder == "all" else [args.embedder]
    for slug in slugs:
        model_id = EMBEDDERS[slug]
        log.info("=== Embedder: %s (%s) ===", slug, model_id)
        for split in SPLITS:
            encode_split(split, model_id)


if __name__ == "__main__":
    main()
