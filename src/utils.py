"""Small helpers shared across pipeline scripts."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch is optional here; ignore if missing or its DLLs fail to load
        # (the sklearn-only stages don't need it).
        pass


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify_embedder(model_id: str) -> str:
    """Filesystem-safe slug for embedding cache subfolders."""
    return model_id.replace("/", "__")
