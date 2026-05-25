"""Step 5b — push the winning classifier to the Hugging Face Hub.

Uploads three artefacts from ``data/artifacts/`` to ``HF_MODEL_REPO``:

  * ``classifier.joblib``     (renamed from ``best_classifier.joblib``)
  * ``label_encoder.joblib``
  * ``config.json``

Also writes a model card (README.md) including the full metrics table from
``results/metrics.csv`` and a runnable inference snippet.

Prerequisite::

    huggingface-cli login

CLI: ``python -m src.upload_model``
"""
from __future__ import annotations

import json
import logging
import sklearn

import pandas as pd
from huggingface_hub import HfApi, create_repo

from .config import (
    ARTIFACTS_DIR,
    HF_DATASET_REPO,
    HF_MODEL_REPO,
    HF_USERNAME,
    RESULTS_DIR,
)
from .utils import setup_logging

log = logging.getLogger("upload_model")


def _build_model_card(cfg: dict, metrics_df: pd.DataFrame) -> str:
    metrics_md = metrics_df.to_markdown(index=False)
    feature_kind = cfg["feature_kind"]
    embedder = cfg.get("embedder")
    labels_list = ", ".join(f"`{l}`" for l in cfg["labels"])

    if feature_kind == "embedding":
        load_snippet = f"""```python
import joblib
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

repo = "{HF_MODEL_REPO}"
clf  = joblib.load(hf_hub_download(repo, "classifier.joblib"))
le   = joblib.load(hf_hub_download(repo, "label_encoder.joblib"))
embedder = SentenceTransformer("{embedder}")

text = "Title. Abstract text goes here..."
emb  = embedder.encode([text], normalize_embeddings=True)
probs = clf.predict_proba(emb)[0]
ranked = sorted(zip(le.classes_, probs), key=lambda kv: -kv[1])
for label, p in ranked:
    print(f"{{label:<14}} {{p:.3f}}")
```"""
    else:  # tfidf
        load_snippet = f"""```python
import joblib
from huggingface_hub import hf_hub_download

repo = "{HF_MODEL_REPO}"
bundle = joblib.load(hf_hub_download(repo, "classifier.joblib"))
le     = joblib.load(hf_hub_download(repo, "label_encoder.joblib"))
vec, clf = bundle["vectorizer"], bundle["classifier"]

text  = "Title. Abstract text goes here..."
X     = vec.transform([text])
probs = clf.predict_proba(X)[0]
ranked = sorted(zip(le.classes_, probs), key=lambda kv: -kv[1])
for label, p in ranked:
    print(f"{{label:<14}} {{p:.3f}}")
```"""

    return f"""---
language: en
license: apache-2.0
library_name: sklearn
tags:
- text-classification
- arxiv
- scikit-learn
- sentence-transformers
datasets:
- {HF_DATASET_REPO}
metrics:
- f1
- accuracy
---

# arXiv CS Sub-field Linear Probe

A lightweight classifier that maps an arXiv CS abstract to one of six
overlapping sub-fields ({labels_list}). The winning configuration is
**`{cfg["config"]}`** — features from `{embedder or "TF-IDF (10k feats, 1-2grams)"}`
fed into a scikit-learn classifier.

Trained dataset: [{HF_DATASET_REPO}](https://huggingface.co/datasets/{HF_DATASET_REPO}).

## Intended use

Educational / research demo of frozen-embedding linear probing for fine-grained
text classification. Predictions are noisy near class boundaries by design
(e.g. `ai_general` vs `lg_ml`).

## Training details

- Feature kind: `{feature_kind}`
{"- Embedder: `" + embedder + "`" if embedder else ""}
- Selection rule: highest validation macro-F1 across 6 configurations.
- Train/val/test split: stratified 80/10/10, `random_state=42`.
- scikit-learn version (at train time): `{sklearn.__version__}`.

## Evaluation

All configurations evaluated on the same held-out test set:

{metrics_md}

## How to load and run inference

{load_snippet}

## Limitations

- Single-label simplification of a multi-label problem.
- No temporal generalisation test: train/val/test sampled from the same window.
- Small per-class sample (~690 papers/class).
- The TF-IDF baseline is competitive on test, so the embedding advantage on
  this task is modest — treat the gap as task-dependent rather than universal.

## License

Apache-2.0 for the trained artefacts. The training data itself is subject to
arXiv's terms; see the dataset card.

## Citation

```bibtex
@misc{{arxiv-subfield-probe,
  author = {{ {HF_USERNAME} }},
  title  = {{ {HF_MODEL_REPO} }},
  year   = 2026,
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{HF_MODEL_REPO}}}}}
}}
```
"""


def main() -> None:
    setup_logging()

    cfg_path = ARTIFACTS_DIR / "config.json"
    clf_path = ARTIFACTS_DIR / "best_classifier.joblib"
    le_path = ARTIFACTS_DIR / "label_encoder.joblib"
    metrics_path = RESULTS_DIR / "metrics.csv"
    for p in (cfg_path, clf_path, le_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artefact: {p} — run train_eval first.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    metrics_df = pd.read_csv(metrics_path)

    log.info("Ensuring model repo exists: %s", HF_MODEL_REPO)
    create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True, private=False)

    api = HfApi()
    log.info("Uploading classifier.joblib")
    api.upload_file(
        path_or_fileobj=str(clf_path),
        path_in_repo="classifier.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    log.info("Uploading label_encoder.joblib")
    api.upload_file(
        path_or_fileobj=str(le_path),
        path_in_repo="label_encoder.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    log.info("Uploading config.json")
    api.upload_file(
        path_or_fileobj=str(cfg_path),
        path_in_repo="config.json",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    log.info("Uploading README.md (model card)")
    card = _build_model_card(cfg, metrics_df)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    url = f"https://huggingface.co/{HF_MODEL_REPO}"
    print(f"\nModel live at: {url}\n")


if __name__ == "__main__":
    main()
