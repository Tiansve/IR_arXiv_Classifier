"""Gradio Space app — arXiv CS sub-field classifier.

Loads the trained classifier + label encoder + config from the HF model repo
and exposes a simple title+abstract -> top-k labels interface.

Supports two model flavours via ``config.json["feature_kind"]``:
  * ``embedding`` — uses a SentenceTransformer named in config
  * ``tfidf``     — uses the bundled vectoriser inside classifier.joblib
"""
from __future__ import annotations

import os

# Windows-only workaround for the MKL/OpenMP DLL clash between conda numpy
# and pip torch: import torch first, allow duplicate OpenMP runtimes.
# (No-op on Linux / HF Space.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

import json
import logging
from pathlib import Path

import gradio as gr
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "Tiansve/arxiv-subfield-linear-probe"
MIN_ABSTRACT_WORDS = 20

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("space")

# ---- Load artefacts ------------------------------------------------------- #
config_path = hf_hub_download(MODEL_REPO, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

clf = joblib.load(hf_hub_download(MODEL_REPO, "classifier.joblib"))
le = joblib.load(hf_hub_download(MODEL_REPO, "label_encoder.joblib"))

FEATURE_KIND = cfg["feature_kind"]
EMBEDDER_NAME = cfg.get("embedder")
CLASS_NAMES = list(le.classes_)

embedder = None
vectorizer = None
classifier = clf  # rebound below for tfidf

if FEATURE_KIND == "embedding":
    from sentence_transformers import SentenceTransformer
    log.info("Loading SentenceTransformer: %s", EMBEDDER_NAME)
    embedder = SentenceTransformer(EMBEDDER_NAME)
elif FEATURE_KIND == "tfidf":
    # classifier.joblib for tfidf is a dict {"vectorizer", "classifier"}
    vectorizer = clf["vectorizer"]
    classifier = clf["classifier"]
else:
    raise ValueError(f"Unknown feature_kind: {FEATURE_KIND!r}")


# ---- Inference ------------------------------------------------------------ #
def _featurise(text: str):
    if FEATURE_KIND == "embedding":
        return embedder.encode([text], normalize_embeddings=True)
    return vectorizer.transform([text])


def classify(title: str, abstract: str):
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not abstract or len(abstract.split()) < MIN_ABSTRACT_WORDS:
        return {f"(paste at least {MIN_ABSTRACT_WORDS} words of abstract)": 1.0}
    text = (title + ". " + abstract).strip(". ").strip()
    X = _featurise(text)
    probs = classifier.predict_proba(X)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ---- Examples ------------------------------------------------------------- #
def _load_examples():
    path = Path(__file__).parent / "examples.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    # Gradio Examples expects a list of [input1, input2, ...] matching the inputs order
    return [[row["title"], row["abstract"]] for row in data]


# ---- UI ------------------------------------------------------------------- #
description = (
    f"Paste an arXiv-style abstract and the model predicts which CS sub-field it belongs to.\n\n"
    f"Trained on ~4.1k recent abstracts across the six categories "
    f"({', '.join(CLASS_NAMES)}). "
    f"Current model: **{cfg['config']}** "
    f"(test macro-F1 = {cfg['test_macro_f1']:.3f}). "
    f"Predictions are noisy near class boundaries by design — see the linked report for analysis."
)

demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(label="Title", placeholder="Paper title (optional)"),
        gr.Textbox(label="Abstract", lines=10, placeholder="Paste the abstract here..."),
    ],
    outputs=gr.Label(num_top_classes=4, label="Predicted sub-field"),
    title="arXiv CS Sub-field Classifier",
    description=description,
    examples=_load_examples(),
    cache_examples=False,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
