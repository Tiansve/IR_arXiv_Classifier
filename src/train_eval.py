"""Step 4 of the pipeline — train and evaluate multiple classifier configs.

Configurations (all use the same train/val/test parquet splits):

  * tfidf_logreg  — TF-IDF (10k feats, 1-2grams) + LogisticRegression  (baseline)
  * mpnet_logreg  — mpnet embeddings           + LogisticRegression
  * mpnet_linsvm  — mpnet embeddings           + CalibratedClassifierCV(LinearSVC)
  * mpnet_mlp     — mpnet embeddings           + MLPClassifier(256,128)
  * minilm_logreg — MiniLM embeddings          + LogisticRegression
  * e5_logreg     — e5 embeddings              + LogisticRegression

For LogReg we tune ``C ∈ {0.1, 1, 10}`` on validation; MLP and LinearSVC use
defaults. Selection is by validation macro-F1; the winning model is persisted
to ``data/artifacts/`` together with the label encoder and a config JSON so
the HF Space can load it.

Outputs to ``results/``:
  * metrics.csv
  * confusion_matrix.png  (best config on test set, normalised over true labels)
  * per_class_report.txt  (sklearn classification_report for the best config)

CLI: ``python -m src.train_eval``
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .config import (
    ARTIFACTS_DIR,
    EMB_DIR,
    EMBEDDERS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
)
from .utils import ensure_dir, set_seed, setup_logging, slugify_embedder

log = logging.getLogger("train_eval")

_LOGREG_C_GRID = (0.1, 1.0, 10.0)
_BEST_CLF_FILE = "best_classifier.joblib"
_LABEL_ENCODER_FILE = "label_encoder.joblib"
_BEST_CONFIG_FILE = "config.json"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_split(split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")


def _load_emb(embedder_slug: str, split: str) -> np.ndarray:
    model_id = EMBEDDERS[embedder_slug]
    return np.load(EMB_DIR / slugify_embedder(model_id) / f"{split}.npy")


# ---------------------------------------------------------------------------
# Per-config training helpers
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    config: str
    best_estimator: object
    val_macro_f1: float
    test_acc: float
    test_macro_f1: float
    test_weighted_f1: float
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    feature_kind: str  # "tfidf" | "embedding"
    embedder_slug: str | None  # None for tfidf


def _logreg_with_tuning(
    X_train, y_train, X_val, y_val
) -> tuple[LogisticRegression, float]:
    """Tune LogReg C on the validation split; return best estimator + val F1."""
    best, best_f1 = None, -1.0
    for c in _LOGREG_C_GRID:
        clf = LogisticRegression(
            C=c, max_iter=2000, n_jobs=-1, random_state=SEED
        )
        clf.fit(X_train, y_train)
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        log.info("  C=%.2f  val_macro_f1=%.4f", c, f1)
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best, best_f1


def _eval_on_test(clf, X_test, y_test) -> tuple[float, float, float, np.ndarray]:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mac = f1_score(y_test, y_pred, average="macro")
    wgt = f1_score(y_test, y_pred, average="weighted")
    return acc, mac, wgt, y_pred


# ---------------------------------------------------------------------------
# Config runners
# ---------------------------------------------------------------------------

def _run_tfidf_logreg(train_df, val_df, test_df, y_train, y_val, y_test) -> TrainResult:
    log.info("=== tfidf_logreg ===")
    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vec.fit_transform(train_df["text"])
    X_val = vec.transform(val_df["text"])
    X_test = vec.transform(test_df["text"])
    clf, val_f1 = _logreg_with_tuning(X_train, y_train, X_val, y_val)
    acc, mac, wgt, y_pred = _eval_on_test(clf, X_test, y_test)
    # Wrap vectoriser + clf into a single object for downstream use
    pipeline = {"vectorizer": vec, "classifier": clf}
    return TrainResult(
        "tfidf_logreg", pipeline, val_f1, acc, mac, wgt, y_test, y_pred,
        feature_kind="tfidf", embedder_slug=None,
    )


def _run_embedding_logreg(embedder_slug, y_train, y_val, y_test) -> TrainResult:
    name = f"{embedder_slug}_logreg"
    log.info("=== %s ===", name)
    X_train = _load_emb(embedder_slug, "train")
    X_val = _load_emb(embedder_slug, "validation")
    X_test = _load_emb(embedder_slug, "test")
    clf, val_f1 = _logreg_with_tuning(X_train, y_train, X_val, y_val)
    acc, mac, wgt, y_pred = _eval_on_test(clf, X_test, y_test)
    return TrainResult(
        name, clf, val_f1, acc, mac, wgt, y_test, y_pred,
        feature_kind="embedding", embedder_slug=embedder_slug,
    )


def _run_embedding_linsvm(embedder_slug, y_train, y_val, y_test) -> TrainResult:
    name = f"{embedder_slug}_linsvm"
    log.info("=== %s ===", name)
    X_train = _load_emb(embedder_slug, "train")
    X_val = _load_emb(embedder_slug, "validation")
    X_test = _load_emb(embedder_slug, "test")
    base = LinearSVC(C=1.0, random_state=SEED)
    clf = CalibratedClassifierCV(base, cv=3)  # so we get predict_proba for the Space
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    log.info("  val_macro_f1=%.4f", val_f1)
    acc, mac, wgt, y_pred = _eval_on_test(clf, X_test, y_test)
    return TrainResult(
        name, clf, val_f1, acc, mac, wgt, y_test, y_pred,
        feature_kind="embedding", embedder_slug=embedder_slug,
    )


def _run_embedding_mlp(embedder_slug, y_train, y_val, y_test) -> TrainResult:
    name = f"{embedder_slug}_mlp"
    log.info("=== %s ===", name)
    X_train = _load_emb(embedder_slug, "train")
    X_val = _load_emb(embedder_slug, "validation")
    X_test = _load_emb(embedder_slug, "test")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=200,
        early_stopping=True,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    log.info("  val_macro_f1=%.4f", val_f1)
    acc, mac, wgt, y_pred = _eval_on_test(clf, X_test, y_test)
    return TrainResult(
        name, clf, val_f1, acc, mac, wgt, y_test, y_pred,
        feature_kind="embedding", embedder_slug=embedder_slug,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _save_confusion_matrix(result: TrainResult, label_names: list[str], path: Path) -> None:
    cm = confusion_matrix(result.y_test_true, result.y_test_pred, normalize="true")
    counts = confusion_matrix(result.y_test_true, result.y_test_pred)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:.2f}\n({counts[i, j]})"
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, cbar=True, vmin=0, vmax=1,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix — {result.config} (test, normalised over true)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_per_class_report(result: TrainResult, label_names: list[str], path: Path) -> None:
    rpt = classification_report(
        result.y_test_true,
        result.y_test_pred,
        target_names=label_names,
        digits=4,
    )
    path.write_text(f"Best config: {result.config}\n\n{rpt}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)
    ensure_dir(ARTIFACTS_DIR)

    train_df = _load_split("train")
    val_df = _load_split("validation")
    test_df = _load_split("test")
    log.info("Loaded splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])
    label_names = list(le.classes_)
    log.info("Classes: %s", label_names)

    results: list[TrainResult] = [
        _run_tfidf_logreg(train_df, val_df, test_df, y_train, y_val, y_test),
        _run_embedding_logreg("mpnet", y_train, y_val, y_test),
        _run_embedding_linsvm("mpnet", y_train, y_val, y_test),
        _run_embedding_mlp("mpnet", y_train, y_val, y_test),
        _run_embedding_logreg("minilm", y_train, y_val, y_test),
        _run_embedding_logreg("e5", y_train, y_val, y_test),
    ]

    # metrics.csv
    metrics_rows = [
        {
            "config": r.config,
            "val_macro_f1": round(r.val_macro_f1, 4),
            "test_acc": round(r.test_acc, 4),
            "test_macro_f1": round(r.test_macro_f1, 4),
            "test_weighted_f1": round(r.test_weighted_f1, 4),
        }
        for r in results
    ]
    metrics_df = pd.DataFrame(metrics_rows).sort_values("val_macro_f1", ascending=False)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Wrote %s", metrics_path)
    print("\n=== metrics.csv ===")
    print(metrics_df.to_string(index=False))

    # Best by val macro-F1
    best = max(results, key=lambda r: r.val_macro_f1)
    log.info("Best config (by val macro-F1): %s (val=%.4f, test_macro_f1=%.4f)",
             best.config, best.val_macro_f1, best.test_macro_f1)

    # Confusion matrix + per-class report for best
    _save_confusion_matrix(best, label_names, RESULTS_DIR / "confusion_matrix.png")
    _save_per_class_report(best, label_names, RESULTS_DIR / "per_class_report.txt")

    # Persist best estimator + label encoder + config
    joblib.dump(best.best_estimator, ARTIFACTS_DIR / _BEST_CLF_FILE)
    joblib.dump(le, ARTIFACTS_DIR / _LABEL_ENCODER_FILE)
    artifact_config = {
        "config": best.config,
        "feature_kind": best.feature_kind,
        "embedder_slug": best.embedder_slug,
        "embedder": EMBEDDERS[best.embedder_slug] if best.embedder_slug else None,
        "labels": label_names,
        "val_macro_f1": best.val_macro_f1,
        "test_acc": best.test_acc,
        "test_macro_f1": best.test_macro_f1,
        "test_weighted_f1": best.test_weighted_f1,
        "seed": SEED,
    }
    (ARTIFACTS_DIR / _BEST_CONFIG_FILE).write_text(
        json.dumps(artifact_config, indent=2), encoding="utf-8"
    )
    log.info("Saved best artefacts to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
