"""Shared helpers for the CNN benchmark scripts (cnn_benchmark_sw.py, cnn_benchmark_fpga.py).

Same output format as the SNN scripts (common.py / benchmark_sw.py / benchmark_fpga.py):
per_class_report() and stats_us() are identical, and every per-class table and
confusion matrix uses the SNN class order (REPORT_CLASSES), so the two networks'
JSON files can be compared key by key.

Image loading and the CNN normalisation are imported from cnn/python/shapes_common.py,
the code the network was trained with.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # assignment2/
sys.path.insert(0, str(ROOT / "cnn" / "python"))
sys.path.insert(0, str(ROOT / "cnn" / "pc"))

from shapes_common import CLASSES, IMG_H, IMG_W, cnn_normalize, list_split, load_u8  # noqa: E402,F401

TEST_ROOT = ROOT / "2D_Geometric_Shapes_Dataset_test"
TRAIN_ROOT = ROOT / "shapes-dataset" / "2D_Geometric_Shapes_Dataset"
MODEL_PATH = ROOT / "cnn" / "runs" / "cnn" / "models" / "shape_cnn.keras"
RESULTS_DIR = ROOT / "eval" / "results_cnn"

# Class order used in every report (same as the SNN: CLASS_FOLDERS in snn_mlp.py)
REPORT_CLASSES = ["circle", "elipse", "fig_eight", "rectangle", "square", "triangle"]
_TO_REPORT = np.array([REPORT_CLASSES.index(c) for c in CLASSES])


def to_report(idx):
    """CNN class indices (CLASSES order) -> REPORT_CLASSES indices. -1 stays -1."""
    idx = np.asarray(idx)
    out = np.full(idx.shape, -1, dtype=np.int64)
    ok = idx >= 0
    out[ok] = _TO_REPORT[idx[ok]]
    return out


def list_test_images(root=TEST_ROOT):
    """All test images as (paths, labels in CNN order)."""
    return list_split(root, "all")


def per_class_report(y_true, y_pred, n_classes=len(REPORT_CLASSES)):
    """Inputs in REPORT_CLASSES order (use to_report first). Same output as common.py."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= p < n_classes:
            cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(REPORT_CLASSES):
        tp = cm[i, i]
        support = cm[i].sum()
        predicted = cm[:, i].sum()
        rec = tp / support if support else float("nan")
        prec = tp / predicted if predicted else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "support": int(support),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        }
    return {
        "n": len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": per_class,
        "confusion": cm.tolist(),
    }


def stats_us(samples_s):
    a = np.asarray(samples_s, dtype=np.float64) * 1e6
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
