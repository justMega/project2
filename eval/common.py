"""Shared helpers for the SNN benchmark scripts.

Feature extraction is imported from snn_mlp.py (the training code) so every
benchmark uses exactly the preprocessing the network was trained with.
The integer model reproduces vitis_part/snn_component/snn_mlp.cpp bit-exactly.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # repository root
PY_CODE = ROOT / "snn" / "python"                    # snn_mlp.py and snn_mlp_model/
sys.path.insert(0, str(PY_CODE))

from snn_mlp import (
    CLASS_FOLDERS,
    image_to_spikes,
    spikes_to_snn_mlp_features,
)

TEST_ROOT = ROOT / "2D_Geometric_Shapes_Dataset_test"
TRAIN_ROOT = ROOT / "shapes-dataset" / "2D_Geometric_Shapes_Dataset"
EXPORT_DIR = PY_CODE / "snn_mlp_model" / "models" / "fpga_export"
STATE_DICT = PY_CODE / "snn_mlp_model" / "models" / "snn_mlp_dict.pt"
RESULTS_DIR = ROOT / "eval" / "results"

FEATURE_SCALE = 128  # features are sent as round(x * 128) int16
LIF_THRESHOLD = 128  # WEIGHT_SCALE in fpga_weights.h
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_test_images(root=TEST_ROOT, per_class=None):
    """Sorted (path, label) list, same ordering as send_images_uart_snn.py."""
    items = []
    for label, cname in enumerate(CLASS_FOLDERS):
        files = sorted(
            p for p in (root / cname).iterdir() if p.suffix.lower() in IMG_EXT
        )
        if per_class is not None:
            files = files[:per_class]
        items += [(p, label) for p in files]
    return items


def features_float(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return spikes_to_snn_mlp_features(image_to_spikes(img)).reshape(-1)


def quantize_features(f):
    return np.round(f * FEATURE_SCALE).astype(np.int16)


class IntSNN:
    """Bit-exact model of snn_mlp.cpp for T_STEPS = 1.

    With T=1 the membranes start at 0, so the leak (beta=115/128) and the
    delayed reset never act: mem = input current, spike = mem > threshold.
    """

    def __init__(self, export_dir=EXPORT_DIR):
        ld = lambda n: np.load(export_dir / f"{n}_int16.npy").astype(np.int64)
        self.w1, self.b1 = ld("fc1_w"), ld("fc1_b")
        self.w2, self.b2 = ld("fc2_w"), ld("fc2_b")
        self.w3, self.b3 = ld("fc3_w"), ld("fc3_b")

    def logits(self, x_int16):
        x = np.atleast_2d(x_int16).astype(np.int64)  # (N, 384)
        s = x @ self.w1.T  # full-precision sum
        cur1 = self.b1 + ((s + 64) >> 7)  # arithmetic shift
        spk1 = (cur1 > LIF_THRESHOLD).astype(np.int64)
        cur2 = self.b2 + spk1 @ self.w2.T
        spk2 = (cur2 > LIF_THRESHOLD).astype(np.int64)
        mem3 = self.b3 + spk2 @ self.w3.T
        return mem3

    def predict(self, x_int16):
        return np.argmax(self.logits(x_int16), axis=1)  # first max, as HLS


def per_class_report(y_true, y_pred, n_classes=len(CLASS_FOLDERS)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= p < n_classes:
            cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(CLASS_FOLDERS):
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
