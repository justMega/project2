"""Software reference benchmark (host CPU).

  * float32 PyTorch/snnTorch model (the trained network)
  * bit-exact int16 model of the HLS kernel (what the FPGA should compute)

Measures accuracy (overall + per class), preprocessing time, and CPU inference
latency / throughput. Writes results/sw_results.json and results/test_features.npz
(cached int16 features + labels, reused by benchmark_fpga.py).

Run from py_code/:  .venv\\Scripts\\python.exe eval\\benchmark_sw.py
"""

import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
from common import (
    CLASS_FOLDERS,
    RESULTS_DIR,
    STATE_DICT,
    TEST_ROOT,
    TRAIN_ROOT,
    IntSNN,
    features_float,
    list_test_images,
    per_class_report,
    quantize_features,
    stats_us,
)

from snn_mlp import N_FEATURES, CableSNNMLP

WARMUP = 200
LAT_REPS = 2000  # batch-1 latency samples
TPUT_BATCH = 1024


def build_model():
    m = CableSNNMLP(input_size=N_FEATURES, num_classes=len(CLASS_FOLDERS))
    m.load_state_dict(torch.load(STATE_DICT, map_location="cpu"))
    m.eval()
    return m


def time_batch1(fn, xs):
    for i in range(WARMUP):
        fn(xs[i % len(xs)])
    t = []
    for i in range(LAT_REPS):
        x = xs[i % len(xs)]
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return stats_us(t)


def time_throughput(fn, X, reps=3):
    fn(X[:TPUT_BATCH])
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(0, len(X), TPUT_BATCH):
            fn(X[i : i + TPUT_BATCH])
        best = max(best, len(X) / (time.perf_counter() - t0))
    return best


def image_hashes(root):
    import cv2

    out = set()
    for c in CLASS_FOLDERS:
        for p in (root / c).iterdir():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            out.add(hashlib.md5(img.tobytes()).hexdigest())
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items = list_test_images()
    y = np.array([l for _, l in items])
    print(f"Test images: {len(items)} from {TEST_ROOT}")

    # ---- preprocessing (imread + Sobel ON/OFF + 12x16 pooling) ----
    Xf, pre_t = [], []
    for p, _ in items:
        t0 = time.perf_counter()
        Xf.append(features_float(p))
        pre_t.append(time.perf_counter() - t0)
    Xf = np.stack(Xf).astype(np.float32)
    Xq = quantize_features(Xf)
    np.savez_compressed(
        RESULTS_DIR / "test_features.npz",
        Xq=Xq,
        y=y,
        paths=np.array([str(p.relative_to(TEST_ROOT)) for p, _ in items]),
    )

    # ---- float model ----
    model = build_model()
    Xt = torch.from_numpy(Xf).unsqueeze(1)  # (N, T=1, 384)
    with torch.no_grad():
        float_logits = model(Xt).numpy()
    float_pred = float_logits.argmax(1)

    # ---- int16 bit-exact model ----
    snn = IntSNN()
    int_logits = snn.logits(Xq)
    int_pred = int_logits.argmax(1)

    # ---- CPU timing ----
    def torch_fn(x):
        with torch.no_grad():
            return model(x)

    xs1 = [Xt[i : i + 1] for i in range(512)]
    timing = {}
    for nthreads in (1, os.cpu_count()):
        torch.set_num_threads(nthreads)
        timing[f"torch_float_threads{nthreads}"] = {
            "batch1_latency_us": time_batch1(torch_fn, xs1),
            "throughput_inf_per_s_batch1024": time_throughput(torch_fn, Xt),
        }
    xq1 = [Xq[i : i + 1] for i in range(512)]
    timing["numpy_int16_model"] = {
        "batch1_latency_us": time_batch1(snn.predict, xq1),
        "throughput_inf_per_s_batch1024": time_throughput(snn.predict, Xq),
    }

    # ---- train/test overlap (identical pixel content) ----
    print("Hashing train/test images for overlap check...")
    test_h, train_h = image_hashes(TEST_ROOT), image_hashes(TRAIN_ROOT)

    res = {
        "conditions": {
            "host": platform.processor(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "logical_cpus": os.cpu_count(),
            "test_set": str(TEST_ROOT),
            "n_test": len(items),
            "model": "SNN MLP 384-64-32-6, LIF, T=1, inference on final LIF3 membrane",
            "float_inference": "snnTorch forward, no early exit",
            "warmup": WARMUP,
            "latency_reps": LAT_REPS,
            "throughput_batch": TPUT_BATCH,
        },
        "preprocessing_us": stats_us(pre_t),
        "float32": per_class_report(y, float_pred),
        "int16_bit_exact": per_class_report(y, int_pred),
        "float_vs_int_prediction_agreement": float((float_pred == int_pred).mean()),
        "timing": timing,
        "data_overlap": {
            "unique_test_images": len(test_h),
            "unique_train_images": len(train_h),
            "test_images_identical_to_a_train_image": len(test_h & train_h),
        },
    }
    (RESULTS_DIR / "sw_results.json").write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "preprocessing_us",
                    "timing",
                    "data_overlap",
                    "float_vs_int_prediction_agreement",
                )
            },
            indent=2,
        )
    )
    print(
        "float acc",
        res["float32"]["accuracy"],
        "int acc",
        res["int16_bit_exact"]["accuracy"],
    )


if __name__ == "__main__":
    main()
