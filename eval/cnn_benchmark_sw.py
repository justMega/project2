"""Software reference benchmark of the CNN (host CPU), counterpart of benchmark_sw.py.

  * float32 Keras model (the trained network, cnn/runs/cnn/models/shape_cnn.keras)

Measures accuracy (overall + per class), host preprocessing time, and CPU inference
latency / throughput with 1 thread and with all threads. Checks that no test image
is pixel-identical to a training image. Writes results_cnn/sw_results.json and
results_cnn/test_inputs.npz (60x60 uint8 images + labels + float logits, reused by
cnn_benchmark_fpga.py).

There is no Python bit-exact model of the fixed-point CNN: the fixed-point design was
checked against Keras by the HLS C-simulation (cnn/hls/ds_cnn_tb.cpp), and the FPGA
results are compared with the float model in cnn_benchmark_fpga.py.

Run from assignment2/ with the CNN venv:
    source cnn/.venv/bin/activate
    python3 eval/cnn_benchmark_sw.py
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

WARMUP = 200
LAT_REPS = 2000  # batch-1 latency samples
TPUT_BATCH = 1024


def load_model(threads):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    from cnn_common import MODEL_PATH

    return tf, tf.keras.models.load_model(MODEL_PATH)


def timing_child(threads):
    """Runs in a separate process: TensorFlow's thread count is fixed at start-up."""
    from cnn_common import RESULTS_DIR, cnn_normalize, stats_us

    tf, model = load_model(threads)
    X = cnn_normalize(np.load(RESULTS_DIR / "test_inputs.npz")["x"])[..., None]
    fn = lambda x: model(x, training=False)  # noqa: E731

    xs = [tf.constant(X[i : i + 1]) for i in range(512)]
    for i in range(WARMUP):
        fn(xs[i % len(xs)])
    t = []
    for i in range(LAT_REPS):
        t0 = time.perf_counter()
        fn(xs[i % len(xs)])
        t.append(time.perf_counter() - t0)

    fn(tf.constant(X[:TPUT_BATCH]))
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(0, len(X), TPUT_BATCH):
            fn(tf.constant(X[i : i + TPUT_BATCH]))
        best = max(best, len(X) / (time.perf_counter() - t0))
    print(json.dumps({"batch1_latency_us": stats_us(t),
                      "throughput_inf_per_s_batch1024": best,
                      "tensorflow": tf.__version__}))


def image_hashes(paths):
    import cv2

    out = set()
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out.add(hashlib.md5(img.tobytes()).hexdigest())
    return out


def main():
    from cnn_common import (
        RESULTS_DIR, TEST_ROOT, TRAIN_ROOT, MODEL_PATH, cnn_normalize, list_split,
        list_test_images, load_u8, per_class_report, stats_us, to_report,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths, y = list_test_images()
    print(f"Test images: {len(paths)} from {TEST_ROOT}")

    # ---- host preprocessing (imread + INTER_AREA resize to 60x60; this is what the
    #      PC does before sending; the normalisation to Q8.8 is done on the board) ----
    X, pre_t = [], []
    for p in paths:
        t0 = time.perf_counter()
        X.append(load_u8(p))
        pre_t.append(time.perf_counter() - t0)
    X = np.stack(X)

    # ---- float model ----
    tf, model = load_model(0)
    # logits = output of the Dense layer "fc", before the softmax (what the FPGA returns)
    logit_model = tf.keras.Model(model.inputs, model.get_layer("fc").output)
    float_logits = logit_model.predict(cnn_normalize(X)[..., None], batch_size=256, verbose=0)
    float_pred = float_logits.argmax(1)
    np.savez_compressed(
        RESULTS_DIR / "test_inputs.npz",
        x=X, y=y, float_logits=float_logits,
        paths=np.array([str(p.relative_to(TEST_ROOT)) for p in paths]),
    )
    tf_version = tf.__version__
    del model, logit_model

    # ---- CPU timing, one process per thread setting ----
    timing = {}
    for n in (1, os.cpu_count()):
        print(f"CPU timing with {n} thread(s)...")
        out = subprocess.run([sys.executable, __file__, "--timing-child", str(n)],
                             capture_output=True, text=True, check=True).stdout
        r = json.loads(out.strip().splitlines()[-1])
        r.pop("tensorflow", None)
        timing[f"keras_float_threads{n}"] = r

    # ---- train/test overlap (identical pixel content) ----
    print("Hashing train/test images for overlap check...")
    train_paths, _ = list_split(TRAIN_ROOT, "all")
    test_h, train_h = image_hashes(paths), image_hashes(train_paths)

    yr = to_report(y)
    res = {
        "conditions": {
            "host": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "tensorflow": tf_version,
            "logical_cpus": os.cpu_count(),
            "test_set": str(TEST_ROOT),
            "n_test": len(paths),
            "model": "CNN: Conv3x3(16)+3 depthwise-separable blocks (32)+GAP+Dense(6), "
                     "60x60 grayscale input, 4390 parameters (3814 after BN folding)",
            "model_file": str(MODEL_PATH),
            "float_inference": "Keras model(x, training=False), eager",
            "class_order": "circle, elipse, fig_eight, rectangle, square, triangle (as the SNN)",
            "warmup": WARMUP,
            "latency_reps": LAT_REPS,
            "throughput_batch": TPUT_BATCH,
        },
        "preprocessing_us": stats_us(pre_t),
        "float32": per_class_report(yr, to_report(float_pred)),
        "fixed_point_model": "no Python bit-exact model; HLS C-simulation (ds_cnn_tb.cpp) "
                             "agrees with Keras on 100 % of the testbench images",
        "timing": timing,
        "data_overlap": {
            "unique_test_images": len(test_h),
            "unique_train_images": len(train_h),
            "test_images_identical_to_a_train_image": len(test_h & train_h),
        },
    }
    (RESULTS_DIR / "sw_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("preprocessing_us", "timing", "data_overlap")},
                     indent=2))
    print("float acc", res["float32"]["accuracy"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-child", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.timing_child is not None:
        timing_child(a.timing_child)
    else:
        main()
