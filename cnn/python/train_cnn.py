#!/usr/bin/env python3
"""
Train the shape CNN (6 classes), evaluate it on the held-out TEST split and
export everything the HLS accelerator and its testbench need.

Adapted from the course reference (dbaniya/CNN_guest_lecture, train_cnn.py):
  - 60 x 60 grayscale input (square source images, see shapes_common.py)
  - 6 classes, explicit train/val/test split shared with the SNN
  - --arch deep (default, implemented in HLS):
        Conv3x3(16)+BN+ReLU+MaxPool        60x60x1  -> 30x30x16
        DW3x3+BN+ReLU, PW1x1(32)+BN+ReLU, MaxPool -> 15x15x32
        DW3x3+BN+ReLU, PW1x1(32)+BN+ReLU, MaxPool ->  7x7x32
        DW3x3+BN+ReLU, PW1x1(32)+BN+ReLU          ->  7x7x32
        GAP -> Dense 32->6 (logits) -> softmax (training only)
  - --arch lecture: the course network (1 conv + 1 DS block + AvgPool bins),
    kept as a baseline: its 8x8 px receptive field cannot measure aspect
    ratios, so it confuses squares with rectangles (about 78 % test accuracy).

Usage (on the PC; about 1 min per epoch on a laptop CPU):
  python3 train_cnn.py --data /path/to/2D_Geometric_Shapes_Dataset --out runs/cnn
  python3 train_cnn.py --data ... --out runs/lecture --arch lecture   # baseline
  python3 train_cnn.py --data ... --out runs/smoke --quick            # 1-minute pipeline check
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, BatchNormalization,
                                     Activation, MaxPooling2D, AveragePooling2D,
                                     GlobalAveragePooling2D, Dense)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from shapes_common import (CLASSES, N_CLASSES, IMG_H, IMG_W, load_split_u8, cnn_normalize)

SEED = 42
BINS_POOL = (6, 6)          # 30x30 -> 5x5


def build_lecture_model(c1: int, c2: int) -> Sequential:
    """Course architecture (1 conv + 1 DS block, AvgPool bins). Baseline only."""
    bn = dict(momentum=0.9, epsilon=1e-3)
    return Sequential([
        Input(shape=(IMG_H, IMG_W, 1)),
        Conv2D(c1, 3, padding="same", use_bias=False, name="conv1"),
        BatchNormalization(name="bn1", **bn),
        Activation("relu", name="relu1"),
        MaxPooling2D(2, name="pool1"),
        DepthwiseConv2D(3, padding="same", use_bias=False, name="dw3x3"),
        BatchNormalization(name="bn_dw", **bn),
        Activation("relu", name="relu_dw"),
        Conv2D(c2, 1, padding="same", use_bias=False, name="pw1x1"),
        BatchNormalization(name="bn_pw", **bn),
        Activation("relu", name="relu_pw"),
        AveragePooling2D(BINS_POOL, name="pool_bins_5x5"),
        Conv2D(N_CLASSES, 1, padding="same", use_bias=True, name="cls_1x1"),
        GlobalAveragePooling2D(name="gap_logits"),
        Activation("softmax", name="softmax"),
    ])


# Deep variant: conv1 + 3 depthwise-separable blocks. Each MaxPool doubles the
# receptive field, so the last layer sees the whole shape (needed to tell a
# square from a rectangle or a circle from an ellipse).
#   60x60x1 -conv3x3,pool-> 30x30x16 -ds,pool-> 15x15x32 -ds,pool-> 7x7x32 -ds-> 7x7x32
#   -> GAP -> dense 32->6
DEEP_BLOCKS = [(16, 32, True), (32, 32, True), (32, 32, False)]   # (cin, cout, maxpool)


def build_deep_model(c1: int) -> Sequential:
    bn = dict(momentum=0.99, epsilon=1e-3)
    layers = [Input(shape=(IMG_H, IMG_W, 1)),
              Conv2D(c1, 3, padding="same", use_bias=False, name="conv1"),
              BatchNormalization(name="bn1", **bn), 
              Activation("relu", name="relu1"),
              MaxPooling2D(2, name="pool1")]
    for i, (cin, cout, pool) in enumerate(DEEP_BLOCKS, start=2):
        layers += [DepthwiseConv2D(3, padding="same", use_bias=False, name=f"dw{i}"),
                   BatchNormalization(name=f"bn_dw{i}", **bn), Activation("relu", name=f"relu_dw{i}"),
                   Conv2D(cout, 1, use_bias=False, name=f"pw{i}"),
                   BatchNormalization(name=f"bn_pw{i}", **bn), Activation("relu", name=f"relu_pw{i}")]
        if pool:
            layers.append(MaxPooling2D(2, name=f"pool{i}"))
    layers += [GlobalAveragePooling2D(name="gap"),
               Dense(N_CLASSES, name="fc"),
               Activation("linear", name="gap_logits"),
               Activation("softmax", name="softmax")]
    return Sequential(layers)


# ---------------------------------------------------------------- BN folding / export
def fold_bn(layer_bn):
    gamma, beta, mean, var = layer_bn.get_weights()
    scale = gamma / np.sqrt(var + layer_bn.epsilon)
    return scale.astype(np.float32), (beta - mean * scale).astype(np.float32)


def write_c_2d(fname: Path, w: np.ndarray):
    rows = ["{" + ", ".join(repr(float(v)) for v in r) + "}" for r in w]
    fname.write_text("{" + ",\n".join(rows) + "}")


def write_c_1d(fname: Path, v: np.ndarray):
    fname.write_text("{" + ", ".join(repr(float(x)) for x in v) + "}")


def export_weights(model, out_dir: Path, c1: int, c2: int):
    """Same file names / layouts as the lecture so the HLS headers stay familiar."""
    out_dir.mkdir(parents=True, exist_ok=True)

    w1 = model.get_layer("conv1").get_weights()[0]            # (3,3,1,c1)
    s, b = fold_bn(model.get_layer("bn1"))
    w1 = (w1 * s.reshape(1, 1, 1, -1)).reshape(9, c1)          # [kh*3+kw][oc]
    write_c_2d(out_dir / "layer_1_conv1_WEIGHTS.txt", w1)
    write_c_1d(out_dir / "layer_1_conv1_BIAS.txt", b)

    wd = model.get_layer("dw3x3").get_weights()[0]            # (3,3,c1,1)
    s, b = fold_bn(model.get_layer("bn_dw"))
    wd = wd[:, :, :, 0] * s.reshape(1, 1, -1)                  # (3,3,c1)
    wd = np.transpose(wd, (2, 0, 1)).reshape(c1, 9)            # [ic][kh*3+kw]
    write_c_2d(out_dir / "layer_2_dw3x3_WEIGHTS.txt", wd)
    write_c_1d(out_dir / "layer_2_dw3x3_BIAS.txt", b)

    wp = model.get_layer("pw1x1").get_weights()[0]            # (1,1,c1,c2)
    s, b = fold_bn(model.get_layer("bn_pw"))
    wp = (wp * s.reshape(1, 1, 1, -1)).reshape(c1, c2)         # [ic][oc]
    write_c_2d(out_dir / "layer_3_pw1x1_WEIGHTS.txt", wp)
    write_c_1d(out_dir / "layer_3_pw1x1_BIAS.txt", b)

    wc, bc = model.get_layer("cls_1x1").get_weights()         # (1,1,c2,N)
    write_c_2d(out_dir / "layer_4_cls_1x1_WEIGHTS.txt", wc.reshape(c2, N_CLASSES))
    write_c_1d(out_dir / "layer_4_cls_1x1_BIAS.txt", bc)

    # Largest magnitudes: must fit the HLS fixed-point type (ap_fixed<16,8>: |x| < 128)
    rng = {n: float(np.abs(a).max()) for n, a in
           [("conv1", w1), ("dw", wd), ("pw", wp), ("cls", wc)]}
    return rng


def export_weights_deep(model, out_dir: Path, c1: int):
    """BN-folded C-style arrays for the deep model (consumed by hls/ds_cnn_weights_data.h)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = {}
    w = model.get_layer("conv1").get_weights()[0]
    s_, b = fold_bn(model.get_layer("bn1"))
    w = (w * s_.reshape(1, 1, 1, -1)).reshape(9, c1)                  # [kh*3+kw][oc]
    write_c_2d(out_dir / "conv1_W.txt", w); write_c_1d(out_dir / "conv1_B.txt", b)
    rng["conv1"] = float(max(np.abs(w).max(), np.abs(b).max()))
    for i, (cin, cout, _) in enumerate(DEEP_BLOCKS, start=2):
        wd = model.get_layer(f"dw{i}").get_weights()[0][:, :, :, 0]      # (3,3,cin)
        s_, b = fold_bn(model.get_layer(f"bn_dw{i}"))
        wd = np.transpose(wd * s_.reshape(1, 1, -1), (2, 0, 1)).reshape(cin, 9)   # [c][tap]
        write_c_2d(out_dir / f"dw{i}_W.txt", wd); write_c_1d(out_dir / f"dw{i}_B.txt", b)
        rng[f"dw{i}"] = float(max(np.abs(wd).max(), np.abs(b).max()))
        wp = model.get_layer(f"pw{i}").get_weights()[0].reshape(cin, cout)
        s_, b = fold_bn(model.get_layer(f"bn_pw{i}"))
        wp = wp * s_.reshape(1, -1)                                       # [ic][oc]
        write_c_2d(out_dir / f"pw{i}_W.txt", wp); write_c_1d(out_dir / f"pw{i}_B.txt", b)
        rng[f"pw{i}"] = float(max(np.abs(wp).max(), np.abs(b).max()))
    wf, bf = model.get_layer("fc").get_weights()                         # (32,6)
    write_c_2d(out_dir / "fc_W.txt", wf); write_c_1d(out_dir / "fc_B.txt", bf)
    rng["fc"] = float(max(np.abs(wf).max(), np.abs(bf).max()))
    return rng


def activation_ranges(model, x, names):
    """Largest activation of each ReLU output: must fit data_t (ap_fixed<16,8>: < 128)."""
    sub = Model(model.inputs, [model.get_layer(n).output for n in names])
    outs = sub.predict(x, verbose=0, batch_size=256)
    return {n: float(np.abs(o).max()) for n, o in zip(names, outs)}


# ---------------------------------------------------------------- plots
def plot_history(hist, out_dir: Path):
    ep = range(1, len(hist["loss"]) + 1)
    for key, title in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        fig = plt.figure(figsize=(5, 3.5))
        plt.plot(ep, hist[key], marker="o", label=f"Train {title.lower()}")
        plt.plot(ep, hist["val_" + key], marker="x", label=f"Val {title.lower()}")
        plt.xlabel("Epoch"); plt.ylabel(title); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{key}_vs_epoch.png", dpi=150); plt.close(fig)


def plot_cm(y_true, y_pred, out_dir: Path, name: str, title: str):
    labels = list(range(N_CLASSES))
    for norm, fmt, suffix in [(None, "d", "counts"), ("true", ".2f", "normalized")]:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)
        np.savetxt(out_dir / f"{name}_{suffix}.csv", cm, delimiter=",",
                   fmt="%d" if norm is None else "%.4f")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            values_format=fmt, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{title} ({suffix})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(out_dir / f"{name}_{suffix}.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="2D_Geometric_Shapes_Dataset folder")
    ap.add_argument("--out", default=Path("runs/cnn"), type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c1", type=int, default=16, help="conv1 / depthwise channels")
    ap.add_argument("--c2", type=int, default=8, help="pointwise (bottleneck) channels")
    ap.add_argument("--hls-dir", type=Path, default=Path(__file__).resolve().parent.parent / "hls",
                    help="copy weights_cstyle_folded/ and tb_data/ here (empty string to skip)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 300 images/class/split, 2 epochs (checks the pipeline only)")
    ap.add_argument("--arch", choices=["deep", "lecture"], default="deep",
                    help="deep = model implemented in HLS; lecture = course baseline")
    ap.add_argument("--tb-per-class", type=int, default=50,
                    help="test images per class exported for the HLS testbench")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    out = args.out
    dirs = {k: out / k for k in ["models", "plots", "reports", "weights_cstyle_folded", "tb_data"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"

    t0 = time.time()
    xtr, ytr, _ = load_split_u8(args.data, "train", cache)
    xva, yva, _ = load_split_u8(args.data, "val", cache)
    xte, yte, pte = load_split_u8(args.data, "test", cache)
    if args.quick:
        def sub(x, y, p, n=300):
            k = np.concatenate([np.where(y == c)[0][:n] for c in range(N_CLASSES)])
            return x[k], y[k], [p[i] for i in k]
        xtr, ytr, _ = sub(xtr, ytr, list(range(len(ytr))))
        xva, yva, _ = sub(xva, yva, list(range(len(yva))))
        xte, yte, pte = sub(xte, yte, pte)
        args.epochs = 2
    print(f"[DATA] train={len(ytr)} val={len(yva)} test={len(yte)}  ({time.time()-t0:.0f}s)")

    # shuffle training set once (reproducible)
    perm = np.random.permutation(len(ytr)); xtr, ytr = xtr[perm], ytr[perm]
    f = lambda x: cnn_normalize(x)[..., None]

    model = build_deep_model(args.c1) if args.arch == "deep" else build_lecture_model(args.c1, args.c2)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(f(xtr), ytr, validation_data=(f(xva), yva), epochs=args.epochs,
                     batch_size=args.batch, callbacks=cbs, verbose=2).history
    train_s = time.time() - t0
    hist = {k: [float(v) for v in vs] for k, vs in hist.items()}
    (dirs["reports"] / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, dirs["plots"])

    # ---- evaluation on the TEST split (never seen during training/early stopping)
    logits_model = Model(model.inputs, model.get_layer("gap_logits").output)
    logits = logits_model.predict(f(xte), verbose=0, batch_size=256)
    pred = logits.argmax(1)
    acc = float((pred == yte).mean())
    report = classification_report(yte, pred, target_names=CLASSES, digits=4)
    print("\n[TEST] accuracy = %.4f\n%s" % (acc, report))
    (dirs["reports"] / "classification_report_test.txt").write_text(
        f"Test accuracy: {acc:.4f}\n\n{report}")
    plot_cm(yte, pred, dirs["plots"], "confusion_test", "Keras float CNN - test set")
    with open(dirs["reports"] / "test_predictions_keras.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "pred"] + [f"logit_{c}" for c in CLASSES])
        for p, t, pr, lg in zip(pte, yte, pred, logits):
            w.writerow([Path(p).name, int(t), int(pr)] + [f"{v:.6f}" for v in lg])

    # ---- models + HLS weights
    model.save(dirs["models"] / "shape_cnn.keras")
    if args.arch == "deep":
        rng = export_weights_deep(model, dirs["weights_cstyle_folded"], args.c1)
        relus = ["relu1"] + [f"relu_{k}{i}" for i in range(2, 2 + len(DEEP_BLOCKS)) for k in ("dw", "pw")]
        act = activation_ranges(model, f(xtr[:3000]), relus + ["gap_logits"])
    else:
        rng = export_weights(model, dirs["weights_cstyle_folded"], args.c1, args.c2)
        act = {}
    print("[EXPORT] max |w| per layer:", rng)
    print("[EXPORT] max |activation| per layer:", act)

    # ---- testbench data: first N test images of each class (uint8 + Keras logits)
    sel = np.concatenate([np.where(yte == c)[0][:args.tb_per_class] for c in range(N_CLASSES)])
    xte[sel].astype(np.uint8).tofile(dirs["tb_data"] / "images_u8.bin")
    yte[sel].astype(np.int32).tofile(dirs["tb_data"] / "labels_i32.bin")
    logits[sel].astype(np.float32).tofile(dirs["tb_data"] / "keras_logits_f32.bin")
    (dirs["tb_data"] / "README.txt").write_text(
        f"{len(sel)} test images ({args.tb_per_class}/class), {IMG_H}x{IMG_W} uint8 row-major,\n"
        f"labels int32, Keras float logits float32 [{N_CLASSES}] per image.\n")

    params = int(model.count_params())
    info = dict(arch=args.arch, classes=CLASSES, img=[IMG_H, IMG_W], c1=args.c1, c2=args.c2,
                bins_pool=list(BINS_POOL), epochs_run=len(hist["loss"]), batch=args.batch,
                lr=args.lr, seed=SEED, params_total=params, train_seconds=round(train_s, 1),
                test_accuracy=acc, n_train=int(len(ytr)), n_val=int(len(yva)),
                n_test=int(len(yte)), max_abs_weight=rng, max_abs_activation=act)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    if args.arch == "deep" and str(args.hls_dir) not in ("", "."):
        import shutil
        for name in ("weights_cstyle_folded", "tb_data"):
            dst = args.hls_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(dirs[name], dst)
        print(f"[HLS] weights and testbench data copied to {args.hls_dir}")
    lim = {"weights (wgt_t, +-64)": max(rng.values()), "activations (feat_t, +-128)": max(act.values()) if act else 0}
    for k, v in lim.items():
        if v > 0.9 * (64 if "weights" in k else 128):
            print(f"[WARNING] max {k} = {v:.1f}: close to the fixed-point range, widen the type in hls/ds_cnn.h")


if __name__ == "__main__":
    main()
