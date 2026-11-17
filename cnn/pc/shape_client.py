#!/usr/bin/env python3
"""
PC application for Assignment 2: send shape images to the PYNQ-Z2 over UART,
collect the CNN / SNN predictions and compare them.

Modes
  uart-test : loopback frames (no hardware IP involved) -> checks every byte
              and measures the pure UART transfer time.
  eval      : runs the whole TEST split (or --max-per-class) through one or
              both networks, prints overall / per-class accuracy, confusion
              matrices, latency and throughput, and saves CSV + JSON + plots.
  demo      : sends a few test images and displays each one with its true
              label and the class predicted by every network.

Examples
  python3 shape_client.py uart-test --port /dev/ttyUSB1
  python3 shape_client.py eval --port /dev/ttyUSB1 --data ../../shapes-dataset/2D_Geometric_Shapes_Dataset --net cnn
  python3 shape_client.py demo --port /dev/ttyUSB1 --data ... --net cnn snn
"""
import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from shapes_common import CLASSES, N_CLASSES, IMG_H, IMG_W, list_split, load_u8  # noqa: E402

N_PIX = IMG_H * IMG_W
NETS = {"cnn": 0x00, "snn": 0x01}
NET_ECHO = 0x7F
STATUS = {0: "OK", 1: "bad length", 2: "bad checksum", 3: "network not in bitstream", 4: "timeout"}
REPLY_LEN = 29
BITS_PER_BYTE = 10          # 8N1


# ------------------------------------------------------------------ framing
def build_frame(net: int, px: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(px, dtype=np.uint8).reshape(-1).tobytes()
    assert len(payload) == N_PIX
    return (b"\xA6\x6A" + bytes([net]) + struct.pack("<H", len(payload)) + payload
            + bytes([sum(payload) & 0xFF]))


def read_exact(ser, n: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"timeout: got {len(buf)}/{n} bytes")
        buf += ser.read(n - len(buf))
    return bytes(buf)


def wait_start(ser, deadline: float):
    # skip anything (e.g. ASCII debug prints) until the reply start byte
    while True:
        b = read_exact(ser, 1, deadline)
        if b[0] == 0xA5:
            return


def infer(ser, net: int, px: np.ndarray, timeout: float = 3.0) -> dict:
    frame = build_frame(net, px)
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(frame)
    ser.flush()
    deadline = t0 + timeout
    wait_start(ser, deadline)
    body = read_exact(ser, REPLY_LEN - 1, deadline)
    rtt = time.perf_counter() - t0
    if body[-1] != 0x5A:
        raise RuntimeError(f"bad reply end byte 0x{body[-1]:02X}")
    net_r, status, pred, infer_ns, prep_ns, rx_us = struct.unpack("<BBBIII", body[:15])
    logits = np.array(struct.unpack("<6h", body[15:27]), dtype=np.float32) / 256.0
    if net_r != net:
        raise RuntimeError(f"reply for net {net_r}, expected {net}")
    return dict(status=status, pred=pred, infer_ns=infer_ns, prep_ns=prep_ns,
                rx_us=rx_us, logits=logits, rtt_s=rtt)


def echo(ser, px: np.ndarray, timeout: float = 3.0) -> dict:
    frame = build_frame(NET_ECHO, px)
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(frame)
    ser.flush()
    deadline = t0 + timeout
    wait_start(ser, deadline)
    body = read_exact(ser, 2 + 4 + N_PIX + 1, deadline)
    rtt = time.perf_counter() - t0
    net_r, status, rx_us = struct.unpack("<BBI", body[:6])
    back = np.frombuffer(body[6:6 + N_PIX], dtype=np.uint8)
    ok = (net_r == NET_ECHO and status == 0 and body[-1] == 0x5A
          and np.array_equal(back, px.reshape(-1)))
    return dict(ok=ok, status=status, rx_us=rx_us, rtt_s=rtt,
                n_bad=int((back != px.reshape(-1)).sum()))


def open_port(args):
    ser = serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=5,
                        xonxoff=False, rtscts=False, dsrdtr=False)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


# ------------------------------------------------------------------ modes
def mode_uart_test(args):
    rng = np.random.default_rng(0)
    ser = open_port(args)
    frame_bytes = len(build_frame(NET_ECHO, np.zeros(N_PIX, np.uint8)))
    theo_ms = frame_bytes * BITS_PER_BYTE / args.baud * 1e3
    print(f"Loopback test: {args.n} frames of {frame_bytes} B at {args.baud} baud "
          f"(theoretical {theo_ms:.2f} ms per direction)")
    rx_us, rtt, fails = [], [], 0
    try:
        for i in range(args.n):
            px = rng.integers(0, 256, N_PIX, dtype=np.uint8)   # every byte value used
            r = echo(ser, px)
            if not r["ok"]:
                fails += 1
                print(f"  frame {i}: FAILED (status={STATUS.get(r['status'])}, "
                      f"{r['n_bad']} wrong bytes)")
            rx_us.append(r["rx_us"]); rtt.append(r["rtt_s"] * 1e3)
    finally:
        ser.close()
    print(f"\n{args.n - fails}/{args.n} frames echoed without error")
    print(f"board-side RX time : mean {np.mean(rx_us)/1e3:.2f} ms")
    print(f"PC round trip      : mean {np.mean(rtt):.2f} ms (= request + echo of {frame_bytes} B)")
    return 0 if fails == 0 else 1


def collect_images(args):
    paths, labels = list_split(args.data, args.split)
    if args.max_per_class:
        keep = np.concatenate([np.where(labels == c)[0][:args.max_per_class]
                               for c in range(N_CLASSES)])
        paths = [paths[i] for i in keep]; labels = labels[keep]
    return paths, labels


def summarize(name, y, pred, rows, out_dir: Path):
    from sklearn.metrics import confusion_matrix, classification_report
    ok = pred >= 0
    if not ok.any():
        status = sorted({STATUS.get(r["status"], str(r["status"])) for r in rows})
        print(f"\n================ {name.upper()} ================\n"
              f"no valid reply ({', '.join(status)}) - skipped")
        return dict(net=name, n=0, errors=int(len(rows)), status=status)
    acc = float((pred[ok] == y[ok]).mean())
    cm = confusion_matrix(y[ok], pred[ok], labels=list(range(N_CLASSES)))
    per_class = {CLASSES[c]: float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None
                 for c in range(N_CLASSES)}
    inf = np.array([r["infer_ns"] for r in rows if r["status"] == 0], float) / 1e3
    prep = np.array([r["prep_ns"] for r in rows if r["status"] == 0], float) / 1e3
    rx = np.array([r["rx_us"] for r in rows if r["status"] == 0], float)
    rtt = np.array([r["rtt_s"] for r in rows if r["status"] == 0], float) * 1e3
    s = dict(net=name, n=int(ok.sum()), errors=int((~ok).sum()), accuracy=acc,
             per_class_accuracy=per_class, confusion=cm.tolist(),
             infer_us_mean=float(inf.mean()), infer_us_std=float(inf.std()),
             infer_us_min=float(inf.min()), infer_us_max=float(inf.max()),
             accel_throughput_img_s=float(1e6 / inf.mean()),
             cpu_prep_us_mean=float(prep.mean()),
             uart_rx_ms_mean=float(rx.mean() / 1e3),
             pc_round_trip_ms_mean=float(rtt.mean()),
             end_to_end_throughput_img_s=float(1e3 / rtt.mean()))
    print(f"\n================ {name.upper()} ================")
    print(f"images {s['n']}  (failed replies: {s['errors']})")
    print(f"overall accuracy        : {acc*100:.2f} %")
    for c, a in per_class.items():
        print(f"   {c:<10s}: {a*100:6.2f} %" if a is not None else f"   {c:<10s}:   n/a")
    print(f"accelerator latency     : {s['infer_us_mean']:.1f} us "
          f"(min {s['infer_us_min']:.1f}, max {s['infer_us_max']:.1f})")
    print(f"accelerator throughput  : {s['accel_throughput_img_s']:.0f} img/s")
    print(f"CPU preprocessing       : {s['cpu_prep_us_mean']:.1f} us")
    print(f"UART transfer (board RX): {s['uart_rx_ms_mean']:.2f} ms")
    print(f"PC round trip           : {s['pc_round_trip_ms_mean']:.2f} ms "
          f"-> {s['end_to_end_throughput_img_s']:.1f} img/s end-to-end")
    print("confusion (rows=true, cols=pred):")
    print("            " + " ".join(f"{c[:6]:>6s}" for c in CLASSES))
    for c in range(N_CLASSES):
        print(f"  {CLASSES[c]:<10s}" + " ".join(f"{v:6d}" for v in cm[c]))
    print(classification_report(y[ok], pred[ok], labels=list(range(N_CLASSES)),
                                target_names=CLASSES, digits=4, zero_division=0))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(values_format="d", cmap="Blues",
                                                              ax=ax, colorbar=False)
        ax.set_title(f"{name.upper()} on FPGA - test set ({acc*100:.2f} %)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(out_dir / f"confusion_{name}_fpga.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    return s


def mode_eval(args):
    paths, labels = collect_images(args)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ser = open_port(args)
    if not paths:
        print(f"No image found in {args.data} for --split {args.split} "
              "(a separate test set needs --split all)")
        return 1
    print(f"Evaluating {len(paths)} images ({args.split} split of {args.data}) on: {', '.join(args.net)}")
    results = {n: [] for n in args.net}
    t_start = time.perf_counter()
    try:
        for i, (p, y) in enumerate(zip(paths, labels)):
            px = load_u8(p)
            for n in args.net:
                try:
                    r = infer(ser, NETS[n], px)
                except (TimeoutError, RuntimeError) as e:
                    print(f"  {p.name} [{n}]: {e}")
                    r = dict(status=255, pred=-1, infer_ns=0, prep_ns=0, rx_us=0,
                             logits=np.zeros(N_CLASSES), rtt_s=0)
                    ser.reset_input_buffer()
                if r["status"] != 0:
                    r["pred"] = -1
                r.update(file=p.name, true=int(y))
                results[n].append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(paths)} images")
    finally:
        ser.close()
    wall = time.perf_counter() - t_start

    summary = {"wall_time_s": wall, "baud": args.baud, "n_images": len(paths),
               "data": str(args.data), "split": args.split}
    for n, rows in results.items():
        y = np.array([r["true"] for r in rows]); pred = np.array([r["pred"] for r in rows])
        summary[n] = summarize(n, y, pred, rows, out_dir)
        with open(out_dir / f"results_{n}_fpga.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "true", "pred", "status", "infer_ns", "prep_ns", "rx_us",
                        "rtt_ms"] + [f"logit_{c}" for c in CLASSES])
            for r in rows:
                w.writerow([r["file"], r["true"], r["pred"], r["status"], r["infer_ns"],
                            r["prep_ns"], r["rx_us"], f"{r['rtt_s']*1e3:.3f}"]
                           + [f"{v:.4f}" for v in r["logits"]])
    (out_dir / "summary_fpga.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results in {out_dir}/")
    return 0


def mode_demo(args):
    import matplotlib.pyplot as plt
    paths, labels = list_split(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], args.per_class, replace=False)
                          for c in range(N_CLASSES)])
    ser = open_port(args)
    cols = N_CLASSES
    rows_n = args.per_class
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.3 * cols, 2.6 * rows_n), squeeze=False)
    try:
        for k, i in enumerate(idx):
            px = load_u8(paths[i])
            lines = [f"true: {CLASSES[labels[i]]}"]
            all_ok = True
            for n in args.net:
                r = infer(ser, NETS[n], px)
                pred = CLASSES[r["pred"]] if r["status"] == 0 else STATUS.get(r["status"])
                good = r["status"] == 0 and r["pred"] == labels[i]
                all_ok &= good
                lines.append(f"{n.upper()}: {pred} {'✓' if good else '✗'} "
                             f"({r['infer_ns']/1e3:.0f} µs)")
                print(f"{paths[i].name:<22s} {lines[-1]}")
            ax = axes[k % rows_n][k // rows_n]
            ax.imshow(px, cmap="gray", vmin=0, vmax=255)
            ax.set_title("\n".join(lines), fontsize=8, color="green" if all_ok else "red")
            ax.axis("off")
    finally:
        ser.close()
    plt.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "demo_predictions.png", dpi=150)
    print(f"saved {out/'demo_predictions.png'}")
    if not args.no_show:
        plt.show()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["uart-test", "eval", "demo"])
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--data", type=Path, help="2D_Geometric_Shapes_Dataset folder")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"],
                    help="images to use: test = indices 8501-10000 of the training dataset; "
                         "all = every image in --data (use it for a separate test set)")
    ap.add_argument("--net", nargs="+", default=["cnn"], choices=list(NETS))
    ap.add_argument("--n", type=int, default=50, help="frames for uart-test")
    ap.add_argument("--max-per-class", type=int, default=0, help="eval: 0 = whole test split")
    ap.add_argument("--per-class", type=int, default=2, help="demo: images per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.mode in ("eval", "demo") and args.data is None:
        ap.error("--data is required for eval/demo")
    return {"uart-test": mode_uart_test, "eval": mode_eval, "demo": mode_demo}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
