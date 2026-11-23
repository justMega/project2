"""On-board benchmark of the SNN accelerator over UART.

Uses the int16 features cached by benchmark_sw.py (so host preprocessing time is
excluded here and reported separately), sends each frame to the Zynq app
(vitis_part/snn_app/src/helloworld.c) and records per image:

  fpga_us   : PS global-timer time from ap_start to ap_done (IP compute only)
  write_us  : host ser.write()+flush() of the 386-byte request
  rtt_us    : host time from start of write to last reply byte received
  pred, logits

The wire time for the bytes is computed from the baud rate (10 bits per byte, 8N1).
NOTE: the firmware's XTime timer only runs if the A9 global timer is enabled.
The current helloworld.c never starts it (SDT xiltimer starts it lazily on the
first sleep call), so enable it first, e.g. via xsdb:  mwr -force 0xF8F00208 1

Run from py_code/:  .venv\\Scripts\\python.exe eval\\benchmark_fpga.py [--port COM7] [--n N]

Besides fpga_results.json, it also writes results_snn_fpga.csv (one row per image)
and summary_fpga.json in the same format as the CNN client (cnn/pc/shape_client.py),
so cnn/pc/compare_networks.py can merge both networks image by image. In that CSV the
labels, predictions and logit columns use the CNN class order (CSV_CLASSES below).
"""

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import numpy as np
import serial
from common import CLASS_FOLDERS, RESULTS_DIR, IntSNN, per_class_report, stats_us

MAGIC = b"\xaa\x55"
NC = len(CLASS_FOLDERS)
REPLY_LEN = 1 + 1 + 4 + 4 * NC + 1

# Class order of the CNN result files (cnn/python/shapes_common.py: CLASSES)
CSV_CLASSES = ["circle", "square", "fig_eight", "rectangle", "triangle", "elipse"]
missing = set(CLASS_FOLDERS) - set(CSV_CLASSES)
assert not missing, f"SNN class names not found in CSV_CLASSES: {missing}"
TO_CSV = [CSV_CLASSES.index(c) for c in CLASS_FOLDERS]  # SNN index -> CSV index


def write_common_format(out_dir, a, y, preds, logits, paths, rows, wall,
                        tx_bytes, rx_bytes, wire):
    """results_snn_fpga.csv + summary_fpga.json, same columns as the CNN client.

    infer_ns : IP compute time (ap_start -> ap_done) measured on the board
    prep_ns  : 0, the board does no preprocessing (features computed on the PC)
    rx_us    : theoretical wire time of the request (the SNN firmware does not
               time the UART reception itself)
    rtt_ms   : PC round trip, write start -> last reply byte
    status   : 0 = valid reply, 1 = timeout / bad frame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rx_us = int(round(wire(tx_bytes)))
    with open(out_dir / "results_snn_fpga.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "pred", "status", "infer_ns", "prep_ns", "rx_us",
                    "rtt_ms"] + [f"logit_{c}" for c in CSV_CLASSES])
        for i, r in enumerate(rows):
            ok = r is not None
            csv_logits = [0] * NC
            if ok:
                for k in range(NC):
                    csv_logits[TO_CSV[k]] = logits[i][k]
            w.writerow([Path(str(paths[i])).name, TO_CSV[int(y[i])],
                        TO_CSV[int(preds[i])] if ok else -1, 0 if ok else 1,
                        r["infer_ns"] if ok else 0, 0, rx_us if ok else 0,
                        f"{r['rtt_s'] * 1e3:.3f}" if ok else "0"]
                       + [f"{v:.4f}" for v in csv_logits])
    summary = {"wall_time_s": wall, "baud": a.baud, "n_images": len(y),
               "data": "2D_Geometric_Shapes_Dataset_test", "split": "all",
               "request_bytes": tx_bytes, "reply_bytes": rx_bytes,
               "uart_rx_source": "theoretical wire time (not measured on the board)",
               "board_prep": "none, 384 int16 features computed on the PC"}
    (out_dir / "summary_fpga.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved results_snn_fpga.csv and summary_fpga.json in {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM7")
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--n", type=int, default=None, help="limit number of images")
    ap.add_argument("--out", default=str(RESULTS_DIR),
                    help="folder for results_snn_fpga.csv and summary_fpga.json")
    a = ap.parse_args()

    d = np.load(RESULTS_DIR / "test_features.npz")
    Xq, y, paths = d["Xq"], d["y"], d["paths"]
    if a.n:
        idx = np.linspace(0, len(y) - 1, a.n).astype(int)  # spread over classes
        Xq, y, paths = Xq[idx], y[idx], paths[idx]

    ser = serial.Serial(a.port, a.baud, timeout=2, write_timeout=2)
    time.sleep(0.3)
    ser.reset_input_buffer()

    preds, logits, fpga_s, write_s, rtt_s, errors = [], [], [], [], [], 0
    rows = []  # per image: None on error, else {"infer_ns", "rtt_s"}
    t_start = time.perf_counter()
    for i in range(len(y)):
        pkt = MAGIC + Xq[i].astype("<i2").tobytes()
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        ser.write(pkt)
        ser.flush()
        t1 = time.perf_counter()
        r = ser.read(REPLY_LEN)
        t2 = time.perf_counter()
        if len(r) != REPLY_LEN or r[0] != 0xA5 or r[-1] != 0x5A:
            errors += 1
            preds.append(-1)
            logits.append([0] * NC)
            rows.append(None)
            continue
        preds.append(r[1])
        fpga_us = struct.unpack("<I", r[2:6])[0]
        fpga_s.append(fpga_us * 1e-6)
        logits.append(struct.unpack("<" + "i" * NC, r[6:-1]))
        write_s.append(t1 - t0)
        rtt_s.append(t2 - t0)
        rows.append({"infer_ns": fpga_us * 1000, "rtt_s": t2 - t0})
        if (i + 1) % 1000 == 0:
            print(
                f"{i + 1}/{len(y)}  acc so far {np.mean(np.array(preds) == y[: i + 1]):.4f}"
            )
    wall = time.perf_counter() - t_start
    ser.close()

    preds = np.array(preds)
    logits = np.array(logits)
    ref = IntSNN()
    ref_logits = ref.logits(Xq)
    ok = preds >= 0

    tx_bytes, rx_bytes = len(MAGIC) + 2 * Xq.shape[1], REPLY_LEN
    wire = lambda n: n * 10 / a.baud * 1e6
    fpga_mean = float(np.mean(fpga_s)) if fpga_s else float("nan")
    rtt_mean = float(np.mean(rtt_s)) if rtt_s else float("nan")

    res = {
        "conditions": {
            "port": a.port,
            "baud": a.baud,
            "frame": "8N1",
            "n_images": len(y),
            "request_bytes": tx_bytes,
            "reply_bytes": rx_bytes,
            "pl_clock_mhz": 100.0,
            "timer": "A9 global timer (CPU/2 = 325 MHz), 1 us resolution",
            "protocol": "stop-and-wait: one frame in flight",
        },
        "comm_errors": errors,
        "fpga": per_class_report(y[ok], preds[ok]),
        "fpga_vs_int16_model": {
            "prediction_agreement": float(
                (preds[ok] == ref_logits[ok].argmax(1)).mean()
            ),
            "logits_bit_exact": float((logits[ok] == ref_logits[ok]).all(1).mean()),
        },
        "latency": {
            "fpga_compute_us": stats_us(fpga_s),
            "host_write_us": stats_us(write_s),
            "host_round_trip_us": stats_us(rtt_s),
            "uart_wire_time_us": {
                "request": wire(tx_bytes),
                "reply": wire(rx_bytes),
                "total": wire(tx_bytes + rx_bytes),
            },
            "non_compute_overhead_mean_us": (rtt_mean - fpga_mean) * 1e6,
        },
        "throughput": {
            "compute_only_inf_per_s": 1.0 / fpga_mean,
            "end_to_end_over_uart_inf_per_s": len(y) / wall,
            "uart_wire_bound_inf_per_s": 1e6 / wire(tx_bytes + rx_bytes),
        },
    }
    (RESULTS_DIR / "fpga_results.json").write_text(json.dumps(res, indent=2))
    write_common_format(a.out, a, y, preds, logits, paths, rows, wall,
                        tx_bytes, rx_bytes, wire)
    print(
        json.dumps(
            {
                k: res[k]
                for k in ("comm_errors", "fpga_vs_int16_model", "latency", "throughput")
            },
            indent=2,
        )
    )
    print("FPGA accuracy:", res["fpga"]["accuracy"])


if __name__ == "__main__":
    main()
