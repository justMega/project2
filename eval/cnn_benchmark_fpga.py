"""On-board benchmark of the CNN accelerator over UART, counterpart of benchmark_fpga.py.

Uses the 60x60 images cached by cnn_benchmark_sw.py (so host preprocessing time is
excluded here and reported separately), sends each one to the Zynq app
(cnn/vitis/main.c, protocol in cnn/vitis/PROTOCOL.md) and records per image:

  fpga_us     : PS global-timer time from ap_start to ap_done (IP compute only, 1 us resolution)
  write_us    : host ser.write()+flush() of the 3606-byte request
  rtt_us      : host time from start of write to last reply byte received
  pred, logits

The reply has the same format as the SNN's (31 bytes):
  0xA5 <pred> <time_us u32> <6 x logits i32, Q8.8> 0x5A
The wire time for the bytes is computed from the baud rate (10 bits per byte, 8N1).
The CNN firmware starts the A9 global timer itself (no xsdb command needed).

Writes results_cnn/fpga_results.json (same keys as the SNN's fpga_results.json) and,
for cnn/pc/compare_networks.py, results_cnn/results_cnn_fpga.csv + summary_fpga.json.

Run from assignment2/ with the CNN bitstream + app running on the board:
    python3 eval/cnn_benchmark_fpga.py --port /dev/ttyUSB1 [--n N]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import serial
from cnn_common import CLASSES, RESULTS_DIR, per_class_report, stats_us, to_report
from shape_client import NETS, REPLY_LEN, build_frame, parse_reply, read_exact, wait_start

NET = NETS["cnn"]
NC = len(CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--n", type=int, default=None, help="limit number of images")
    ap.add_argument("--timeout", type=float, default=3.0)
    a = ap.parse_args()

    d = np.load(RESULTS_DIR / "test_inputs.npz")
    X, y, paths, flog = d["x"], d["y"], d["paths"], d["float_logits"]
    if a.n:
        idx = np.linspace(0, len(y) - 1, a.n).astype(int)  # spread over classes
        X, y, paths, flog = X[idx], y[idx], paths[idx], flog[idx]

    ser = serial.Serial(a.port, a.baud, timeout=0.2, write_timeout=5,
                        xonxoff=False, rtscts=False, dsrdtr=False)
    time.sleep(0.3)
    ser.reset_input_buffer()

    preds, logits, rows, errors = [], [], [], 0
    fpga_s, write_s, rtt_s = [], [], []
    t_start = time.perf_counter()
    for i in range(len(y)):
        frame = build_frame(NET, X[i])
        ser.reset_input_buffer()
        t0 = time.perf_counter()
        try:
            ser.write(frame)
            ser.flush()
            t1 = time.perf_counter()
            deadline = t0 + a.timeout
            wait_start(ser, deadline)
            body = read_exact(ser, REPLY_LEN - 1, deadline)
            t2 = time.perf_counter()
            r = parse_reply(body, len(frame), a.baud)
            if r["status"] != 0:
                raise RuntimeError("error reply from the board (pred = 0xFF)")
        except Exception as e:  # timeout or bad frame
            errors += 1
            print(f"image {i}: {e}")
            preds.append(-1)
            logits.append([0.0] * NC)
            rows.append(None)
            continue
        preds.append(r["pred"])
        logits.append(list(r["logits"]))
        fpga_s.append(r["infer_ns"] * 1e-9)
        write_s.append(t1 - t0)
        rtt_s.append(t2 - t0)
        rows.append({"infer_ns": r["infer_ns"], "rx_us": r["rx_us"], "rtt_s": t2 - t0})
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(y)}  acc so far {np.mean(np.array(preds) == y[: i + 1]):.4f}")
    wall = time.perf_counter() - t_start
    ser.close()

    preds = np.array(preds)
    logits = np.array(logits)
    ok = preds >= 0

    tx_bytes, rx_bytes = len(build_frame(NET, X[0])), REPLY_LEN
    wire = lambda n: n * 10 / a.baud * 1e6  # noqa: E731
    fpga_mean = float(np.mean(fpga_s)) if fpga_s else float("nan")
    rtt_mean = float(np.mean(rtt_s)) if rtt_s else float("nan")
    err = np.abs(logits[ok] - flog[ok])

    res = {
        "conditions": {
            "port": a.port,
            "baud": a.baud,
            "frame": "8N1",
            "n_images": len(y),
            "request_bytes": tx_bytes,
            "reply_bytes": rx_bytes,
            "pl_clock_mhz": 100.0,
            "timer": "A9 global timer (CPU/2 = 325 MHz), read directly, 1 us resolution in the reply",
            "protocol": "stop-and-wait: one frame in flight",
            "input": "60x60 uint8 grayscale image (raw pixels); Q8.8 conversion on the PS",
            "class_order": "circle, elipse, fig_eight, rectangle, square, triangle (as the SNN)",
        },
        "comm_errors": errors,
        "fpga": per_class_report(to_report(y[ok]), to_report(preds[ok])),
        "fpga_vs_float_model": {
            "prediction_agreement": float((preds[ok] == flog[ok].argmax(1)).mean()),
            "logits_max_abs_error": float(err.max()) if err.size else None,
            "logits_mean_abs_error": float(err.mean()) if err.size else None,
            "note": "the CNN uses fixed point (ap_fixed), so logits are close to the "
                    "float model but not bit-identical",
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

    # same per-image format as cnn/pc/shape_client.py (for compare_networks.py)
    with open(RESULTS_DIR / "results_cnn_fpga.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "pred", "status", "infer_ns", "prep_ns", "rx_us",
                    "rtt_ms"] + [f"logit_{c}" for c in CLASSES])
        for i, r in enumerate(rows):
            good = r is not None
            w.writerow([Path(str(paths[i])).name, int(y[i]), int(preds[i]), 0 if good else 4,
                        r["infer_ns"] if good else 0, 0,
                        r["rx_us"] if good else 0, f"{r['rtt_s'] * 1e3:.3f}" if good else "0"]
                       + [f"{v:.4f}" for v in logits[i]])
    (RESULTS_DIR / "summary_fpga.json").write_text(json.dumps(
        {"wall_time_s": wall, "baud": a.baud, "n_images": len(y),
         "data": "2D_Geometric_Shapes_Dataset_test", "split": "all",
         "request_bytes": tx_bytes, "reply_bytes": rx_bytes,
         "uart_rx_source": "theoretical wire time (not in the board reply)"}, indent=2))

    print(json.dumps({k: res[k] for k in ("comm_errors", "fpga_vs_float_model", "latency",
                                          "throughput")}, indent=2))
    print("FPGA accuracy:", res["fpga"]["accuracy"])
    print(f"Saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
