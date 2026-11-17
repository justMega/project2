#!/usr/bin/env python3
"""
Assignment 2, workflow step 5: compare the CNN and the SNN measured on the board.

Merges, image by image, the per-image results written by
`shape_client.py eval` for each network (results_<net>_fpga.csv), and adds the
post-implementation Vivado reports of each hardware build. Works whether both
networks were in one bitstream (one results folder) or in two separate builds
(one results folder each), and also with the CNN alone.

Produces in --out:
  comparison.md        table for the report (accuracy, latency, throughput,
                       resources, power/energy, UART time separately, conditions)
  comparison.json      the same numbers, machine-readable
  per_image.csv        file, true label, CNN prediction, SNN prediction
  examples_grid.png    input images with true label + each network's prediction
  per_class_accuracy.png
  confusion_<net>.png

Example (two separate builds, same test images):
  python3 compare_networks.py \\
      --data ../../2D_Geometric_Shapes_Dataset_test \\
      --cnn-results results_newtest --cnn-vivado ../vivado/reports \\
      --snn-results ../../snn/pc/results_newtest --snn-vivado ../../snn/vivado/reports \\
      --out comparison_newtest
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from shapes_common import CLASSES, N_CLASSES, IMG_H, IMG_W, load_u8  # noqa: E402

NETS = ("cnn", "snn")
FRAME_BYTES = 3 + 2 + IMG_H * IMG_W + 1      # request frame, see vitis/PROTOCOL.md


# ------------------------------------------------------------------ inputs
def load_results(folder: Path, net: str):
    """Per-image rows of results_<net>_fpga.csv (+ summary_fpga.json if present)."""
    f = Path(folder) / f"results_{net}_fpga.csv"
    if not f.exists():
        return None
    rows = []
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(file=r["file"], true=int(r["true"]), pred=int(r["pred"]),
                             status=int(r["status"]), infer_ns=int(r["infer_ns"]),
                             prep_ns=int(r["prep_ns"]), rx_us=int(r["rx_us"]),
                             rtt_ms=float(r["rtt_ms"])))
    summary = {}
    s = Path(folder) / "summary_fpga.json"
    if s.exists():
        summary = json.loads(s.read_text())
    return dict(rows=rows, summary=summary, path=str(f))


def parse_utilization(rpt: Path):
    """Whole-design usage from Vivado report_utilization."""
    out = {}
    if not rpt.exists():
        return out
    keys = {"Slice LUTs": "lut", "Slice Registers": "ff", "Block RAM Tile": "bram", "DSPs": "dsp"}
    for line in rpt.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0] in keys and keys[cells[0]] not in out:
            try:
                used = float(cells[1])
                avail = float(cells[-2])
            except ValueError:
                continue
            out[keys[cells[0]]] = dict(used=used, available=avail, pct=100.0 * used / avail)
    return out


def parse_hier_utilization(rpt: Path, ip_hint: str):
    """Usage of the accelerator IP alone (report_utilization -hierarchical)."""
    if not rpt.exists():
        return {}
    header, best = None, None
    for line in rpt.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "Instance":
            header = cells
            continue
        if header and len(cells) == len(header):
            name = cells[0]
            if re.fullmatch(ip_hint, name) or (ip_hint in name and name.endswith("_0")):
                best = dict(zip(header, cells))
                break
    if best is None:
        return {}
    num = lambda k: float(best.get(k, 0) or 0)
    return dict(instance=best["Instance"], lut=num("Total LUTs"), ff=num("FFs"),
                bram=num("RAMB36") + num("RAMB18") / 2.0, dsp=num("DSP Blocks"))


def parse_power(rpt: Path, ip_hint: str):
    """Total on-chip power, PS7 power and the IP's dynamic power (report_power)."""
    if not rpt.exists():
        return {}
    text = rpt.read_text()
    out = {}
    m = re.search(r"Total On-Chip Power \(W\)\s*\|\s*([\d.]+)", text)
    if m:
        out["total_w"] = float(m.group(1))
    m = re.search(r"Device Static \(W\)\s*\|\s*([\d.]+)", text)
    if m:
        out["static_w"] = float(m.group(1))
    hier = text[text.rfind("By Hierarchy"):] if "By Hierarchy" in text else ""
    for line in hier.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            try:
                val = float(cells[1])
            except ValueError:
                continue
            if cells[0] == "ps7":
                out["ps7_w"] = val
            elif "ip_w" not in out and (re.fullmatch(ip_hint, cells[0]) or
                                        (ip_hint in cells[0] and cells[0].endswith("_0"))):
                out["ip_w"] = val
                out["ip_instance"] = cells[0]
    return out


# ------------------------------------------------------------------ metrics
def metrics(res, util, hier, power, clock_mhz):
    rows = res["rows"]
    ok = [r for r in rows if r["status"] == 0 and r["pred"] >= 0]
    y = np.array([r["true"] for r in ok])
    p = np.array([r["pred"] for r in ok])
    inf_us = np.array([r["infer_ns"] for r in ok], float) / 1e3
    prep_us = np.array([r["prep_ns"] for r in ok], float) / 1e3
    rx_ms = np.array([r["rx_us"] for r in ok], float) / 1e3
    rtt = np.array([r["rtt_ms"] for r in ok], float)
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    per_class = {CLASSES[c]: (float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None)
                 for c in range(N_CLASSES)}
    m = dict(
        n_images=len(rows), n_valid=len(ok), failed_replies=len(rows) - len(ok),
        accuracy=float((y == p).mean()) if len(ok) else None,
        misclassified=int((y != p).sum()),
        per_class_accuracy=per_class, confusion=cm.tolist(),
        latency_us=dict(mean=float(inf_us.mean()), std=float(inf_us.std()),
                        min=float(inf_us.min()), max=float(inf_us.max())),
        latency_cycles=float(inf_us.mean() * clock_mhz),
        accel_throughput_img_s=float(1e6 / inf_us.mean()),
        cpu_prep_us=float(prep_us.mean()),
        uart_rx_ms=float(rx_ms.mean()),
        uart_theoretical_ms=None,
        pc_round_trip_ms=float(rtt.mean()),
        end_to_end_throughput_img_s=float(1e3 / rtt.mean()),
        uart_share_of_round_trip=float(rx_ms.mean() / rtt.mean()),
        resources_design=util, resources_ip=hier, power=power)
    if power.get("ip_w"):
        m["energy_ip_mj"] = power["ip_w"] * inf_us.mean() / 1e3        # W x ms = mJ
    if power.get("total_w"):
        m["energy_chip_mj"] = power["total_w"] * inf_us.mean() / 1e3
    return m


# ------------------------------------------------------------------ outputs
def fmt(v, spec="{:.2f}", none="–"):
    return none if v is None else spec.format(v)


def write_markdown(nets, M, conditions, agreement, out: Path):
    L = []
    hdr = "| | " + " | ".join(n.upper() for n in nets) + " |"
    sep = "|---|" + "---|" * len(nets)
    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(M[n]) for n in nets) + " |")

    L += ["## Recognition accuracy", "", hdr, sep]
    row("Test images", lambda m: f"{m['n_valid']:,}")
    row("Overall accuracy", lambda m: f"{100*m['accuracy']:.2f} % ({m['misclassified']} error{'s' if m['misclassified'] != 1 else ''})")
    for c in CLASSES:
        row(f"– {c}", lambda m, c=c: fmt(m["per_class_accuracy"][c] and 100*m["per_class_accuracy"][c], "{:.2f} %"))
    if agreement is not None:
        L += ["", f"CNN and SNN predict the same class on {100*agreement:.2f} % of the images."]

    L += ["", "## Inference latency and throughput (on the board)", "", hdr, sep]
    row("Accelerator latency (ap_start → ap_done)",
        lambda m: f"{m['latency_us']['mean']/1e3:.3f} ms (± {m['latency_us']['std']:.2f} µs)")
    row("Latency in clock cycles", lambda m: f"{m['latency_cycles']:,.0f}")
    row("Accelerator throughput", lambda m: f"{m['accel_throughput_img_s']:.0f} img/s")
    row("CPU preprocessing + cache flush", lambda m: f"{m['cpu_prep_us']:.0f} µs")

    L += ["", "## Communication overhead (reported separately)", "", hdr, sep]
    row("UART transfer of one image (board RX)", lambda m: f"{m['uart_rx_ms']:.2f} ms")
    row("PC round trip (send → reply)", lambda m: f"{m['pc_round_trip_ms']:.2f} ms")
    row("UART share of the round trip", lambda m: f"{100*m['uart_share_of_round_trip']:.0f} %")
    row("End-to-end throughput", lambda m: f"{m['end_to_end_throughput_img_s']:.1f} img/s")

    L += ["", "## FPGA resources (post-implementation)", "", hdr, sep]
    for k, name in [("lut", "LUT"), ("ff", "FF"), ("bram", "BRAM tiles (36 Kb)"), ("dsp", "DSP")]:
        row(f"{name}, whole design", lambda m, k=k: (
            f"{m['resources_design'][k]['used']:,.1f} / {m['resources_design'][k]['available']:,.0f} "
            f"({m['resources_design'][k]['pct']:.1f} %)".replace(".0 /", " /")
            if k in m["resources_design"] else "–"))
        row(f"{name}, accelerator IP only", lambda m, k=k: (
            f"{m['resources_ip'][k]:,.1f}".replace(".0", "") if k in m["resources_ip"] else "–"))

    L += ["", "## Power and energy (Vivado post-implementation estimate)", "", hdr, sep]
    row("Total on-chip power", lambda m: fmt(m["power"].get("total_w"), "{:.3f} W"))
    row("– of which PS7 (ARM)", lambda m: fmt(m["power"].get("ps7_w"), "{:.3f} W"))
    row("– of which accelerator IP (dynamic)", lambda m: fmt(m["power"].get("ip_w"), "{:.3f} W"))
    row("Energy per inference, IP only", lambda m: fmt(m.get("energy_ip_mj"), "{:.3f} mJ"))
    row("Energy per inference, whole chip", lambda m: fmt(m.get("energy_chip_mj"), "{:.2f} mJ"))

    L += ["", "## Measurement conditions", ""]
    for k, v in conditions.items():
        L.append(f"- {k}: {v}")
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def plot_per_class(nets, M, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(N_CLASSES)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    markers = {"cnn": "o", "snn": "s"}
    for i, n in enumerate(nets):
        v = [100 * (M[n]["per_class_accuracy"][c] or 0) for c in CLASSES]
        ax.plot(x + (i - (len(nets) - 1) / 2) * 0.12, v, markers[n], markersize=8, label=n.upper())
    lo = min(100 * (M[n]["per_class_accuracy"][c] or 0) for n in nets for c in CLASSES)
    ax.set_ylim(max(0, lo - 1), 100.3)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy on the FPGA")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "per_class_accuracy.png", dpi=150)
    plt.close(fig)


def plot_confusions(nets, M, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    for n in nets:
        cm = np.array(M[n]["confusion"])
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(values_format="d", cmap="Blues",
                                                              ax=ax, colorbar=False)
        ax.set_title(f"{n.upper()} on FPGA ({100*M[n]['accuracy']:.2f} %)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out / f"confusion_{n}.png", dpi=150)
        plt.close(fig)


def plot_examples(nets, merged, data: Path, per_class: int, seed: int, out: Path):
    """Input image + true label + prediction of each network (green = right, red = wrong).
    Shows every image that at least one network got wrong (up to one row), then
    `per_class` random images per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    wrong = [k for k, r in merged.items() if any(r.get(n) is not None and r[n] != r["true"] for n in nets)]
    picks = []
    for c in range(N_CLASSES):
        files = sorted(k for k, r in merged.items() if r["true"] == c and k not in wrong)
        if files:
            picks += list(rng.choice(files, min(per_class, len(files)), replace=False))
    picks = sorted(wrong)[:N_CLASSES] + picks
    cols = N_CLASSES
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.3 * cols, 3.3 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, f in enumerate(picks):
        r = merged[f]
        ax = axes[k // cols][k % cols]
        img_path = Path(data) / CLASSES[r["true"]] / f
        if img_path.exists():
            ax.imshow(load_u8(img_path), cmap="gray", vmin=0, vmax=255)
        lines = [f"true: {CLASSES[r['true']]}"]
        colors = []
        for n in nets:
            p = r.get(n)
            good = p == r["true"]
            lines.append(f"{n.upper()}: {CLASSES[p] if p is not None and p >= 0 else 'n/a'}")
            colors.append("green" if good else "red")
        ax.set_title(f"{f}\n" + "\n".join(lines), fontsize=8,
                     color="green" if all(c == "green" for c in colors) else "red")
    fig.suptitle("Input image (60×60 as sent over UART), true label and prediction of each network",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2)
    plt.savefig(out / "examples_grid.png", dpi=150)
    plt.close(fig)
    return len(wrong)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder the results were measured on")
    ap.add_argument("--cnn-results", type=Path, help="folder with results_cnn_fpga.csv")
    ap.add_argument("--snn-results", type=Path, help="folder with results_snn_fpga.csv "
                                                     "(same folder as the CNN if one bitstream)")
    ap.add_argument("--cnn-vivado", type=Path, help="Vivado reports/ folder of the CNN build")
    ap.add_argument("--snn-vivado", type=Path, help="Vivado reports/ folder of the SNN build")
    ap.add_argument("--cnn-ip", default="ds_cnn_0", help="CNN IP instance name in the block design")
    ap.add_argument("--snn-ip", default="snn", help="SNN IP instance name (or part of it)")
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--per-class", type=int, default=2, help="example images per class in the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("comparison"))
    args = ap.parse_args()

    folders = {"cnn": args.cnn_results, "snn": args.snn_results or args.cnn_results}
    vivado = {"cnn": args.cnn_vivado, "snn": args.snn_vivado}
    ip = {"cnn": args.cnn_ip, "snn": args.snn_ip}
    R, M = {}, {}
    for n in NETS:
        if folders[n] is None:
            continue
        res = load_results(folders[n], n)
        if res is None:
            continue
        R[n] = res
        rep = Path(vivado[n]) if vivado[n] else Path("/nonexistent")
        M[n] = metrics(res, parse_utilization(rep / "utilization.rpt"),
                       parse_hier_utilization(rep / "utilization_hier.rpt", ip[n]),
                       parse_power(rep / "power.rpt", ip[n]), args.clock_mhz)
        print(f"[{n.upper()}] {len(res['rows'])} images from {res['path']}")
    nets = [n for n in NETS if n in R]
    if not nets:
        sys.exit("No results_<net>_fpga.csv found: run shape_client.py eval first.")
    if len(nets) == 1:
        print(f"Only {nets[0].upper()} results found: the comparison shows this network alone.")

    # image-by-image merge (same test images for both networks)
    merged = {}
    for n in nets:
        for r in R[n]["rows"]:
            m = merged.setdefault(r["file"], {"true": r["true"]})
            m[n] = r["pred"] if r["status"] == 0 else None
    agreement = None
    if len(nets) == 2:
        common = [r for r in merged.values() if r.get("cnn") is not None and r.get("snn") is not None]
        if len(common) != len(merged):
            print(f"WARNING: only {len(common)} of {len(merged)} images were evaluated by both networks.")
        agreement = float(np.mean([r["cnn"] == r["snn"] for r in common])) if common else None

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true"] + [f"{n}_pred" for n in nets])
        for f in sorted(merged):
            r = merged[f]
            w.writerow([f, CLASSES[r["true"]]] +
                       [CLASSES[r[n]] if r.get(n) is not None and r[n] >= 0 else "" for n in nets])

    s0 = R[nets[0]]["summary"]
    conditions = {
        "Board": "PYNQ-Z2 (Zynq-7020, xc7z020clg400-1), bare-metal application on the ARM Cortex-A9",
        "Accelerator clock": f"{args.clock_mhz:.0f} MHz (FCLK_CLK0)",
        "Test images": f"{len(merged):,} images from {args.data} "
                       f"({s0.get('split', 'test')} split), identical for every network",
        "Image format": f"{IMG_W}×{IMG_H} grayscale, 8 bit, raw pixels sent by the PC",
        "UART": f"{args.baud:,} baud, 8N1, {FRAME_BYTES:,}-byte request frame, 29-byte reply "
                f"(theoretical {FRAME_BYTES * 10 / args.baud * 1e3:.2f} ms per image)",
        "Timing": "accelerator time = ap_start → ap_done, ARM global timer; UART time measured "
                  "on the board; round trip measured on the PC",
        "Resources and power": "Vivado post-implementation reports (power: vectorless estimate)",
        "Generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    for n in nets:
        M[n]["uart_theoretical_ms"] = FRAME_BYTES * 10 / args.baud * 1e3
    md = write_markdown(nets, M, conditions, agreement, args.out)
    (args.out / "comparison.json").write_text(json.dumps(
        dict(nets=nets, metrics=M, conditions=conditions, agreement=agreement), indent=2))
    plot_per_class(nets, M, args.out)
    plot_confusions(nets, M, args.out)
    n_wrong = plot_examples(nets, merged, args.data, args.per_class, args.seed, args.out)

    print("\n" + md)
    print(f"\n{n_wrong} image(s) misclassified by at least one network (shown first in the grid).")
    print(f"Saved in {args.out}/: comparison.md, comparison.json, per_image.csv, "
          f"examples_grid.png, per_class_accuracy.png, confusion_*.png")


if __name__ == "__main__":
    main()
