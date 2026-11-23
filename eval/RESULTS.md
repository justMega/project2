# SNN shape classifier: CPU vs FPGA evaluation

Measured 2026-11-17. Raw data: `results/sw_results.json`, `results/fpga_results.json`.

## Conditions

| Item | Value |
|---|---|
| Network | SNN MLP 384 → 64 → 32 → 6, LIF neurons, T = 1 time step; class score = final LIF3 membrane |
| Input | 80×60 grayscale → Sobel ON/OFF edge spikes (90th percentile) → 12×16 grid densities = 384 features |
| Test set | `2D_Geometric_Shapes_Dataset_test`, 12 000 images (2 000 per class); 0 of them are pixel-identical to any of the 60 000 training images (MD5 check) |
| Reference (float) | PyTorch 2.12 CPU + snnTorch 0.9.4, float32, no early exit |
| Reference (int) | NumPy model of `snn_mlp.cpp` (int16 weights/features, scale 128). Matches HLS C-sim logits 60/60 |
| Host | Intel Core i7-8750H (6C/12T), Windows 11, Python 3.12.10 |
| FPGA board | Zynq-7020 (xc7z020clg400-1), TUL PYNQ-Z2 class board (FTDI 0403:6010) |
| PL design | Vivado/Vitis 2026.1; `snn_mlp` HLS IP + AXI BRAM ctrl + AXI GPIO + SmartConnect; **FCLK0 = 100 MHz** (HLS was synthesized for 8 ns on a -2 part, but the design is implemented at 10 ns on a -1 part) |
| PS | Cortex-A9 @ 650 MHz, bare-metal `snn_app` (stop-and-wait: one frame in flight) |
| FPGA timer | A9 global timer (325 MHz), ap_start → ap_done, 1 µs resolution |
| UART | PS UART0, 4 000 000 baud 8N1, FTDI USB-UART, Windows FTDI latency timer = 16 ms (default) |
| Protocol | request 770 B (`AA 55` + 384 × int16), reply 31 B (`A5`, pred, time_us, 6 × int32 logits, `5A`) |

## 1. Recognition accuracy (12 000 test images)

| Implementation | Overall |
|---|---|
| CPU float32 (PyTorch/snnTorch) | **85.23 %** |
| CPU int16 bit-exact model | **85.37 %** |
| FPGA (measured over UART) | **85.37 %** (0 comm errors) |

FPGA vs int16 model: **100 % identical predictions and 100 % bit-identical logits** on all 12 000 images.
Float vs int16: 97.6 % identical predictions. Quantization costs nothing here: +0.13 pp.
The training log's final validation accuracy was 87.0 % (same-distribution held-out split).

### Per class (recall / precision / F1, %)

| Class | Float32 recall | Float32 prec. | Float32 F1 | FPGA recall | FPGA prec. | FPGA F1 |
|---|---|---|---|---|---|---|
| circle | 88.95 | 84.59 | 86.7 | 90.70 | 84.10 | 87.3 |
| elipse | 87.10 | 85.35 | 86.2 | 87.70 | 85.02 | 86.3 |
| fig_eight | 93.55 | 95.17 | 94.4 | 93.50 | 94.88 | 94.2 |
| rectangle | 79.25 | 87.28 | 83.1 | 79.15 | 87.41 | 83.1 |
| square | 75.00 | 75.23 | 75.1 | 74.65 | 76.10 | 75.4 |
| triangle | 87.55 | 84.18 | 85.8 | 86.50 | 84.97 | 85.7 |

FPGA confusion matrix (rows = true, cols = predicted; order circle, elipse, fig_eight, rectangle, square, triangle):

```
circle     1814   42    0    5  128   11
elipse       28 1754   68   12   15  123
fig_eight     0  116 1870    1    0   13
rectangle    30   18    3 1583  277   89
square      268   23    0  146 1493   70
triangle     17  110   30   64   49 1730
```

Main errors: square ↔ circle (268 / 128), rectangle → square (277), square → rectangle (146), elipse → triangle (123), fig_eight → elipse (116).

## 2. Latency

| Stage | Mean | Median | p95 | Notes |
|---|---|---|---|---|
| Host preprocessing (imread + Sobel + pooling) | 1239 µs | 1238 µs | 1495 µs | Python/OpenCV, needed by every implementation |
| **FPGA compute** (ap_start → ap_done) | **157 µs** | 157 µs | 157 µs | std 0 µs over 12 000 runs. HLS: 15 838 cycles × 10 ns = 158.4 µs |
| CPU float, batch 1, 1 thread | 887 µs | 849 µs | 1112 µs | dominated by snnTorch/PyTorch Python overhead |
| CPU float, batch 1, 12 threads | 933 µs | 897 µs | 1131 µs | |
| CPU int16 NumPy model, batch 1 | 100 µs | 90 µs | 144 µs | |

HLS breakdown (cycles): read features from BRAM via m_axi 396, FC1+LIF 12 801 (81 %), FC2+LIF 2 241, FC3 + init ~400.

## 3. UART transfer time (reported separately)

| Quantity | Value |
|---|---|
| Wire time, request (770 B × 10 bit / 4 Mbaud) | 1925 µs |
| Wire time, reply (31 B) | 77.5 µs |
| Wire time, total | **2002.5 µs** |
| Host `write()+flush()` call | 305 µs mean (returns when the driver buffers the data, not when it is on the wire) |
| Host round trip (write start → last reply byte) | **15 916 µs** mean, 15 934 median, 16 186 p95 |
| Round trip minus FPGA compute | 15 759 µs |
| … of which beyond wire time | ≈ 13 757 µs |

The extra ~13.8 ms is the Windows FTDI driver's **16 ms latency timer**. The 31-byte reply is shorter than a USB packet, so the driver holds it until the timer expires. This is a host USB-driver artifact, not UART or FPGA time. Setting the latency timer to 1 ms (Device Manager → COM7 → Port Settings → Advanced) should bring the round trip close to ~2.2–3.2 ms. That figure is a projection and was not measured. PS-side BRAM writes (384 × `Xil_Out16`) are included in the round trip but were not timed separately.

## 4. Throughput

| Mode | Inferences / s |
|---|---|
| FPGA compute only (not pipelined, II = latency) | **6 369** |
| FPGA end-to-end over UART, as measured (16 ms FTDI timer) | **62.5** |
| UART wire bound at 4 Mbaud (770 + 31 B) | 499 |
| CPU float, batch 1024, 1 thread | 414 713 |
| CPU float, batch 1024, 12 threads | 297 347 |
| CPU int16 NumPy, batch 1024 | 28 323 |

The end-to-end system is I/O bound: the UART wire alone allows at most 499 inf/s, about 13× below what the accelerator can compute.

## 5. FPGA resource utilization (xc7z020)

| Resource | Full design, placed | SNN IP only (OOC synth) | HLS estimate | Available |
|---|---|---|---|---|
| LUT | 8 813 (16.57 %) | 4 430 (8.33 %) | 9 180 | 53 200 |
| – LUT as memory | 854 (4.91 %) | 360 | – | 17 400 |
| FF | 9 200 (8.65 %) | 3 815 (3.59 %) | 4 360 | 106 400 |
| BRAM tiles (36 Kb) | 13 (9.29 %) | 11 (7.86 %) | 23 × BRAM18 | 140 |
| DSP48E1 | 8 (3.64 %) | 8 (3.64 %) | 8 | 220 |

Timing at 100 MHz: WNS +1.371 ns, WHS +0.025 ns; all constraints met.

## 6. Power / energy

**No measured power is available.** The board has no on-board current sensing that this setup reads, and no external meter was used. The figures below are Vivado *vectorless estimates* (confidence: Low, default toggle rates). They are not measurements:

| Item | Estimated power | Energy per inference (× 157 µs) |
|---|---|---|
| `snn_mlp_0` IP (dynamic) | 0.060 W | ≈ 9.4 µJ |
| Total PL dynamic (excl. PS7) | 0.111 W | ≈ 17 µJ |
| Total on-chip incl. PS7 (1.256 W) and static (0.141 W) | 1.509 W | ≈ 237 µJ |

To measure power, put a USB power meter on the board's 5 V input and compare idle against continuous inference. The AXI GPIO timing pin (high during ap_start → ap_done) can gate a scope or power-analyzer capture.

## Firmware issue found

`helloworld.c` uses `XTime_GetTime()`, but under the SDT/xiltimer BSP the A9 global timer only starts on the first `sleep()`/`usleep()` call. The app never sleeps, so it always reported `time_us = 0`. For these measurements the timer was enabled over JTAG (`mwr -force 0xF8F00208 1`). Permanent fix: add `usleep(1);` (or `Xil_Out32(0xF8F00208, 1)`) at the start of `main()`.

## Reproduce

```
cd py_code
.venv\Scripts\python.exe eval\benchmark_sw.py      # CPU float + int16 model, caches features
# program board (Vitis launch or xsdb), then: xsdb> mwr -force 0xF8F00208 1
.venv\Scripts\python.exe eval\benchmark_fpga.py --port COM7
```
