# CNN shape classifier: CPU vs FPGA evaluation

Measured 2026-11-17. Raw data: `results_cnn/sw_results.json`, `results_cnn/fpga_results.json`, per image: `results_cnn/results_cnn_fpga.csv`.
Scripts: `cnn_benchmark_sw.py`, `cnn_benchmark_fpga.py` (same measurements and JSON keys as the SNN's `benchmark_sw.py` / `benchmark_fpga.py`).
Class order in every table: circle, elipse, fig_eight, rectangle, square, triangle (as the SNN).

## Conditions

| Item | Value |
|---|---|
| Network | CNN: Conv3×3(16)+BN+ReLU+MaxPool → 3 depthwise-separable blocks (DW3×3 + PW1×1, 32 channels, BN+ReLU, MaxPool after blocks 1 and 2) → global average pooling → Dense 32 → 6. 4 390 parameters, 3 814 after BN folding |
| Input | 224×224 image → 60×60 grayscale (`cv2.INTER_AREA`) on the PC, sent as raw uint8 pixels. The PS converts them to Q8.8 (z = 4x/255 − 2) |
| Training data | `2D_Geometric_Shapes_Dataset`, images 1–7000 of each class (train), 7001–8500 (validation), 8501–10000 (internal test) |
| Test set | `2D_Geometric_Shapes_Dataset_test`, 12 000 images (2 000 per class); 0 of them are pixel-identical to any of the 60 000 training images (MD5 check) |
| Reference (float) | Keras (TensorFlow 2.22.0-rc0, CPU), float32, `model(x, training=False)` |
| Fixed point (HLS) | features `ap_fixed<16,8>`, weights `ap_fixed<18,7>`, accumulators `ap_fixed<32,12>`. C-simulation agrees with Keras on 100 % of the testbench images |
| Host | Linux x86_64, 8 logical CPUs, Python 3.14.4 |
| FPGA board | TUL PYNQ-Z2, Zynq-7020 (xc7z020clg400-1) |
| PL design | Vivado/Vitis 2026.1; `ds_cnn` HLS IP (m_axi to DDR through HP0, s_axilite control) + 2 SmartConnects + proc_sys_reset; **FCLK0 = 100 MHz**, timing met (WNS +0.318 ns, WHS +0.027 ns) |
| PS | Cortex-A9 @ 650 MHz, bare-metal `shape_cnn_app` (stop-and-wait: one frame in flight) |
| FPGA timer | A9 global timer (325 MHz), read directly by the firmware, ap_start → ap_done, 1 µs resolution in the reply |
| UART | PS UART0, 4 000 000 baud 8N1, FTDI USB-UART, Linux FTDI latency timer = 16 ms (default) |
| Protocol | request 3 606 B (`A6 6A`, net, length u16, 3 600 pixels, checksum), reply 31 B (`A5`, pred, time_us u32, 6 × int32 logits in Q8.8, `5A`), same reply format as the SNN |

## 1. Recognition accuracy (12 000 test images)

| Implementation | Overall |
|---|---|
| CPU float32 (Keras) | **99.93 %** |
| FPGA (measured over UART) | **99.93 %** (0 comm errors, 8 misclassified images) |

FPGA vs float model: **100 % identical predictions**. Logits differ by 0.034 on average and 0.25 at most, because of fixed-point rounding. Quantization costs no accuracy.

### Per class (recall / precision / F1, %)

| Class | Recall | Precision | F1 |
|---|---|---|---|
| circle | 100.00 | 100.00 | 100.0 |
| elipse | 99.90 | 99.85 | 99.9 |
| fig_eight | 99.95 | 99.95 | 99.9 |
| rectangle | 99.85 | 100.00 | 99.9 |
| square | 100.00 | 99.85 | 99.9 |
| triangle | 99.90 | 99.95 | 99.9 |

The CPU float model and the FPGA give the same table: their predictions are identical.

FPGA confusion matrix (rows = true, cols = predicted; order circle, elipse, fig_eight, rectangle, square, triangle):

```
circle     2000    0    0    0    0    0
elipse        0 1998    1    0    0    1
fig_eight     0    1 1999    0    0    0
rectangle     0    0    0 1997    3    0
square        0    0    0    0 2000    0
triangle      0    2    0    0    0 1998
```

The 8 errors are rectangle → square (3), triangle → elipse (2), elipse → fig_eight, elipse → triangle and fig_eight → elipse (1 each).

## 2. Latency

| Stage | Mean | Median | p95 | Notes |
|---|---|---|---|---|
| Host preprocessing (imread + resize to 60×60) | 293 µs | 282 µs | 327 µs | Python/OpenCV |
| PS preprocessing (u8 → Q8.8 + cache flush) | 164 µs | – | – | constant; measured with the earlier reply format, which carried this field |
| **FPGA compute** (ap_start → ap_done) | **1 402 µs** | 1 402 µs | 1 402 µs | std 0.2 µs over 12 000 runs (≈ 140 200 cycles at 100 MHz). HLS estimate: 148 353 cycles |
| CPU float, batch 1, 1 thread | 9 226 µs | 8 936 µs | 11 062 µs | dominated by TensorFlow eager-call overhead |
| CPU float, batch 1, 8 threads | 12 749 µs | 12 059 µs | 17 053 µs | thread overhead makes batch 1 slower |

## 3. UART transfer time (reported separately)

| Quantity | Value |
|---|---|
| Wire time, request (3 606 B × 10 bit / 4 Mbaud) | 9 015 µs |
| Board-side reception time, measured by the PS | **9 012 µs** (std 0), earlier reply format |
| Wire time, reply (31 B) | 77.5 µs |
| Wire time, total | **9 092.5 µs** |
| Host `write()+flush()` call | 10 287 µs mean (blocks until the driver has sent most of the 3.6 kB) |
| Host round trip (write start → last reply byte) | **15 854 µs** mean, 15 847 median, 16 521 p95 |
| Round trip minus FPGA compute | 14 452 µs |
| … of which beyond wire time and PS work | ≈ 5 200 µs |

The board receives the frame in 9 012 µs for a theoretical 9 015 µs: the link runs at the full 4 Mbaud without losses. The remaining ~5 ms come from the FTDI driver's **16 ms latency timer**. The 31-byte reply is shorter than a USB packet, so it waits in the driver until the timer expires. This is a host USB-driver artifact, not UART or FPGA time. With `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer`, an earlier test measured about 11.5 ms per round trip.

## 4. Throughput

| Mode | Inferences / s |
|---|---|
| FPGA compute only (not pipelined, II = latency) | **713** |
| FPGA end-to-end over UART, as measured (16 ms FTDI timer) | **62.5** |
| UART wire bound at 4 Mbaud (3 606 + 31 B) | 110 |
| CPU float, batch 1024, 1 thread | 2 920 |
| CPU float, batch 1024, 8 threads | 4 008 |

The end-to-end system is I/O bound: the UART wire alone allows at most 110 inf/s, 6.5× below what the accelerator can compute.

## 5. FPGA resource utilization (xc7z020, post-implementation)

| Resource | Full design | CNN IP only (`ds_cnn_0`) | Available |
|---|---|---|---|
| LUT | 11 950 (22.46 %) | 11 110 | 53 200 |
| – LUT as memory | 278 (1.60 %) | – | 17 400 |
| FF | 14 272 (13.41 %) | 13 446 | 106 400 |
| BRAM tiles (36 Kb) | 46.5 (33.21 %) | 46.5 | 140 |
| DSP48E1 | 102 (46.36 %) | 102 | 220 |

Parallelism: 16 output channels in the first convolution, 8 in the depthwise-separable blocks. Feature maps are stored in BRAM and partitioned by channel, and the weights are stored in ROM.

## 6. Power / energy

**No measured power is available.** These figures are Vivado post-implementation *vectorless estimates* (confidence: Medium). They are not measurements.

| Item | Estimated power | Energy per inference (× 1 402 µs) |
|---|---|---|
| `ds_cnn_0` IP (dynamic) | 0.374 W | ≈ 0.52 mJ |
| Total PL dynamic (excl. PS7) | 0.381 W | ≈ 0.53 mJ |
| Total on-chip incl. PS7 (1.261 W) and static (0.150 W) | 1.793 W | ≈ 2.51 mJ |

## Comparison with the SNN (see `RESULTS.md`)

| | CNN | SNN |
|---|---|---|
| Accuracy (12 000 test images, FPGA) | **99.93 %** | 85.37 % |
| Data sent per image | 3 606 B (raw image) | 770 B (384 features computed on the PC) |
| Host preprocessing | 293 µs | 1 239 µs |
| FPGA compute latency | 1 402 µs | **157 µs** |
| FPGA compute throughput | 713 inf/s | **6 369 inf/s** |
| End-to-end over UART (16 ms FTDI timer) | 62.5 inf/s | 62.5 inf/s |
| LUT / FF / BRAM / DSP (full design) | 11 950 / 14 272 / 46.5 / 102 | 8 813 / 9 200 / 13 / **8** |
| IP dynamic power (estimate) | 0.374 W | **0.060 W** |
| Energy per inference, IP (estimate) | 0.52 mJ | **≈ 9.4 µJ** |

The host CPUs differ (CNN: Linux laptop, 8 threads; SNN: i7-8750H, Windows), so the CPU timings of the two networks are not directly comparable.

## Reproduce

```
cd assignment2
source cnn/.venv/bin/activate
python3 eval/cnn_benchmark_sw.py                         # CPU float model, caches the 60x60 images
# program the board (Vitis: shape_cnn_app), then:
python3 eval/cnn_benchmark_fpga.py --port /dev/ttyUSB1
```
