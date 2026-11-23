# SNN vs CNN on Zynq-7020: shape classification

Both models were tested on the same 12 000-image test set (`2D_Geometric_Shapes_Dataset_test`, 2 000 per class). None of these images is pixel-identical to any of the 60 000 training images (MD5 check, both runs). Raw data: `results/`, `results_cnn/`. SNN details: `RESULTS.md`.

## 1. Test conditions

| | SNN | CNN |
|---|---|---|
| Network | MLP 384 → 64 → 32 → 6, LIF neurons, T = 1 | Conv3×3(16) + 3 depthwise-separable blocks (32) + GAP + Dense(6); 3 814 params after BN folding |
| Arithmetic on FPGA | int16 weights / features (scale 128), int32 accumulate | `ap_fixed` (input Q8.8) |
| Host preprocessing | resize 80×60 → Sobel ON/OFF spikes → 12×16 grid densities (384 features) | resize to 60×60 grayscale |
| Sent over UART | 384 × int16 features: 770 B incl. 2 B header | 60×60 uint8 pixels: 3 606 B (3 600 B pixels + 6 B framing) |
| Reply | 31 B (pred, time_us, 6 × int32 logits) | 31 B |
| FPGA board / part | Zynq-7020 xc7z020clg400-1 | same board |
| PL clock | 100 MHz | 100 MHz |
| FPGA latency timer | A9 global timer (325 MHz) around ap_start → ap_done, 1 µs resolution | same |
| UART | 4 Mbaud 8N1, FTDI USB-UART, stop-and-wait | same |
| Host OS / USB driver | Windows 11, FTDI VCP driver (latency timer 16 ms) | Linux, `ftdi_sio` (default latency timer 16 ms) |
| CPU reference | Intel Core i7-8750H (12 threads), PyTorch 2.12 + snnTorch 0.9.4 | x86-64 Linux host (8 threads, CPU model not recorded), TensorFlow/Keras 2.22.0-rc0, eager |
| Reference used to check the FPGA | bit-exact NumPy integer model of `snn_mlp.cpp` | float Keras model (HLS C-sim agrees with Keras on the testbench images) |

**Comparability:** the CPU baselines and the host-side UART timings come from **different machines, operating systems and frameworks** (see §6). The FPGA compute numbers come from identical boards, clocks and timers, so they compare directly.

## 2. Recognition accuracy

| | SNN | CNN |
|---|---|---|
| Float model on CPU | 85.23 % | 99.93 % |
| **FPGA, measured** | **85.37 %** | **99.93 %** |
| FPGA vs reference | 100 % same predictions, logits **bit-exact** | 100 % same predictions, logit error mean 0.034 / max 0.25 |
| Quantization effect | +0.13 pp | 0.00 pp |
| Communication errors | 0 / 12 000 | 0 / 12 000 |

### Per class on the FPGA (recall / precision, %)

| Class | SNN recall | SNN precision | CNN recall | CNN precision |
|---|---|---|---|---|
| circle | 90.70 | 84.10 | 100.00 | 100.00 |
| elipse | 87.70 | 85.02 | 99.90 | 99.85 |
| fig_eight | 93.50 | 94.88 | 99.95 | 99.95 |
| rectangle | 79.15 | 87.41 | 99.85 | 100.00 |
| square | 74.65 | 76.10 | 100.00 | 99.85 |
| triangle | 86.50 | 84.97 | 99.90 | 99.95 |

Errors (out of 12 000): **SNN 1 756**, **CNN 8**. The SNN mostly confuses square ↔ circle (268 + 128) and rectangle ↔ square (277 + 146). The CNN's 8 errors are rectangle → square (3), triangle → elipse (2), and one each of elipse → fig_eight, elipse → triangle and fig_eight → elipse.

## 3. Inference latency

| | SNN | CNN | Ratio |
|---|---|---|---|
| **FPGA compute** (ap_start → ap_done) | **157 µs** (std 0) | **1 402 µs** (std 0.2) | CNN 8.9× slower |
| Host preprocessing | 1 239 µs (Win, Sobel + pooling) | 293 µs (Linux, resize) | different hosts |
| CPU float, batch 1, 1 thread (median) | 849 µs | 8 936 µs | different hosts/frameworks |
| CPU float, batch 1, all threads (median) | 897 µs (12 thr) | 12 059 µs (8 thr) | different hosts/frameworks |

Both accelerators have essentially no variation from image to image: the latency is fixed and repeatable.

## 4. UART transfer time (separate from compute)

| | SNN | CNN |
|---|---|---|
| Request wire time (bytes × 10 bit / 4 Mbaud) | **1 925 µs** | **9 015 µs** |
| Reply wire time | 77.5 µs | 77.5 µs |
| Total wire time | **2 002.5 µs** | **9 092.5 µs** |
| Wire time ÷ FPGA compute time | 12.8× | 6.5× |
| Host round trip, measured (mean) | 15 916 µs | 15 854 µs |
| Round trip − compute − wire (≈ USB driver delay) | ≈ 13 757 µs | ≈ 5 360 µs |

UART dominates both systems, and for the CNN it carries 4.7× more data per image. The wire times are computed from the byte count and the baud rate; the board does not report them.

**The measured round trips are set by the USB driver, not by the designs.** On both hosts the FTDI latency timer was 16 ms. A reply shorter than a USB packet is held until the timer fires, so each request–reply cycle is rounded up to about 16 ms. That is why both end-to-end rates come out at exactly 62.5 /s (= 1 / 16 ms) even though the two designs differ a lot.

## 5. Throughput

| | SNN | CNN |
|---|---|---|
| **FPGA compute only** (not pipelined, one image at a time) | **6 369 /s** | **713 /s** |
| Limit set by UART wire time | 499 /s | 110 /s |
| Projected end-to-end with 1 ms latency timer (wire + compute, stop-and-wait) — *not measured* | ≈ 460 /s | ≈ 95 /s |
| End-to-end, measured (16 ms latency timer) | 62.5 /s | 62.5 /s |
| CPU float, batch 1024, 1 thread | 414 713 /s | 2 920 /s |
| CPU float, batch 1024, all threads | 297 347 /s (12 thr) | 4 008 /s (8 thr) |

## 6. FPGA resource utilization and timing (xc7z020)

| Resource | SNN full design (placed) | SNN IP only (OOC synth) | CNN | Available |
|---|---|---|---|---|
| LUT | 8 813 (16.57 %) | 4 430 (8.33 %) | **n/a** | 53 200 |
| FF | 9 200 (8.65 %) | 3 815 (3.59 %) | **n/a** | 106 400 |
| BRAM (36 Kb tiles) | 13 (9.29 %) | 11 (7.86 %) | **n/a** | 140 |
| DSP48E1 | 8 (3.64 %) | 8 (3.64 %) | **n/a** | 220 |
| Timing @ 100 MHz | WNS +1.371 ns, met | – | **n/a** | |

The CNN's Vivado/HLS reports were not available: they're not in this checkout or in `origin/cnn`. They need to be added before the resource comparison can be made.

## 7. Power and energy

**No measured power for either design.** SNN figures are Vivado vectorless estimates (confidence: Low):

| | SNN | CNN |
|---|---|---|
| Accelerator IP, dynamic (est.) | 0.060 W → ≈ 9.4 µJ / inference | n/a |
| Whole chip incl. PS7 (est.) | 1.509 W → ≈ 237 µJ / inference | n/a |

The whole-chip power is dominated by the PS7 (1.26 W), which is the same in both designs. So the per-inference energy is roughly proportional to latency: with a similar total power, the CNN would use about 9× more energy per inference than the SNN. This is an inference from latency, not a measurement. To measure it, put a USB power meter on the board's 5 V input and compare idle against continuous inference for each bitstream.

## 8. Caveats

1. **CPU numbers are not comparable across the two columns.** They come from different hardware, different OSes, and PyTorch vs Keras eager; Keras eager's ~9 ms single-image latency is mostly framework overhead. For a fair CPU comparison, run both on one machine.
2. **"Host write time" means different things.** On Windows, `write()` returns once the data is buffered: 305 µs for the SNN, less than its 1.93 ms wire time. On Linux, `flush()` waits for transmission: 10.3 ms for the CNN, about its 9.0 ms wire time. It is not reported above for that reason.
3. **Preprocessing is split differently.** The SNN's Sobel/pooling runs on the host (1.24 ms) and is excluded from its FPGA path; the CNN does its Q8.8 conversion on the PS, inside the round trip.
4. **CNN CSV columns.** In `results_cnn/results_cnn_fpga.csv`, `prep_ns` is always 0 and `rx_us` is always the theoretical 9 015 µs; they are not measurements. The CSV (and `test_inputs.npz`) uses the class order circle, square, fig_eight, rectangle, triangle, elipse; the JSON results were remapped to the SNN order (verified: the confusion matrices match).
5. **SNN firmware timer.** The SNN app needed its A9 global timer enabled over JTAG (see `RESULTS.md`), otherwise it reports 0 µs.

## 9. Summary

| Criterion | Better | Margin |
|---|---|---|
| Accuracy (overall and every class) | **CNN** | 99.93 % vs 85.37 % (8 vs 1 756 errors) |
| FPGA compute latency / throughput | **SNN** | 157 µs vs 1 402 µs (8.9×) |
| UART data per image | **SNN** | 770 B vs 3 606 B (4.7×) |
| Estimated energy per inference | **SNN** (inferred) | ≈ 9× lower, following latency |
| Host-side work | **CNN** | sends raw pixels; SNN needs host Sobel + pooling |
| FPGA resources | open | CNN reports missing |

In the deployed system, both are limited by the UART link and the USB driver, not the accelerator. If accuracy matters, the CNN is the clear choice. Its 1.4 ms compute time is still small compared with its own 9 ms of UART transfer.
