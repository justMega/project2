## Recognition accuracy

| | CNN |
|---|---|
| Test images | 12,000 |
| Overall accuracy | 99.93 % (8 errors) |
| – circle | 100.00 % |
| – square | 100.00 % |
| – fig_eight | 99.95 % |
| – rectangle | 99.85 % |
| – triangle | 99.90 % |
| – elipse | 99.90 % |

## Inference latency and throughput (on the board)

| | CNN |
|---|---|
| Accelerator latency (ap_start → ap_done) | 1.402 ms (± 0.06 µs) |
| Latency in clock cycles | 140,230 |
| Accelerator throughput | 713 img/s |
| CPU preprocessing + cache flush | 164 µs |

## Communication overhead (reported separately)

| | CNN |
|---|---|
| UART transfer of one image (board RX) | 9.01 ms |
| PC round trip (send → reply) | 14.68 ms |
| UART share of the round trip | 61 % |
| End-to-end throughput | 68.1 img/s |

## FPGA resources (post-implementation)

| | CNN |
|---|---|
| LUT, whole design | 11,950 / 53,200 (22.5 %) |
| LUT, accelerator IP only | 11,110 |
| FF, whole design | 14,272 / 106,400 (13.4 %) |
| FF, accelerator IP only | 13,446 |
| BRAM tiles (36 Kb), whole design | 46.5 / 140 (33.2 %) |
| BRAM tiles (36 Kb), accelerator IP only | 46.5 |
| DSP, whole design | 102 / 220 (46.4 %) |
| DSP, accelerator IP only | 102 |

## Power and energy (Vivado post-implementation estimate)

| | CNN |
|---|---|
| Total on-chip power | 1.793 W |
| – of which PS7 (ARM) | 1.261 W |
| – of which accelerator IP (dynamic) | 0.374 W |
| Energy per inference, IP only | 0.524 mJ |
| Energy per inference, whole chip | 2.51 mJ |

## Measurement conditions

- Board: PYNQ-Z2 (Zynq-7020, xc7z020clg400-1), bare-metal application on the ARM Cortex-A9
- Accelerator clock: 100 MHz (FCLK_CLK0)
- Test images: 12,000 images from ../../2D_Geometric_Shapes_Dataset_test (all split), identical for every network
- Image format: 60×60 grayscale, 8 bit, raw pixels sent by the PC
- UART: 4,000,000 baud, 8N1, 3,606-byte request frame, 29-byte reply (theoretical 9.02 ms per image)
- Timing: accelerator time = ap_start → ap_done, ARM global timer; UART time measured on the board; round trip measured on the PC
- Resources and power: Vivado post-implementation reports (power: vectorless estimate)
- Generated: 2026-11-17 13:53
