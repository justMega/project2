# CNN vs SNN shape classification on the PYNQ-Z2

SDU Embedded Systems (E26), Assignment 2: *Accelerated AI for Vision Applications*.
Teo Humar, Maks Jagodič, Mateusz Okula, Jérémy Flèche.

A PC sends an image over UART to the Zynq-7020 of a PYNQ-Z2 board. A bare-metal
application on the ARM processor passes it to a CNN or an SNN accelerator written in
Vitis HLS, reads the result and sends it back to the PC. Both networks classify six
shapes: circle, ellipse, figure-eight, rectangle, square and triangle.

## Results (12,000 test images, FPGA at 100 MHz)

| | CNN | SNN |
|---|---|---|
| Accuracy on the FPGA | **99.93 %** | 85.37 % |
| FPGA compute latency | 1,402 µs | **157 µs** |
| Compute-only throughput | 713 img/s | **6,369 img/s** |
| Data sent per image over UART | 3,606 B (raw 60×60 image) | 770 B (384 features) |
| End-to-end throughput (UART) | 62.5 img/s | 62.5 img/s |
| LUT / FF / BRAM / DSP | 11,950 / 14,272 / 46.5 / 102 | 8,813 / 9,200 / 13 / 8 |
| Energy per inference (IP, Vivado estimate) | 0.52 mJ | 9.4 µJ |

Details: [`eval/RESULTS_CNN.md`](eval/RESULTS_CNN.md), [`eval/RESULTS.md`](eval/RESULTS.md) (SNN),
[`eval/COMPARISON.md`](eval/COMPARISON.md).

## Repository layout

```
cnn/
  python/      training (train_cnn.py) and shared preprocessing
  runs/cnn/    trained Keras model, training curves and reports
  hls/         HLS accelerator (ds_cnn.cpp), testbench, weights, build script
  vivado/      block-design build script, exported hardware (.xsa), reports
  vitis/       bare-metal application (main.c) and UART protocol (PROTOCOL.md)
  pc/          PC client (UART test, evaluation, demo) and comparison script
snn/
  python/      SNN model and training (snn_mlp.py), feature extraction, UART client
  hls/         HLS accelerator (snn_mlp.cpp), testbench, weights, reports
  vivado/      block design (.bd) and exported hardware (.xsa)
  vitis/       bare-metal application (helloworld.c)
eval/          CPU and FPGA benchmarks of both networks, results
shapes-dataset/  dataset generator
```

## UART protocol (4 Mbaud, 8N1)

- CNN request (3,606 B): `A6 6A`, network ID, length (u16), 3,600 pixels, checksum.
- SNN request (770 B): `AA 55`, 384 × int16 features.
- Reply of both networks (31 B): `A5`, predicted class, time in µs (u32), 6 × int32 logits, `5A`.

Full description: [`cnn/vitis/PROTOCOL.md`](cnn/vitis/PROTOCOL.md).

## Reproducing

The datasets are not in the repository: generate them with
`shapes-dataset/generate_shapes_dataset.py` (training set in
`shapes-dataset/2D_Geometric_Shapes_Dataset/`, test set in `2D_Geometric_Shapes_Dataset_test/`).

- **CNN** (Vivado/Vitis 2026.1, Linux): see [`cnn/README.md`](cnn/README.md).
- **SNN**: the scripts in `snn/python/` use paths relative to the folder they are run
  from; run them from a folder that contains the dataset folders. The HLS component
  is built from `snn/hls/` (`hls_config.cfg`), the platform from `snn/vivado/design_1_wrapper.xsa`.
- **Benchmarks** (from the repository root, board programmed with the matching design):
  ```
  python3 eval/cnn_benchmark_sw.py && python3 eval/cnn_benchmark_fpga.py --port /dev/ttyUSB1
  python3 eval/benchmark_sw.py     && python3 eval/benchmark_fpga.py     --port /dev/ttyUSB1
  ```
