# Assignment 2: CNN part (shape recognition on PYNQ-Z2)

PC → UART → ARM (bare-metal) → CNN HLS IP → ARM → UART → PC, for 6 shape classes
(circle, square, fig_eight, rectangle, triangle, elipse).
Derived from the course reference [dbaniya/CNN_guest_lecture](https://github.com/dbaniya/CNN_guest_lecture).

```
cnn/
├── python/shapes_common.py   classes, 60x60 preprocessing, train/val/test split (SHARED with SNN)
├── python/train_cnn.py       Keras training, test evaluation, BN folding, weight + testbench export
├── hls/                      Vitis HLS accelerator (ds_cnn.cpp/.h), testbench, configs
├── vivado/build_cnn_bd.tcl   block design + bitstream + reports + .xsa, in one script
├── vitis/main.c              bare-metal application (UART, DDR buffer, IP control, timing)
├── vitis/PROTOCOL.md         UART frame format (SHARED with SNN)
└── pc/shape_client.py        PC app: UART loopback test, evaluation, demo display
    pc/board_emulator.py      fake board on a virtual serial port (test the PC app without the FPGA)
```

## Design decisions (for the report)

| Item | Choice | Why |
|---|---|---|
| Input | 60 × 60 grayscale uint8, `cv2.INTER_AREA` from 224 × 224 | Square aspect ratio keeps circle≠ellipse and square≠rectangle; 4× fewer pixels than 120×120 |
| Split | index 1–7000 train, 7001–8500 val, 8501–10000 test, per class | Deterministic and identical for CNN and SNN without sharing any file |
| Network | conv3×3(16)+pool → 3 depthwise-separable blocks (32 ch) with 2 pools → GAP → dense 6 (4.4 k params) | The course network (1 DS block, 8×8 px receptive field) only reached **78 %** in test: it cannot see aspect ratios (squares ↔ rectangles confused) |
| Fixed point | features Q8.8 (16 b), weights Q7.11 (18 b = DSP width), 32-bit accumulators, rounding + saturation | C-sim matches Keras (logit error < 0.003) |
| Memory | image in PS DDR (cache flushed), read by the IP via `m_axi` → S_AXI_HP0; all feature maps in on-chip BRAM/LUTRAM; weights in ROMs | Only 7 KB cross the bus; the IP never writes to DDR |
| Control | AXI-Lite: input pointer, ap_start/ap_done, prediction, 6 logits | |
| UART | 4 Mbaud 8N1, framed + checksum, raw pixels; normalisation on the ARM | Same input bytes for CNN and SNN (see `vitis/PROTOCOL.md`) |
| Timing | ARM global timer: accelerator (`ap_start→ap_done`), CPU preprocessing, board-side UART RX; PC round trip | UART time is reported separately from inference |

## Step by step

### 1. Train (on the PC, roughly 1 min per epoch on CPU)
```bash
pip install tensorflow opencv-python scikit-learn matplotlib pyserial
cd python
python3 train_cnn.py --data ../../shapes-dataset/2D_Geometric_Shapes_Dataset --out ../runs/cnn
# optional baseline for the report (course architecture):
python3 train_cnn.py --data ../../shapes-dataset/2D_Geometric_Shapes_Dataset --out ../runs/lecture --arch lecture
```
Outputs in `runs/cnn/`: `plots/` (loss/accuracy curves, test confusion matrices), `reports/`
(test classification report, per-image Keras predictions), `models/shape_cnn.keras`, `run_info.json`.
The weights (`weights_cstyle_folded/`) and testbench images (`tb_data/`) are **copied into `hls/` automatically**.
Check the last lines: a `[WARNING]` means a weight/activation is close to the fixed-point range.

### 2. Vitis HLS: C simulation, synthesis, IP export
```bash
cd hls
vitis-run --mode hls --csim    --config hls_config.cfg --work_dir ds_cnn_hls   # expect "*** TEST PASSED ***"
v++ -c --mode hls              --config hls_config.cfg --work_dir ds_cnn_hls   # synthesis -> latency + resources
vitis-run --mode hls --package --config hls_config.cfg --work_dir ds_cnn_hls   # IP for Vivado
```
(or `vitis-run --mode hls --tcl run_hls.tcl`, or the GUI with `hls_config.cfg`).
For the report, keep the csim output (accuracy, HLS-vs-Keras agreement) and the synthesis report (latency in cycles, LUT/FF/DSP/BRAM).
Expected order of magnitude: about 130 k cycles, around 1.3 ms at 100 MHz. The HLS report gives the real value.

### 3. Vivado: block design, bitstream, reports
```bash
cd vivado
vivado -mode batch -source build_cnn_bd.tcl -tclargs ../hls
```
Writes `shape_cnn.xsa` and `reports/utilization.rpt`, `power.rpt`, `timing.rpt` (post-implementation, for the comparison table).
Needs the PYNQ-Z2 board files. Otherwise, set the PS manually: UART0 on MIO 14–15, DDR preset of the board, HP0 enabled, FCLK0 = 100 MHz.

### 4. Vitis: bare-metal application
1. New platform component from `shape_cnn.xsa` (OS standalone, ps7_cortexa9_0).
2. New application component (empty C), add `vitis/main.c`, build.
3. Program the FPGA and run. At boot it prints at 115200 baud the CNN base address and register offsets, then switches the UART to 4 Mbaud.

If the build stops at `#error CNN base address not found`, search `DS_CNN` in `xparameters.h` and set `CNN_BASE`.
If 4 Mbaud is unreliable, set `UART_BAUD` to 921600 in `main.c` **and** use `--baud 921600` on the PC.

### 5. Test the UART alone (assignment step 3)
```bash
cd pc
python3 shape_client.py uart-test --port /dev/ttyUSB1 --n 100
```
Sends random 60×60 frames that the board echoes back (no IP involved). Checks every byte and measures the transfer time.

### 6. Full evaluation + demo
```bash
python3 shape_client.py eval --port /dev/ttyUSB1 --data ../../shapes-dataset/2D_Geometric_Shapes_Dataset --net cnn
python3 shape_client.py demo --port /dev/ttyUSB1 --data ../../shapes-dataset/2D_Geometric_Shapes_Dataset --net cnn
```
`eval` runs the 9 000 test images (use `--max-per-class 200` for a quicker run). It prints overall and per-class accuracy, the confusion matrix, accelerator latency and throughput, CPU time, UART time and end-to-end throughput, and saves `results/summary_fpga.json`, CSV files and confusion PNG.
`demo` shows images with their true label and each network's prediction (for the video).
Once the SNN is in the same bitstream (or with DFX), use `--net cnn snn` to compare both on the same images.

Without the board: `python3 board_emulator.py --keras ../runs/cnn/models/shape_cnn.keras` prints a `/dev/pts/N` port to use instead of `/dev/ttyUSB1`.

## For the SNN teammate
- Use `python/shapes_common.py` (`CLASSES`, `list_split`, `load_u8`) so the labels, split and 60 × 60 images are identical.
- The firmware already dispatches `net = 0x01` to `run_snn(px, &result)` when compiled with `HAVE_SNN`. It receives the same 3600 raw pixels; Sobel/feature extraction can run on the ARM there.
