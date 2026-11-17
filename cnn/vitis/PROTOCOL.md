# UART protocol PC ↔ PYNQ-Z2 (shared by CNN and SNN)

Physical link: PS UART0 through the PYNQ-Z2 micro-USB (FTDI). **4 000 000 baud, 8N1, no flow control.**
All multi-byte fields are little-endian.

## Image format
- 60 × 60 pixels, 8-bit grayscale (0 = black, 255 = white), row-major (row 0 first).
- Produced on the PC by `shapes_common.load_u8()`: `cv2.imread(GRAYSCALE)` then `cv2.resize(…, (60, 60), INTER_AREA)` from the 224 × 224 dataset images.
- The PC sends **raw pixels**. Each network's own normalisation is done on the board, so both networks receive exactly the same data.
  - CNN: CPU converts to Q8.8 `q = round(1024·x/255) − 512` (value = 4x/255 − 2).
  - SNN: up to the SNN application (e.g. Sobel ON/OFF features computed on the ARM).

## PC → board: request frame (3 606 bytes)

| Byte | Field | Value |
|---|---|---|
| 0 | start 0 | `0xA6` |
| 1 | start 1 | `0x6A` |
| 2 | net | `0x00` CNN, `0x01` SNN, `0x7F` UART loopback test |
| 3–4 | length | payload size, uint16 = 3600 (`10 0E`) |
| 5–3604 | payload | 3600 uint8 pixels |
| 3605 | checksum | sum of the 3600 payload bytes modulo 256 |

## Board → PC: inference reply (29 bytes)

| Byte | Field | Type |
|---|---|---|
| 0 | start | `0xA5` |
| 1 | net | echo of request |
| 2 | status | 0 OK, 1 bad length, 2 bad checksum, 3 network not in this bitstream, 4 timeout |
| 3 | predicted class | uint8 (0 circle, 1 square, 2 fig_eight, 3 rectangle, 4 triangle, 5 elipse) |
| 4–7 | infer_ns | uint32, accelerator time `ap_start → ap_done` (ARM global timer) |
| 8–11 | prep_ns | uint32, CPU preprocessing + cache flush |
| 12–15 | rx_us | uint32, UART reception time measured on the board (first header byte → checksum) |
| 16–27 | logits[6] | int16, Q8.8 (divide by 256) |
| 28 | end | `0x5A` |

## Board → PC: loopback reply (`net = 0x7F`, 3 609 bytes)
`0xA5, 0x7F, status, rx_us (uint32), 3600 echoed pixels, 0x5A`. The PC compares the echoed pixels with the sent ones, which tests the link in both directions before any hardware IP is involved.

## Timing at 4 Mbaud (10 bits per byte on the wire)
Request: 3606 B → 9.0 ms. Reply: 29 B → 0.07 ms. Loopback reply: 3609 B → 9.0 ms.
