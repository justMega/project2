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

## Board → PC: inference reply (31 bytes, same format as the SNN application)

`0xA5 <pred> <time_us u32 LE> <6 logits i32 LE> 0x5A` → 7 + 4 × 6 = 31 bytes.

| Byte | Field | Type |
|---|---|---|
| 0 | start | `0xA5` |
| 1 | predicted class | uint8, common class order (alphabetical, as the SNN): 0 circle, 1 elipse, 2 fig_eight, 3 rectangle, 4 square, 5 triangle. `0xFF` = error (bad length, bad checksum, timeout or network not in this bitstream) |
| 2–5 | time_us | uint32, accelerator time `ap_start → ap_done` in µs (ARM global timer) |
| 6–29 | logits[6] | int32, same class order as `pred`. CNN: Q8.8 (divide by 256) |
| 30 | end | `0x5A` |

The firmware converts the CNN's own class order (circle, square, fig_eight, rectangle,
triangle, elipse, used for training) to the common order before sending; the PC client
converts it back. The CPU preprocessing time and the board-side UART reception time are
still measured on the board (printed with `DEBUG_PRINT 1`) but are not in the reply; the
PC reports the UART time as the theoretical wire time.

## Board → PC: loopback reply (`net = 0x7F`, 3 609 bytes)
`0xA5, 0x7F, status (0 OK, 1 bad length, 2 bad checksum, 4 timeout), rx_us (uint32), 3600 echoed pixels, 0x5A`. The PC compares the echoed pixels with the sent ones, which tests the link in both directions before any hardware IP is involved.

## Timing at 4 Mbaud (10 bits per byte on the wire)
Request: 3606 B → 9.0 ms. Reply: 31 B → 0.08 ms. Loopback reply: 3609 B → 9.0 ms.
