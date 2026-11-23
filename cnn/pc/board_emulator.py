#!/usr/bin/env python3
"""
Software stand-in for the PYNQ-Z2 firmware (same UART protocol as vitis/main.c),
so the PC application can be developed and tested without the board.

It creates a virtual serial port (Linux/macOS pty) and prints its name:
  python3 board_emulator.py                         # dummy predictions
  python3 board_emulator.py --keras runs/cnn/models/shape_cnn.keras   # real CNN predictions
then, in another terminal:
  python3 shape_client.py uart-test --port /dev/pts/N
  python3 shape_client.py eval --port /dev/pts/N --data ... --max-per-class 20
"""
import argparse
import os
import struct
import sys
import time
import tty

import numpy as np

IMG = 60
N_PIX = IMG * IMG
N_CLASSES = 6
# inference reply (same as the SNN): 0xA5 <pred> <time_us u32> <6 x i32> 0x5A
CNN_TO_COMMON = np.array([0, 4, 2, 3, 5, 1])        # CNN order -> alphabetical order
ERROR_REPLY = bytes([0xA5, 0xFF]) + bytes(4 + 4 * N_CLASSES) + b"\x5A"


def read_n(fd, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", help="trained .keras model for real predictions (CNN)")
    args = ap.parse_args()

    predict = None
    if args.keras:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        model = tf.keras.models.load_model(args.keras)
        names = [l.name for l in model.layers]
        last = "fc" if "fc" in names else "gap_logits"   # layer before the softmax
        logit_model = tf.keras.Model(model.inputs, model.get_layer(last).output)
        lut = ((1024 * np.arange(256) + 127) // 255 - 512).astype(np.float32) / 256.0  # = ARM Q8.8

        def predict(px):
            x = lut[px].reshape(1, IMG, IMG, 1)
            return logit_model(x, training=False).numpy()[0]

    master, slave = os.openpty()
    tty.setraw(slave)
    print(f"Emulated board on {os.ttyname(slave)}  (Ctrl+C to stop)", flush=True)

    while True:
        try:
            if read_n(master, 1)[0] != 0xA6:
                continue
            if read_n(master, 1)[0] != 0x6A:
                continue
            t0 = time.perf_counter()
            net = read_n(master, 1)[0]
            (length,) = struct.unpack("<H", read_n(master, 2))
            if length != N_PIX:
                os.write(master, ERROR_REPLY)
                continue
            px = np.frombuffer(read_n(master, N_PIX), dtype=np.uint8)
            csum = read_n(master, 1)[0]
            rx_us = int((time.perf_counter() - t0) * 1e6)
            if csum != int(px.sum()) & 0xFF:
                if net == 0x7F:
                    os.write(master, bytes([0xA5, 0x7F, 2]) + bytes(4 + N_PIX) + b"\x5A")
                else:
                    os.write(master, ERROR_REPLY)
                continue
            if net == 0x7F:
                os.write(master, bytes([0xA5, 0x7F, 0]) + struct.pack("<I", rx_us)
                         + px.tobytes() + b"\x5A")
                continue
            if net == 0x00:
                t1 = time.perf_counter()
                if predict is not None:
                    logits = predict(px)
                else:  # dummy: deterministic pseudo-logits
                    logits = np.sin(np.arange(N_CLASSES) + float(px.mean()))
                infer_ns = int((time.perf_counter() - t1) * 1e9)
                q = np.clip(np.round(logits * 256), -32768, 32767).astype(np.int64)
                common = np.zeros(N_CLASSES, "<i4")
                common[CNN_TO_COMMON] = q                    # reorder to the common order
                reply = (bytes([0xA5, int(CNN_TO_COMMON[int(np.argmax(logits))])])
                         + struct.pack("<I", (infer_ns + 500) // 1000) + common.tobytes()
                         + b"\x5A")
            else:
                reply = ERROR_REPLY
            os.write(master, reply)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
