"""
Shared definitions for the Assignment 2 shape classifiers (CNN and SNN).

Both networks MUST use this file so that:
  - the class order (label indices) is identical,
  - the train / validation / test split is identical,
  - the image preprocessing (grayscale, 60x60) is identical.

Split rule (deterministic, no random state needed):
  The generator writes <class>/<class>_<i>.png with i = 1 .. 10000, and every
  image is drawn independently, so splitting by index is unbiased:
      i in    1 ..  7000  -> train      (70 %)
      i in 7001 ..  8500  -> validation (15 %)
      i in 8501 .. 10000  -> test       (15 %)
"""
from pathlib import Path
import re

import cv2
import numpy as np

# Label index = position in this list. Do not reorder.
CLASSES = ["circle", "square", "fig_eight", "rectangle", "triangle", "elipse"]
N_CLASSES = len(CLASSES)

# Network input: 60 x 60 grayscale.
# Square on purpose: the source images are 224 x 224, so a non-square size
# (e.g. the lecture's 60 x 80) would stretch circles into ellipses and
# squares into rectangles, i.e. destroy exactly what separates the classes.
IMG_H, IMG_W = 60, 60

TRAIN_MAX_IDX = 7000
VAL_MAX_IDX = 8500

_IDX_RE = re.compile(r"_(\d+)\.png$", re.IGNORECASE)


def split_of(path: Path) -> str:
    m = _IDX_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot read image index from {path.name}")
    i = int(m.group(1))
    if i <= TRAIN_MAX_IDX:
        return "train"
    if i <= VAL_MAX_IDX:
        return "val"
    return "test"


def list_split(root: Path, split: str):
    """Return sorted (paths, labels) of one split ("train", "val", "test"),
    or of every image in the folder with split="all" (external test sets)."""
    paths, labels = [], []
    for label, cname in enumerate(CLASSES):
        d = Path(root) / cname
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        files = sorted(d.glob("*.png"), key=lambda p: int(_IDX_RE.search(p.name).group(1)))
        for p in files:
            if split == "all" or split_of(p) == split:
                paths.append(p)
                labels.append(label)
    return paths, np.asarray(labels, dtype=np.int64)


def load_u8(path: Path) -> np.ndarray:
    """Read one image as the 60 x 60 uint8 grayscale array that is sent over UART."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    if img.shape != (IMG_H, IMG_W):
        # INTER_AREA averages the source pixels, so 1-2 px lines survive the downscale.
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def load_split_u8(root: Path, split: str, cache_dir: Path = None):
    """Load a whole split as uint8 (N, 60, 60). Cached as .npz for speed."""
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{split}_{IMG_H}x{IMG_W}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=False)
            return d["x"], d["y"], [Path(p) for p in d["paths"]]
    paths, labels = list_split(root, split)
    x = np.stack([load_u8(p) for p in paths])
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, y=labels, paths=np.array([str(p) for p in paths]))
    return x, labels, paths


# ---------------- CNN-specific normalization ----------------
# z = (x/255 - 0.5) / 0.25 = 4x/255 - 2   (range [-2, 2])
# On the board the ARM converts each received uint8 pixel to Q8.8 with
#   q = round(1024 * x / 255) - 512      (integer, see u8_to_q88 below)
def cnn_normalize(x_u8: np.ndarray) -> np.ndarray:
    return (x_u8.astype(np.float32) / 255.0 - 0.5) / 0.25


def u8_to_q88(x_u8: np.ndarray) -> np.ndarray:
    """Exactly what the bare-metal app computes (integer arithmetic)."""
    px = x_u8.astype(np.int32)
    return ((1024 * px + 127) // 255 - 512).astype(np.int16)
