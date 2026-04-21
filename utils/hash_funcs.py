from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
import imagehash
import hashlib



def compute_hashes(img_path: Path) -> tuple[str, int]:
    """Compute md5 hash and phash (single file read)."""
    with open(img_path, "rb") as f:
        raw = f.read()
    md5 = compute_hash(raw)
    ph = compute_phash(raw)
    return md5, ph


def compute_hash(raw: BytesIO) -> str:
    return hashlib.md5(raw).hexdigest()


def compute_phash(raw: bytes) -> int:
    img = Image.open(BytesIO(raw))
    arr = np.array(img)
    # use only RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    img = Image.fromarray(arr, mode="RGB")
    ph = imagehash.phash(img)
    return int(str(ph), 16)