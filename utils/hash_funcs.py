from config import PathConfig
from pathlib import Path
from PIL import Image
import numpy as np
import imagehash
import hashlib


def compute_hash(img_id: str, root: Path = PathConfig.train_dir) -> str:
    h = hashlib.md5()
    with open(root / f"{img_id}.png", "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(img_id: str, root: Path = PathConfig.train_dir) -> int:
    with Image.open(root / f"{img_id}.png") as img:
        arr = np.array(img)
        # use only 3 channels: RGB
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        img = Image.fromarray(arr, mode="RGB")
        ph = imagehash.phash(img)
        return int(str(ph), 16)