from config import PathConfig
from pathlib import Path
from PIL import Image
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
        return int(imagehash.phash(img.convert("L")))