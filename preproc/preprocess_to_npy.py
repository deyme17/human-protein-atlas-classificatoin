import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

from config import PathConfig, DataConfig


TARGET_SIZE = (DataConfig.input_dim, DataConfig.input_dim)


def process_one(image_id: str, out_dir: Path) -> str | None:
    out_path = out_dir / f"{image_id}.npy"
    if out_path.exists():
        return None

    src_root = PathConfig.train_dir if '-' in image_id else PathConfig.external_dir
    src_path = src_root / f"{image_id}.png"
    if not src_path.exists():
        return None

    try:
        img = Image.open(src_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        np.save(out_path, arr)
    except Exception as e:
        return f"ERROR {image_id}: {e}"
    return None


def preprocess(image_ids: list[str], out_dir: Path, num_workers: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_one, iid, out_dir): iid for iid in image_ids}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            result = f.result()
            if result:
                errors.append(result)

    print(f"Done. {len(errors)} errors.")
    for e in errors:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # collect all image IDs from both sources
    train_ids = pd.read_csv(PathConfig.train_labels_path)["Id"].tolist()
    ext_ids = pd.read_csv(PathConfig.external_labels_path)["Id"].tolist()
    all_ids = train_ids + ext_ids

    print(f"Preprocessing {len(all_ids)} images -> {PathConfig.npy_cache_dir}")
    preprocess(all_ids, PathConfig.npy_cache_dir, num_workers=args.workers)