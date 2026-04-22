import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

from config import PathConfig, DataConfig


TARGET_SIZE = (DataConfig.input_dim, DataConfig.input_dim)


def load_channel(path: Path) -> np.ndarray:
    img = imageio.imread(str(path))
    return img[:, :, 0] if img.ndim == 3 else img


def process_one(image_id: str, out_dir: Path, from_rgby: bool) -> str | None:
    out_path = out_dir / f"{image_id}.npy"
    if out_path.exists():
        return None

    src_root = PathConfig.train_dir if '-' in image_id else PathConfig.external_dir
    try:
        if from_rgby:
            src_path = src_root / f"{image_id}.png"
            if not src_path.exists():
                return f"MISSING {image_id}"

            img = Image.open(src_path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
        else:
            channels = {c: src_root / f"{image_id}_{c}.png" for c in ("red", "green", "blue", "yellow")}
            if not all(p.exists() for p in channels.values()):
                return f"MISSING {image_id}"

            arr = np.stack([load_channel(p) for p in channels.values()], axis=2)  # (H, W, 4)
            img = Image.fromarray(arr, mode="RGBA")

        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        np.save(out_path, np.asarray(img, dtype=np.uint8))

    except Exception as e:
        return f"ERROR {image_id}: {e}"
    return None


def preprocess(image_ids: list[str], out_dir: Path, from_rgby: bool, num_workers: int = 8) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors, valid_ids = [], []

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_one, iid, out_dir, from_rgby): iid for iid in image_ids}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            iid = futures[f]
            result = f.result()
            if result:
                errors.append(result)
            else:
                valid_ids.append(iid)

    print(f"Done. {len(valid_ids)}/{len(image_ids)} valid. {len(errors)} errors.")
    for e in errors:
        print(e)

    return valid_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--from_rgby", action="store_true",
                        help="Load from single RGBY(A) PNGs")
    args = parser.parse_args()

    train_ids = pd.read_csv(PathConfig.train_labels_path)["Id"].tolist()
    ext_ids = pd.read_csv(PathConfig.external_labels_path)["Id"].tolist()
    all_ids = train_ids + ext_ids

    mode = "RGBY channels" if not args.from_rgby else "merged RGBA PNG"
    print(f"Preprocessing {len(all_ids)} images from {mode} -> {PathConfig.npy_cache_dir}")

    valid_ids = preprocess(all_ids, PathConfig.npy_cache_dir, args.from_rgby, args.workers)
    pd.Series(valid_ids).to_csv(PathConfig.npy_cache_dir / "valid_ids.csv", index=False, header=False)