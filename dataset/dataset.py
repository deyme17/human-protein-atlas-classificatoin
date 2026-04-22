import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from config import PathConfig
from pathlib import Path
from PIL import Image



class ProteinDataset(Dataset):
    def __init__(self, image_ids: list[str],
                 labels: list[torch.Tensor] | None = None,
                 train_root: Path = PathConfig.train_dir,
                 external_root: Path = PathConfig.external_dir,
                 npy_cache_dir: Path = PathConfig.npy_cache_dir,
                 transform=None
                ):
        self.image_ids = image_ids
        self.labels = labels
        self.train_root = train_root
        self.external_root = external_root
        self.transform = transform
        self.npy_cache_dir = npy_cache_dir
        if not npy_cache_dir.exists():
            raise FileNotFoundError(
                f"NPY cache not found at {npy_cache_dir}."
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(self.image_ids[idx])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx] if self.labels is not None else torch.tensor([])
        return image, label

    def _load_image(self, image_id: str) -> torch.Tensor:
        """
        Load pre-resized uint8 .npy -> float32 tensor (4, H, W) in [0, 1].
        """
        npy_path = self.npy_cache_dir / f"{image_id}.npy"
        arr = np.load(npy_path)                       # (H, W, 4) uint8
        arr = arr.astype(np.float32) * (1.0 / 255.0)  # fused cast+scale, no copy
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_image_png(self, image_id: str) -> torch.Tensor:
        root = self.train_root if '-' in image_id else self.external_root
        path = root / f"{image_id}.png"
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()