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
                 transform=None
                ):
        self.image_ids = image_ids
        self.labels = labels
        self.train_root = train_root
        self.external_root = external_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(self.image_ids[idx])
        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image, torch.tensor([])

    def _load_image(self, image_id: str) -> torch.Tensor:
        """
        Load 4-channel PNG image tensor: ((R, G, B, Y), H, W)
        """
        path = self._resolve_root(image_id) / f"{image_id}.png"
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        image = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def _resolve_root(self, image_id: str) -> Path:
        return self.train_root if '-' in image_id else self.external_root