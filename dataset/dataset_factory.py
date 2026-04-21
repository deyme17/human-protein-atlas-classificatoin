from config import DataConfig, PathConfig
from pathlib import Path

from torch.utils.data import DataLoader, Sampler
import torch

from .dataset import ProteinDataset


def get_dataset(image_ids: list[str],
                labels: list[torch.Tensor] | None = None,
                train_root: Path = PathConfig.train_dir,
                external_root: Path = PathConfig.external_dir, 
                transform=None) -> ProteinDataset:
    return ProteinDataset(
        image_ids=image_ids,
        labels=labels,
        train_root=train_root,
        external_root=external_root, 
        transform=transform
    )


def get_dataloader(dataset: ProteinDataset,
                   batch_size: int,
                   shuffle: bool,
                   sampler: Sampler|None = None,
                   num_workers: int = DataConfig.n_workers,
                   drop_last: bool = False,
                   pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )