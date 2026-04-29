from config import DataConfig, PathConfig
from pathlib import Path

from torch.utils.data import DataLoader, Sampler
import torch

from .dataset import ProteinDataset


def get_dataset(image_ids: list[str],
                labels: list[torch.Tensor] | None = None,
                data_root: Path = PathConfig.train_dir,
                external_root: Path = PathConfig.external_dir, 
                transform=None) -> ProteinDataset:
    return ProteinDataset(
        image_ids=image_ids,
        labels=labels,
        data_root=data_root,
        external_root=external_root, 
        transform=transform
    )


def get_dataloader(dataset: ProteinDataset,
                   batch_size: int, shuffle: bool,
                   sampler: Sampler|None = None,
                   num_workers: int = DataConfig.n_workers,
                   prefetch_factor: int = DataConfig.prefetch_factor,
                   persistent_workers: bool = DataConfig.persistent_workers,
                   drop_last: bool = False, pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else None,
        drop_last=drop_last,
        pin_memory=pin_memory
    )