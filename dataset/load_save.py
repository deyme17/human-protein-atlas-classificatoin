from .dataset import ProteinDataset
from config import PathConfig
import torch
from pathlib import Path


def save_dataset(ds: ProteinDataset, path: Path):
    state = {
        'image_ids': ds.image_ids,
        'labels': ds.labels,
        'transform': ds.transform 
    }
    torch.save(state, path)


def load_dataset(path: Path, 
                 train_root: Path = PathConfig.train_dir, 
                 external_root: Path = PathConfig.external_dir, 
                 transform=None) -> ProteinDataset:
    state = torch.load(path, weights_only=False)
    ds = ProteinDataset(
        image_ids=state['image_ids'],
        labels=state['labels'],
        train_root=train_root,
        external_root=external_root,
        transform=transform or state['transform']
    )
    return ds