from .dataset import ProteinDataset
import torch
from pathlib import Path


def save_dataset(ds: ProteinDataset, path: Path):
    state = {
        'image_ids': ds.image_ids,
        'labels': ds.labels,
        'train_root': ds.train_root,
        'external_root': ds.external_root,
        'transform': ds.transform 
    }
    torch.save(state, path)


def load_dataset(path: Path, transform=None) -> ProteinDataset:
    state = torch.load(path, weights_only=False)
    ds = ProteinDataset(
        image_ids=state['image_ids'],
        labels=state['labels'],
        train_root=state['train_root'],
        external_root=state['external_root'],
        transform=transform or state['transform']
    )
    return ds