from config import DataConfig, PathConfig
from pathlib import Path
import random as rnd
import pandas as pd
import numpy as np

from torch import nn
from torch.optim import Optimizer
from torch.optim import lr_scheduler as lrs
import torch


def set_seed(seed: int):
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_train_data(df: pd.DataFrame) -> tuple[list, list]:
    ids = df['Id'].tolist()
    labels_list = []
    
    for label_str in df['Target']:
        hot_matrix = torch.zeros(DataConfig.n_classes, dtype=torch.float32)
        for class_idx in map(int, str(label_str).split()):
            hot_matrix[class_idx] = 1
        labels_list.append(hot_matrix)
        
    return ids, labels_list


def save_checkpoint(model: nn.Module, 
                    optimizer: Optimizer, 
                    scheduler: lrs.LRScheduler, 
                    epoch: int, model_name: str = "model") -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None
    }, Path(PathConfig.checkpoints_dir / f"{model_name}.pt"))