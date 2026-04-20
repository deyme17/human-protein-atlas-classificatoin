from sklearn.model_selection import StratifiedGroupKFold
from .data_leakage import assert_no_leakage
from config import DataConfig
import pandas as pd
import numpy as np
import torch



def multilabel_stratify_key(y: np.ndarray) -> np.ndarray:
    """
    For each sample, return the index of its rarest positive class.
    """
    class_freq = y.sum(axis=0)
    class_freq[class_freq == 0] = np.inf
    weighted = y / class_freq[None, :]
    keys = np.where(y.sum(axis=1) > 0, np.argmax(weighted, axis=1),-1)
    return keys



def group_stratified_split(df: pd.DataFrame, 
                           labels: list[torch.Tensor],
                           val_fraction: float = DataConfig.val_fraction, 
                           n_splits: int = 5,
                           random_state: int = 42) -> tuple:
    """
    Drop-in replacement for iterative_train_test_split.
    """
    image_ids = df["Id"].tolist()
    groups = df["group_id"].values               # shape (N,)
    X = np.arange(len(image_ids)).reshape(-1, 1)
    y_np = torch.stack(labels).numpy()           # (N, n_classes)
    strat_key = multilabel_stratify_key(y_np)    # (N,)

    # pick n_splits so that one fold ~ val_fraction
    n_splits = max(n_splits, round(1 / val_fraction))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # take the first fold as validation
    train_idx, val_idx = next(splitter.split(X, strat_key, groups=groups))

    train_ids = [image_ids[i] for i in train_idx]
    val_ids = [image_ids[i] for i in val_idx]
    train_labels = [torch.tensor(y_np[i], dtype=torch.float32) for i in train_idx]
    val_labels = [torch.tensor(y_np[i], dtype=torch.float32) for i in val_idx]
    
    assert_no_leakage(df, train_ids, val_ids)

    return train_ids, train_labels, val_ids, val_labels