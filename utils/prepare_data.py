from config import DataConfig
import random as rnd
import pandas as pd
import numpy as np
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