import random as rnd
import numpy as np
import torch

def set_seed(seed: int):
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)