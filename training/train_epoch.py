from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_



def train_epoch(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, loader: DataLoader, 
                device: torch.device, max_norm: float | None,epoch: int, num_epochs: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        if max_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader.dataset)