from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def eval_epoch(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: torch.device, 
               threshold: torch.Tensor | float, epoch: int, num_epochs: int) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_targets, all_predictions = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            loss = criterion(out, labels)

            preds = (torch.sigmoid(out) >= threshold).int()
            all_predictions.append(preds.cpu())
            all_targets.append(labels.cpu())

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    preds_np   = torch.cat(all_predictions).numpy()
    targets_np = torch.cat(all_targets).numpy()

    metrics = {
        'f1_macro': f1_score(targets_np, preds_np, average='macro', zero_division=0),
        'f1_micro': f1_score(targets_np, preds_np, average='micro', zero_division=0),
        'f1_samples': f1_score(targets_np, preds_np, average='samples', zero_division=0),
    }
    return total_loss / len(loader.dataset), metrics