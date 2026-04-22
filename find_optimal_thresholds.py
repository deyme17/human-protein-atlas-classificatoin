import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path

from config import TrainConfig, PathConfig, DataConfig
from utils import load_checkpoint
from models import get_model
from dataset import load_dataset, get_dataloader
from inference import predict



def find_optimal_thresholds(y_true: torch.Tensor, y_prob: torch.Tensor,
                            n_steps: int = 50) -> torch.Tensor:
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()
    n_classes = y_true_np.shape[1]

    thresholds, f1s = [], []

    for cls_idx in tqdm(range(n_classes), desc="Optimizing thresholds"):
        if y_true_np[:, cls_idx].sum() == 0:
            thresholds.append(0.95)
            f1s.append(0.0)
            print(f"\tClass {cls_idx}: no positives in val -> threshold=1.0")
            continue

        best_thresh, best_f1 = 0.2, 0.0
        for thresh in np.linspace(0.05, 0.95, n_steps):
            y_pred = (y_prob_np[:, cls_idx] >= thresh).astype(float)
            f1 = f1_score(y_true_np[:, cls_idx], y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        thresholds.append(best_thresh)
        f1s.append(best_f1)
        print(f"\tClass {cls_idx}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")

    avg_f1 = sum(f1s) / len(f1s)
    print(f"\nAvg F1 @ optimal thresholds: {avg_f1:.4f}")
    return torch.tensor(thresholds, dtype=torch.float32)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find optimal per-class thresholds on validation set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to load")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of threshold candidates per class")
    args = parser.parse_args()

    out_path = PathConfig.thresholds_dir / f"{args.checkpoint}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # data
    valid_ds = load_dataset(PathConfig.valid_ds_path)
    valid_loader = get_dataloader(
        valid_ds,
        batch_size=DataConfig.valid_batch,
        shuffle=False,
        num_workers=DataConfig.n_workers,
    )

    # model
    model = get_model().to(TrainConfig.device)
    model, _ = load_checkpoint(model_name=args.checkpoint, model=model)

    # prediction
    y_prob = predict(model, valid_loader, device=TrainConfig.device)
    y_true = torch.stack(valid_ds.labels)                      # (N, C)

    # find_optimal_thresholds and save
    thresholds = find_optimal_thresholds(y_true, y_prob, n_steps=args.n_steps)
    torch.save(thresholds, out_path)
    print(f"Thresholds saved to '{out_path}'.")