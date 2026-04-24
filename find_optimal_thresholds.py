import torch
import numpy as np

from config import TrainConfig, PathConfig, DataConfig
from utils import load_checkpoint
from models import get_model
from dataset import load_dataset, get_dataloader
from inference import predict



def find_optimal_thresholds(y_true: torch.Tensor, y_prob: torch.Tensor) -> torch.Tensor:
    """
    For each class, pick the threshold so that the fraction of positive
    predictions ~ fraction of positive examples in the reference set.
    The idea was stolen from this smart asian guy: https://www.kaggle.com/pudae81 :)
    """
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()
    n_classes = y_true_np.shape[1]

    prevalences = y_true_np.mean(axis=0)

    thresholds = []
    for cls_idx in range(n_classes):
        p = prevalences[cls_idx]
        # The threshold that yields ~p positive predictions
        # is simply the (1-p) quantile of predicted probabilities
        thresh = float(np.clip(np.quantile(y_prob_np[:, cls_idx], 1.0 - p), 0.05, 0.95))
        thresholds.append(thresh)

    return torch.tensor(thresholds, dtype=torch.float32)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find optimal per-class thresholds on validation set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to load")
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
    y_true = torch.stack(valid_ds.labels)   # (N, C)

    # find_optimal_thresholds and save
    thresholds = find_optimal_thresholds(y_true, y_prob)
    torch.save(thresholds, out_path)
    print(f"Thresholds saved to '{out_path}'.")