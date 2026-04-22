import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig, PathConfig, DataConfig
from utils import load_checkpoint
from models import get_model
from dataset import load_dataset, get_dataloader



def predict(model: nn.Module, loader: DataLoader,
            device: torch.device = TrainConfig.device) -> torch.Tensor:
    probs = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Inference", leave=False):
            imgs = imgs.to(device)
            probs.append(torch.sigmoid(model(imgs)).cpu())
    return torch.cat(probs, dim=0)  # (N, C)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference, save probs tensor.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to load")
    args = parser.parse_args()

    out_path = PathConfig.probs_dir / f"{args.checkpoint}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # data
    test_ds = load_dataset(PathConfig.test_ds_path)
    test_loader = get_dataloader(
        test_ds,
        batch_size=DataConfig.test_batch,
        shuffle=False,
        num_workers=DataConfig.n_workers,
    )
    # model
    model = get_model().to(TrainConfig.device)
    model, _ = load_checkpoint(model_name=args.checkpoint, model=model)
    model.print_model_info()

    # prediction
    probs = predict(model, test_loader, device=TrainConfig.device)
    torch.save({"probs": probs, "image_ids": test_ds.image_ids}, out_path)
    print(f"Probs saved to '{out_path}' {tuple(probs.shape)}.")