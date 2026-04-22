import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from torch.utils.data import DataLoader
from config import TrainConfig, PathConfig, DataConfig

from utils import save_checkpoint, set_seed, load_checkpoint, visualize_training
from training import train_epoch, eval_epoch
from models import get_model
from dataset import load_dataset, get_dataloader, get_sampler
from losses.losses_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from schedulers.scheduler_factory import get_scheduler



def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
          scheduler: lrs.LRScheduler|None = None, threshold: torch.Tensor|float = 0.3, num_epochs: int = TrainConfig.epochs, curr_epoch: int = 0, 
          max_norm: float|None = TrainConfig.max_norm, patience: int = TrainConfig.early_stop, model_name: str = "model", device: torch.device = TrainConfig.device) -> dict:
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available.")

    history = {'train_loss': [], 'valid_loss': [], 'f1_macro': [], 'f1_micro': [], 'f1_samples': []}
    best_f1_macro = 0.0
    patience_counter = 0

    for epoch in range(curr_epoch, num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, max_norm, epoch, num_epochs)
        valid_loss, metrics = eval_epoch(model, criterion, valid_loader, device, threshold, epoch, num_epochs)

        f1_macro = metrics['f1_macro']

        if scheduler is not None:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(f1_macro)
            else:
                scheduler.step()

        print(f"[Epoch: {epoch+1}] Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | F1_MACRO: {f1_macro:.3f}") 

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        for k, v in metrics.items():
            history[k].append(v)

        if f1_macro >= best_f1_macro:
            save_checkpoint(model, optimizer, scheduler, epoch, model_name)
            best_f1_macro = f1_macro
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    return history


# main section
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--name", type=str, default="model", help="Checkpoint name (used for saving)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--patience", type=int, default=TrainConfig.early_stop, help="Early stopping parameter")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Number of training epochs")
    parser.add_argument("--threshold", type=float, default=0.3, help="Classification threshold")
    parser.add_argument("--max_norm", type=float, default=None, help="Gradient clipping max norm")
    args = parser.parse_args()

    set_seed(TrainConfig.seed)

    # data
    train_ds = load_dataset(PathConfig.train_ds_path)
    valid_ds = load_dataset(PathConfig.valid_ds_path)

    sampler = (get_sampler(train_ds.labels, rare_threshold=TrainConfig.sampler["rare_threshold"])
                                                  if TrainConfig.sampler["use"] else None)
    
    print(f"Creating dataloaders with num_workers={DataConfig.n_workers}")
    train_loader = get_dataloader(
        train_ds,
        batch_size=DataConfig.train_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=DataConfig.n_workers,
        pin_memory=TrainConfig.device == torch.device("cuda")
    )
    valid_loader = get_dataloader(
        valid_ds,
        batch_size=DataConfig.valid_batch,
        shuffle=False,
        num_workers=DataConfig.n_workers,
        pin_memory=TrainConfig.device == torch.device("cuda")
    )

    # model
    model = get_model().to(TrainConfig.device)

    # optimizer / loss / scheduler
    param_groups = model.get_param_groups(TrainConfig.backbone_lr, TrainConfig.classifier_lr)
    optimizer = get_optimizer(param_groups)
    criterion = get_loss()

    curr_epoch = 0

    # load checkpoint
    if args.checkpoint is not None:
        model, state_dict = load_checkpoint(
            model_name=args.checkpoint, model=model
        )
        optimizer.load_state_dict(state_dict["optimizer"])
        curr_epoch = state_dict["epoch"] + 1
        print(f"Resumed '{args.checkpoint}' at epoch {curr_epoch}.")

    scheduler = get_scheduler(optimizer, last_epoch=curr_epoch - 1)
    if args.checkpoint is not None and state_dict["scheduler"] is not None:
        scheduler.load_state_dict(state_dict["scheduler"])

    # train
    model.print_model_info()
    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        scheduler=scheduler,
        threshold=args.threshold,
        num_epochs=args.epochs,
        curr_epoch=curr_epoch,
        max_norm=args.max_norm,
        patience=args.patience,
        model_name=args.name,
        device=TrainConfig.device,
    )
    visualize_training(history, save=True)