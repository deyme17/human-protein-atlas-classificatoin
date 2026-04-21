import torch
from pathlib import Path


class PathConfig:
    data_dir = Path("data").resolve()
    checkpoints_dir = Path("checkpoints").resolve()
    submission_dir = Path("submissions").resolve()

    # dataset structure
    train_dir = data_dir / "train"
    external_dir =  data_dir / "external"
    test_dir = data_dir / "test"

    train_labels_path = data_dir / "train.csv"
    external_labels_path = data_dir / "external.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    saved_datasets_dir = data_dir / "saved_datasets"

    train_ds_path = saved_datasets_dir / "train_ds.pth"
    valid_ds_path = saved_datasets_dir / "valid_ds.pth"
    test_ds_path = saved_datasets_dir / "test_ds.pth"

    # mkdirs
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    saved_datasets_dir.mkdir(parents=True, exist_ok=True)


class DataConfig:
    val_fraction = 0.15
    train_batch = 16
    valid_batch = 32
    test_batch = 32

    input_dim = 512
    input_ch = 4
    n_classes = 28
    n_workers = 3
    

class TrainConfig:
    seed = 275
    backbone_lr = 0.0001
    classifier_lr = 0.001

    epochs = 20
    early_stop = 5
    max_norm = 1.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
    model = {
        "name": "efficientnet_v2",
        "parameters": {
            "num_classes": DataConfig.n_classes,
            "freeze_until": -1,
            "dropout_rate": 0.3,
            "small": True,
        },
    }
    optimizer = {
        "name": "adamw",
        "parameters": {
            # lr is not required because optimizers accepts param_group as argument
            "betas": (0.9, 0.999),
            "weight_decay": 1e-4,
            "amsgrad": False,
        },
    }
    loss = {
        "name": "focal",
        "parameters": {
            "alphas": torch.tensor([0.00899, 0.09245, 0.03203, 0.07371, 0.06241, 0.0461, 0.11505, 0.04105, 2.20648, 2.54594, 4.13715, 
                                    0.10608, 0.16829, 0.21585, 0.10887, 5.5162, 0.21822, 0.55162, 0.12795, 0.07837, 0.67545, 0.03076, 
                                    0.14432, 0.03898, 0.36106, 0.01409, 0.35335, 9.92916], dtype=float, device=device),
            "gamma": 2.5,
            "reduction": "mean",
            "device": device
        },
    }
    lr_scheduler = {
        "name": "reduce_lr_on_plateau",
        "parameters": {
            "mode": "min", 
            "factor": 0.5, 
            "patience": 4,
            "threshold": 0.001, 
            "threshold_mode": "rel",
            "cooldown": 0, 
            "min_lr": 0
        },
    }
    sampler = {
        "use": True,
        "rare_threshold": 100,  
        # we use sampler only for super-rare classes,
        # because we already have focal/asl
    }