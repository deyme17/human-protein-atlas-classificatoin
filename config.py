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
    probs_dir = submission_dir / "probs"
    thresholds_dir = checkpoints_dir / "thresholds"

    train_ds_path = saved_datasets_dir / "train_ds.pth"
    valid_ds_path = saved_datasets_dir / "valid_ds.pth"
    test_ds_path = saved_datasets_dir / "test_ds.pth"

    # mkdirs
    train_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    saved_datasets_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)
    thresholds_dir.mkdir(parents=True, exist_ok=True)

class DataConfig:
    val_fraction = 0.15
    train_batch = 32
    valid_batch = 16
    test_batch = 16

    input_dim = 512
    input_ch = 4
    n_classes = 28

    n_workers = 4
    persistent_workers = n_workers > 0
    prefetch_factor = 2

class TrainConfig:
    seed = 277
    backbone_lr = 0.0003
    classifier_lr = 0.001

    epochs = 20
    early_stop = 5
    max_norm = 1.5
    device = torch.device('cuda')
 
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
            "weight_decay": 5e-3,
            "amsgrad": False,
        },
    }
    loss = {
        "name": "focal",
        "parameters": {
            "alphas": torch.tensor([0.01845, 0.17107, 0.04861, 0.15917, 0.10345, 
                                    0.09003, 0.14209, 0.05644, 2.42231, 2.79383, 
                                    2.97643, 0.23818, 0.24082, 0.36874, 0.19757, 
                                    8.2799, 0.39912, 1.19213, 0.28303, 0.14384, 
                                    1.21438, 0.03857, 0.19495, 0.05152, 1.26498, 
                                    0.01897, 0.75148, 4.13995], dtype=float),
            "gamma": 2.2,
            "reduction": "mean",
            "device": device
        },
    }
    lr_scheduler = {
        "name": "reduce_lr_on_plateau",
        "parameters": {
            "mode": "max", 
            "factor": 0.65, 
            "patience": 4,
            "threshold": 0.001, 
            "threshold_mode": "rel",
            "cooldown": 0, 
            "min_lr": 0
        },
    }
    sampler = {
        "use": False,
        "rare_threshold": 100,
        # we use sampler only for super-rare classes,
        # because we already have focal/asl
    }