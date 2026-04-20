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

    datasets_dir = data_dir / "datasets"

    train_ds_path = datasets_dir / "train_ds.pth"
    valid_ds_path = datasets_dir / "valid_ds.pth"
    test_ds_path = datasets_dir / "test_ds.pth"

    def __init__(self):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


class DataConfig:
    val_fraction = 0.15
    train_batch = 16
    valid_batch = 32
    test_batch = 32

    input_dim = 512
    input_ch = 4
    n_classes = 28
    n_workers = 3
    

class ModelConfig:
    seed = 275
    backbone_lr = 0.0001
    classifier_lr = 0.001
    lr_reduce = 0.5
    lr_patience = 3
    focal_gamma = 2.5
    epochs = 20
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')