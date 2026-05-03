# human-protein-atlas-classification

Human Protein Atlas (HPA) image classification pipeline with focus on class imbalance mitigation, data leakage prevention, and robust multi-label training on 4-channel microscopy images.

## Overview

Multi-label classification of 28 subcellular protein localization patterns from 4-channel fluorescence microscopy images (RGBY: red, green, blue, yellow). The pipeline handles extreme class imbalance (pos/neg ratio up to 1:2600) and prevents data leakage via cell-line-aware stratified splitting.

## Results

**Public Leaderboard Score: 0.55457** (Macro F1) — achieved with a single EfficientNetV2-S model, no ensembling.

| Split | Macro F1 |
|---|---|
| Validation | 0.583 |
| Public Leaderboard | 0.555 |

### Training Configuration

| Parameter | Value |
|---|---|
| Input size | 512 × 512 |
| Backbone | EfficientNetV2-S |
| Dropout | 0.3 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Weight decay | 5e-3 |
| Backbone LR | 3e-4 |
| Classifier LR | 1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.65, patience=4) |
| Loss | Focal (γ=2.2, per-class α) |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | 5 epochs |
| Grad clip (max norm) | 1.5 |

## Project Structure

```
human-protein-atlas-classification/
├── config.py                    # PathConfig, DataConfig, TrainConfig
├── train.py                     # training entry point
├── inference.py                 # saves probs tensor to disk
├── find_optimal_thresholds.py   # per-class threshold optimization on validation set
├── make_submission.py           # builds submission.csv from probs + thresholds
├── create_dataset.py            # preprocessing → saves .pth datasets
├── data/
│
├── training/
│   ├── train_epoch.py
│   └── eval_epoch.py
│
├── preproc/
│   ├── stratified_split.py      # cell-line-aware train/val split
│   └── data_leakage.py          # leakage detection utilities
│
├── dataset/
│   ├── dataset.py               # ProteinDataset
│   └── dataset_factory.py       # get_dataloader()
|
├── models/                      # get_model()
├── losses/                      # get_loss() Focal loss / ASL
├── transforms/                  # get_transforms()
├── optimizers/                  # get_optimizer(), Adam / AdamW / SGD
├── schedulers/                  # get_scheduler(), Cosine / StepLR / ReduceLROnPlateau
│
├── utils/
├── tools/
│   ├── convert_to_rgby.py       # converts raw images to 4-channel .png
│   ├── load_external.py         # loads external HPA data
│   └── make_external_csv.py     # builds labels CSV for external data
|
├── checkpoints/                 # saved model checkpoints (.pt)
│   └── thresholds/              # per-class threshold tensors (.pt)
├── submissions/
│   └── probs/                   # raw probability tensors (.pt)
│
├── data_review.ipynb            # class distribution, sample visualization
└── evaluation.ipynb             # metric analysis
```

## Setup

```bash
pip install requirements.txt
```

## Pipeline

### 0. Prepare images

**Download official data**

```bash
kaggle competitions download -c human-protein-atlas-image-classification -f train.zip
kaggle competitions download -c human-protein-atlas-image-classification -f test.zip
mkdir -p data/raw/train data/raw/test
unzip train.zip -d data/raw/train
unzip test.zip -d data/raw/test
```

**Download external images**

```bash
python tools/load_external.py
# → data/raw/external/
```

**Convert to 4-channel RGBY**

```bash
python tools/convert_to_rgby.py --input_dir data/raw/train --output_dir data/train
python tools/convert_to_rgby.py --input_dir data/raw/test --output_dir data/test
python tools/convert_to_rgby.py --input_dir data/raw/external --output_dir data/external
```

After conversion `data/raw/` can be removed. The expected data layout:

```
data/
├── train/                 # RGBY .png train images
├── test/                  # RGBY .png test images
├── external/              # RGBY .png external images
├── train.csv
├── external.csv
└── sample_submission.csv
```

### 1. Preprocess data

```bash
python create_dataset.py
```

Removes dublications, applies stratified cell-line-aware split, and saves datasets to `data/saved_datasets/`.

### 2. Train

```bash
# fresh run
python train.py --name model_name

# resume from checkpoint
python train.py --name model_name --checkpoint model_name
```

### 3. Find optimal thresholds

```bash
python find_optimal_thresholds.py --checkpoint model_name
```

Runs per-class threshold search on the validation set. Saves to `checkpoints/thresholds/<name>.pt`.

### 4. Run inference

```bash
python inference.py --checkpoint model_name
```

Saves raw probability tensor to `submissions/probs/<name>.pt`.

### 5. Make submission

```bash
# with optimized thresholds
python make_submission.py --name model_name # use thresholds and probs of a model (if exist)
```

## Key Design Decisions

**Model** — EfficientNetV2-S pretrained on ImageNet, adapted for 4-channel RGBY input. The first conv layer is expanded from 3→4 channels: RGB weights are copied from the pretrained model, the 4th channel (yellow/ER) is initialized with the mean of the RGB weights. Differential learning rates are used: lower LR for the backbone (`3e-4`), higher for the classifier head (`1e-3`).

**Class imbalance** — 28 classes with extreme imbalance (pos/neg up to 1:2660). Handled via Focal Loss (γ=2.2) with per-class alpha weights derived from inverse label frequency. No random oversampler was used.

**Data leakage** — cell lines appear across train and external splits. DSU-based leakage detection ensures no cell line overlaps the validation set.

**Threshold optimization** — global threshold (0.5) is used during training for monitoring. Per-class thresholds are calibrated on the validation set using a prevalence-matching strategy: for each class, the threshold is set to the `(1 − prevalence)`-th quantile of predicted probabilities, so the fraction of positive predictions matches the true class frequency. Inspired by [@pudae81](https://www.kaggle.com/pudae81)'s 3rd place solution.

## References

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — [arXiv](https://arxiv.org/abs/1708.02002)
- Ridnik et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021 — [arXiv](https://arxiv.org/abs/2009.14119)
- [@pudae81](https://www.kaggle.com/pudae81), *3rd place solution* — threshold calibration strategy and external data tooling — [GitHub](https://github.com/pudae/kaggle-hpa/tree/master/tools)
- [Human Protein Atlas Kaggle Competition](https://www.kaggle.com/c/human-protein-atlas-image-classification)