# human-protein-atlas-classification

Human Protein Atlas (HPA) image classification pipeline with focus on class imbalance mitigation, data leakage prevention, and robust multi-label training on 4-channel microscopy images.

## Overview

Multi-label classification of 28 subcellular protein localization patterns from 4-channel fluorescence microscopy images (RGBY: red, green, blue, yellow). The pipeline handles extreme class imbalance (pos/neg ratio up to 1:2600) and prevents data leakage via cell-line-aware stratified splitting.

## Project Structure

```
human-protein-atlas-classification/
├── config.py                    # PathConfig, DataConfig, TrainConfig
├── train.py                     # training entry point
├── inference.py                 # saves probs tensor to disk
├── find_optimal_thresholds.py   # per-class threshold optimization on validation set
├── make_submission.py           # builds submission.csv from probs + thresholds
├── create_dataset.py            # preprocessing → saves .pth datasets
│
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
├── models/
│   └── model_factory.py         # get_model()
|
├── losses/
│   └── losses_factory.py        # get_loss() Focal loss / ASL (Ridnik et al., ICCV 2021)
|
├── transforms/                  # get_transforms()
├── optimizers/                  # get_optimizer(), Adam / AdamW / SGD
├── schedulers/                  # get_scheduler(), Cosine / StepLR / ReduceLROnPlateau
│
├── tools/
│   ├── convert_to_rgby.py       # converts raw images to 4-channel .png
│   ├── load_external.py         # loads external HPA data
│   ├── preprocess_to_npy.py     # convert png images to numpy arrays
│   └── make_external_csv.py     # builds labels CSV for external data
│
├── utils/
│
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

**Preprocess to .npy cache**

```bash
python tools/preprocess_to_npy.py
# reads separate _red/_green/_blue/_yellow PNGs from data/raw/
# → data/npy_cache/
```

`data/raw/` can be removed after preprocessing. The expected data layout:

```
data/
├── npy_cache/             # pre-resized RGBA .npy images
│   └── valid_ids.csv      # IDs that were successfully processed
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
python make_submission.py \
    --probs submissions/probs/model_name.pt \
    --thresholds checkpoints/thresholds/model_name.pt
```

## Key Design Decisions

**Class imbalance** — 28 classes with extreme imbalance (pos/neg up to 1:2660). Handled via Asymmetric Loss (ASL) as the primary mechanism, with optional mild `WeightedRandomSampler` (log-scaled, mean-pooled) for coverage of the rarest classes.

**Data leakage** — cell lines appear across train and external splits. DSU-based leakage detection ensures no cell line overlaps the validation set.

**Threshold optimization** — global threshold (0.3) is used during training for monitoring. Per-class thresholds are optimized post-hoc on the validation set via grid search.

## References

- Ridnik et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021 — [arXiv](https://arxiv.org/abs/2009.14119)
- [Human Protein Atlas Kaggle Competition](https://www.kaggle.com/c/human-protein-atlas-image-classification)