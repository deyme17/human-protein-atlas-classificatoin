import torch
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_thresholds, load_probs

from config import PathConfig, DataConfig



def make_submission(probs: torch.Tensor, image_ids: list[str],
                    thresholds: torch.Tensor | float,
                    sample_submission: pd.DataFrame) -> pd.DataFrame:
    assert len(probs) == len(image_ids), "probs and image_ids length mismatch"

    if isinstance(thresholds, float):
        thresholds = torch.full((DataConfig.n_classes,), thresholds)
    assert len(thresholds) == DataConfig.n_classes, "thresholds length != n_classes"

    preds = (probs >= thresholds).int().numpy()
    predictions = {}
    for img_id, pred in zip(image_ids, preds):
        labels = np.where(pred == 1)[0]
        predictions[img_id] = " ".join(map(str, labels)) if len(labels) > 0 else ""

    submission = sample_submission.copy()
    submission["Predicted"] = submission["Id"].map(predictions)

    assert submission["Predicted"].isna().sum() == 0, "Missing predictions!"
    n_empty = (submission["Predicted"] == "").sum()
    print(f"Empty predictions: {n_empty} / {len(submission)}")
    return submission



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build submission from saved probs.")
    parser.add_argument("--name", type=str, required=True, help="Name of probs and thresholds .pt files")
    parser.add_argument("--threshold", type=float, default=0.3, help="Single threshold (used if thresholds not found)")
    parser.add_argument("--out", type=str, default=None, help="Output filename (default: <probs_stem>.csv)")
    args = parser.parse_args()
    name = args.name

    # load probs
    state = load_probs(name)
    probs = state["probs"]
    image_ids = state["image_ids"]
    print(f"Loaded probs {tuple(probs.shape)} for {len(image_ids)} images.")

    # load thresholds
    try:
        thresholds = load_thresholds(name)
        print(f"Loaded per-class thresholds '{name}'.")
    except:
        thresholds = args.threshold
        print(f"Using single threshold={thresholds}.")

    #  build and save submission
    sample_submission = pd.read_csv(PathConfig.sample_submission_path)
    submission = make_submission(probs, image_ids, thresholds, sample_submission)

    out_name = args.out or f"{Path(name).stem}.csv"
    out_path = PathConfig.submission_dir / out_name
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to '{out_path}'.")