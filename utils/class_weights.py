import torch
from config import PathConfig
from dataset import load_dataset


# --- funcs ---

def calculate_focal_alphas(train_labels: list) -> list:
    eps = 1e-6
    train_labels = torch.stack(train_labels)
    pos_counts = train_labels.sum(dim=0)
    alpha = 1.0 / (pos_counts + eps)
    alpha = alpha / alpha.sum() * len(pos_counts)
    return alpha.tolist()

def calculate_pos_weight(train_labels: list) -> list:
    eps = 1e-6
    train_labels = torch.stack(train_labels)
    pos_counts = train_labels.sum(dim=0)
    neg_counts = train_labels.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + eps)
    pos_weight = torch.clamp(pos_weight, max=100.0)
    return pos_weight.tolist()


# main section
if __name__ == "__main__":
    train_ds = load_dataset(PathConfig.train_ds_path)

    alpha_weights = calculate_focal_alphas(train_ds.labels)
    alpha_weights = [round(w, 5) for w in alpha_weights]
    print(f"\nAlpha weights:\n{alpha_weights}")

    pos_weights = calculate_pos_weight(train_ds.labels)
    pos_weights = [round(w, 5) for w in pos_weights]
    print(f"\nPos weights:\n{pos_weights}")