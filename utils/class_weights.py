import torch
from config import PathConfig


# --- funcs ---

def calculate_focal_alphas(train_labels: list[torch.Tensor]) -> torch.Tensor:
    eps = 1e-6
    train_labels = torch.stack(train_labels)
    pos_counts = train_labels.sum(dim=0)
    alpha = 1.0 / (pos_counts + eps)
    alpha = alpha / alpha.sum() * len(pos_counts)
    return alpha


def calculate_pos_weight(train_labels: list[torch.Tensor]) -> torch.Tensor:
    eps = 1e-6
    train_labels = torch.stack(train_labels)
    pos_counts = train_labels.sum(dim=0)
    neg_counts = train_labels.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + eps)
    return pos_weight


def calculate_sample_weights(train_labels: list[torch.Tensor],
                             rare_threshold: float = 100.0) -> torch.Tensor:
    labels = torch.stack(train_labels)
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    rare_mask = (pos_weight > rare_threshold).float()
    class_boost = 1.0 + rare_mask * torch.log1p(pos_weight / rare_threshold)

    active_boost = labels * class_boost
    active_counts = labels.sum(dim=1).clamp(min=1)
    sample_weights = active_boost.sum(dim=1) / active_counts

    return sample_weights.clamp(min=1.0)



# main section
if __name__ == "__main__":
    from dataset import load_dataset
    train_ds = load_dataset(PathConfig.train_ds_path)

    alpha_weights = calculate_focal_alphas(train_ds.labels)
    alpha_weights = [round(w, 5) for w in alpha_weights.tolist()]
    print(f"\nAlpha weights:\n{alpha_weights}")

    pos_weights = calculate_pos_weight(train_ds.labels)
    pos_weights = [round(w, 5) for w in pos_weights.tolist()]
    print(f"\nPos weights:\n{pos_weights}")

    sampler_weights = calculate_sample_weights(train_ds.labels)
    sampler_weights = [round(w, 5) for w in sampler_weights.tolist()]
    print(f"\nSampler weights:\n{sampler_weights[:100]}")