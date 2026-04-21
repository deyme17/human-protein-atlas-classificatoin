from utils.class_weights import calculate_sample_weights
from torch.utils.data import WeightedRandomSampler


def get_sampler(labels) -> WeightedRandomSampler:
    sample_weights = calculate_sample_weights(labels)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )