import torch


class ChannelDropout:
    def __init__(self, p: float = 0.1, drop_channels: tuple[int] = (0, 2, 3)):
        self.p = p
        self.drop_channels = drop_channels

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            channel_idx = torch.randint(0, len(self.drop_channels), (1,)).item()
            channel = self.drop_channels[channel_idx]
            img = img.clone()
            img[channel].zero_()
        return img