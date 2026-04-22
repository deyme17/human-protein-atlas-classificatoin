import torchvision.transforms as T
from .channel_dropout import ChannelDropout
from .normalize import PerImageNormalize
from config import DataConfig


def get_transforms(is_train: bool = False):
    if is_train:
        return T.Compose([
            # T.Resize((DataConfig.input_dim, DataConfig.input_dim)), dont need this if npy chache are used
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            PerImageNormalize(),
            # ChannelDropout(p=0.2, drop_channels=(0, 2, 3)) # the channel 1 (green) remains
        ])
    else:
        return T.Compose([
            # T.Resize((DataConfig.input_dim, DataConfig.input_dim)),
            PerImageNormalize(),
        ])