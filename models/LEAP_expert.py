import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

class LEAP_Expert(nn.Module):
    def __init__(self, feature_extractor, mil_head):
        super(LEAP_Expert, self).__init__()
        self.feature_extractor = feature_extractor
        self.mil_head = mil_head

    def forward(self, x):
        batch_size = x.size(0)
        num_tiles = x.size(1)
        x = x.view(batch_size * num_tiles, x.size(2), x.size(3), x.size(4))  # Flatten the batch and tile dimensions (VGG wants batch_size*num_tiles, channel, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_tiles, -1)  # Reshape back to (batch_size, num_tiles, feature_dim)
        logits = self.mil_head(features)
        return logits
    
def build_LEAP_pipeline(cfg: DictConfig) -> LEAP_Expert:
    # Instantiate the  feature extractor and  head using the configuration file:
    extractor = instantiate(cfg.extractor)
    head = instantiate(cfg.head)
    model = LEAP_Expert(extractor, head)
    return model