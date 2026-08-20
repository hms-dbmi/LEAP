from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


class E2E_MILModel(nn.Module):
    """A patch feature extractor and an attention-MIL head trained as one network.

    forward() takes a batch of bags, (B, N_PATCHES, C, H, W), and returns slide-level
    logits, (B, OUT_FEATURES).
    """

    def __init__(self, feature_extractor: nn.Module, mil_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mil_head = mil_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches = x.size(0), x.size(1)
        x = x.view(batch_size * num_patches, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_patches, -1)
        return self.mil_head(features)


def build_model(cfg: DictConfig, device: Optional[str] = None) -> E2E_MILModel:
    """Instantiate the extractor and head named by cfg.extractor / cfg.head and pair them."""
    model = E2E_MILModel(instantiate(cfg.extractor), instantiate(cfg.head))
    return model.to(device) if device is not None else model


def load_model(cfg: DictConfig, checkpoint: str, device: str) -> E2E_MILModel:
    """Build the model from cfg and load a state dict, tolerating a 'state_dict' wrapper."""
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def slide_embedding(model: E2E_MILModel, patches: torch.Tensor) -> np.ndarray:
    """The slide vector a survival head is fitted on.

    Runs one slide's patches, (N_PATCHES, C, H, W) on the model's device, through the extractor
    and the attention pooling, then replays every layer of the head MLP except the final one.
    Returns the last hidden layer's activations, shape (D_MLP,).
    """
    head = model.mil_head
    bag = model.feature_extractor(patches).unsqueeze(0)
    pooled, _weights = head.attention_layer(head.tiles_emb(bag[..., head.metadata_cols:], None), None)

    hidden = pooled
    for layer in list(head.mlp.children())[:-1]:
        hidden = layer(hidden)
    return hidden.reshape(-1).float().cpu().numpy()
