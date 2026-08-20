"""Attention-based MIL head.

Adapted from Owkin's HistoSSLscaling (https://github.com/owkin/HistoSSLscaling), whose licence
terms govern this file. The MIL method is Ilse, Tomczak and Welling, "Attention-based Deep
Multiple Instance Learning", ICML 2018 (arXiv:1802.04712).
"""
from typing import List, Optional

import torch
from torch import nn

from leap.nn.layers import MLP, GatedAttention, TilesMLP


class ABMIL(nn.Module):
    """Attention-based MIL classifier (Ilse et al., 2018).

    Patch features are projected by a patch-wise MLP, pooled by gated attention into one
    slide vector, then mapped to `out_features` logits by a final MLP.

    Parameters
    ----------
    in_features: patch feature dimension (must match the backbone output).
    out_features: number of output logits; 1 for binary classification.
    d_model_attention: dimension the attention layer operates in.
    temperature: softmax temperature of the attention layer.
    tiles_mlp_hidden: hidden sizes of the patch-wise projection MLP.
    mlp_hidden: hidden sizes of the final MLP.
    mlp_dropout: dropout per final-MLP hidden layer; same length as mlp_hidden.
    mlp_activation: activation of the final MLP.
    bias: add bias to the patch-wise projection MLP.
    metadata_cols: number of leading feature columns to drop as metadata before projection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 0,
    ) -> None:
        super().__init__()

        if mlp_dropout is not None:
            if mlp_hidden is None:
                raise ValueError(
                    "mlp_hidden must be given, and match mlp_dropout in length, "
                    "if mlp_dropout is given."
                )
            if len(mlp_hidden) != len(mlp_dropout):
                raise ValueError("mlp_hidden and mlp_dropout must have the same length")

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )
        self.attention_layer = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )
        self.mlp = MLP(
            in_features=d_model_attention,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.metadata_cols = metadata_cols

    def score_model(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """x: (B, N_PATCHES, FEATURES) -> attention logits (B, N_PATCHES, 1)."""
        return self.attention_layer.attention(self.tiles_emb(x, mask), mask)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """features: (B, N_PATCHES, metadata_cols + FEATURES) -> logits (B, OUT_FEATURES)."""
        tiles_emb = self.tiles_emb(features[..., self.metadata_cols:], mask)
        scaled_tiles_emb, _ = self.attention_layer(tiles_emb, mask)
        return self.mlp(scaled_tiles_emb)
