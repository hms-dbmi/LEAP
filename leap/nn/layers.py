"""Attention-based MIL building blocks.

Adapted from Owkin's HistoSSLscaling (https://github.com/owkin/HistoSSLscaling), whose licence
terms govern this file. MaskedLinear additionally derives from HuggingFace Transformers
(https://github.com/huggingface/transformers).
"""
from typing import List, Optional, Tuple, Union

import torch


class MaskedLinear(torch.nn.Linear):
    """Linear layer applied patch-wise, filling masked positions with `mask_value`.

    Used so padded patches cannot influence a subsequent activation: with mask_value='-inf'
    a softmax over the patch axis assigns them exactly zero weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """x: (B, N_PATCHES, IN_FEATURES); mask: (B, N_PATCHES, 1), True where padded.

        Returns (B, N_PATCHES, OUT_FEATURES).
        """
        x = super().forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )


class TilesMLP(torch.nn.Module):
    """MLP applied independently to every patch, mask-aware in its hidden layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """x: (B, N_PATCHES, IN_FEATURES) -> (B, N_PATCHES, OUT_FEATURES)."""
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x


class MLP(torch.nn.Sequential):
    """Plain MLP with optional per-hidden-layer dropout.

    `hidden` and `dropout`, when both given, must have the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is None:
                raise ValueError("hidden must be given, and match dropout in length, if dropout is given.")
            if len(hidden) != len(dropout):
                raise ValueError("hidden and dropout must have the same length")

        d_model = in_features
        layers = []
        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h
                if activation is not None:
                    seq.append(activation)
                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))
                layers.append(torch.nn.Sequential(*seq))
        layers.append(torch.nn.Linear(d_model, out_features))

        super().__init__(*layers)


class GatedAttention(torch.nn.Module):
    """Gated attention pooling over the patch axis (Ilse et al., 2018, arXiv:1802.04712)."""

    def __init__(self, d_model: int = 128, temperature: float = 1.0):
        super().__init__()
        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")
        self.temperature = temperature

    def attention(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """v: (B, N_PATCHES, D) -> attention logits (B, N_PATCHES, 1)."""
        h_v = torch.tanh(self.att(v))
        u_v = torch.sigmoid(self.gate(v))
        return self.w(h_v * u_v, mask=mask) / self.temperature

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """v: (B, N_PATCHES, D) -> (pooled (B, D), attention weights (B, N_PATCHES, 1)).

        Weights are a softmax over the patch axis and sum to 1.
        """
        attention_weights = torch.softmax(self.attention(v=v, mask=mask), 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)
        return scaled_attention.squeeze(1), attention_weights
