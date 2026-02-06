"""Trainable RGB-to-grayscale projection used during preprocessing."""

from __future__ import annotations

import torch
from torch import nn


class Color2GrayNet(nn.Module):
    """A trainable 1x1 convolution mapping RGB images to one channel.

    Parameters
    ----------
    init : {"random", "luma"}, default="random"
        Weight initialization strategy.
        - ``"random"`` initializes with Xavier uniform weights.
        - ``"luma"`` initializes with standard luminance coefficients
          ``[0.299, 0.587, 0.114]``.
    """

    def __init__(self, init: str = "random") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)

        if init == "luma":
            luma = torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32)
            self.conv.weight.data.copy_(luma.view(1, 3, 1, 1))
        elif init == "random":
            nn.init.xavier_uniform_(self.conv.weight)
        else:
            raise ValueError(f"Unsupported initialization mode: {init!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grayscale projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 3, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Grayscale tensor of shape ``(batch, 1, 28, 28)``.
        """
        return self.conv(x)
