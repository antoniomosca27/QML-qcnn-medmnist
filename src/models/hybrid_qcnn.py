"""Hybrid QCNN model combining quantum patch features and a linear head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers import QuantumConvLayer3
from src.quantum.encoder import image_to_patches, patch_grid_size


class HybridQCNN(nn.Module):
    """Hybrid model with shared quantum convolution and linear classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    stride : int, default=3
        Patch extraction stride for 3x3 patches from ``30x30`` images.
    """

    def __init__(self, n_classes: int, stride: int = 3) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        self.stride = stride
        self.n_steps = patch_grid_size(stride=stride)
        self.n_patch = self.n_steps * self.n_steps
        self.n_patches = self.n_patch  # Backward-compatible alias.
        self.qconv = QuantumConvLayer3()
        self.fc = nn.Linear(self.n_patch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of grayscale images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, 30, 30)``.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape ``(batch, n_classes)``.
        """
        batch_size = x.size(0)
        patch_batches = [image_to_patches(img, stride=self.stride) for img in x]
        patches = torch.vstack(patch_batches)

        features = self.qconv(patches)
        features = features.view(batch_size, self.n_steps, self.n_steps)
        features = features.view(batch_size, self.n_patch)
        return self.fc(features)

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, 30, 30)``.
        y : torch.Tensor
            Labels of shape ``(batch,)`` or ``(batch, 1)``.
        logits : torch.Tensor | None, optional
            Precomputed logits with shape ``(batch, n_classes)``. If omitted,
            logits are computed from ``x``.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if logits is None:
            logits = self.forward(x)
        labels = y.view(-1).long()
        return F.cross_entropy(logits, labels)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for :meth:`compute_loss`."""
        return self.compute_loss(x, y, logits=logits)
