from __future__ import annotations

import pytest
import torch

from src.quantum.encoder import image_to_patches, patch_grid_size


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 7, 14])
def test_patch_grid_size_matches_unfold(stride: int) -> None:
    image = torch.randn(1, 30, 30, dtype=torch.float32)
    patches = image_to_patches(image, stride=stride)
    n_steps = patch_grid_size(stride=stride)
    assert patches.shape == (n_steps * n_steps, 9)
