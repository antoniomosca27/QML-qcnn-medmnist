"""Patch encoding utilities for quantum convolution inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from qiskit import QuantumCircuit


def patch_grid_size(image_size: int = 30, patch_size: int = 3, stride: int = 3) -> int:
    """Return the number of patch positions along one image dimension.

    Parameters
    ----------
    image_size : int, default=30
        Side length of the square image.
    patch_size : int, default=3
        Side length of the square patch.
    stride : int, default=3
        Sliding-window stride.

    Returns
    -------
    int
        Number of extraction steps along a single axis.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1.")
    if image_size < patch_size:
        raise ValueError("image_size must be >= patch_size.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    return (image_size - patch_size) // stride + 1


def patch_to_2qubits(patch: torch.Tensor) -> "QuantumCircuit":
    """Encode a 2x2 patch into a two-qubit circuit with angle encoding.

    Parameters
    ----------
    patch : torch.Tensor
        Tensor of shape ``(4,)`` representing flattened patch values.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit with two qubits.
    """
    if patch.numel() != 4:
        raise ValueError("patch must contain exactly 4 values for 2x2 encoding.")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, name="AngleEnc2")
    qc.ry(float(patch[0]) * torch.pi, 0)
    qc.ry(float(patch[1]) * torch.pi, 1)
    qc.cx(0, 1)
    qc.ry(float(patch[2]) * torch.pi, 0)
    qc.ry(float(patch[3]) * torch.pi, 1)
    return qc


def patch_to_3qubits(patch: torch.Tensor) -> "QuantumCircuit":
    """Encode a 3x3 patch into a three-qubit circuit with angle encoding.

    Parameters
    ----------
    patch : torch.Tensor
        Tensor of shape ``(9,)`` representing flattened patch values.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit with three qubits.
    """
    if patch.numel() != 9:
        raise ValueError("patch must contain exactly 9 values for 3x3 encoding.")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, name="AngleEnc3")
    for qubit in range(3):
        qc.ry(float(patch[qubit]) * torch.pi, qubit)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for qubit in range(3):
        qc.ry(float(patch[qubit + 3]) * torch.pi, qubit)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for qubit in range(3):
        qc.ry(float(patch[qubit + 6]) * torch.pi, qubit)
    return qc


def image_to_patches(img: torch.Tensor, *, stride: int = 3) -> torch.Tensor:
    """Extract flattened 3x3 patches from a ``(1, 30, 30)`` image tensor.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape ``(1, 30, 30)``.
    stride : int, default=3
        Patch extraction stride.

    Returns
    -------
    torch.Tensor
        Flattened patches with shape ``(num_patches, 9)``.
    """
    if img.shape != (1, 30, 30):
        raise ValueError("img must have shape (1, 30, 30).")
    patch_grid_size(stride=stride)

    return img.unfold(1, 3, stride).unfold(2, 3, stride).contiguous().view(-1, 9)
