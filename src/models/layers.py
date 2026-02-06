"""PyTorch-to-Qiskit bridge for batched quantum convolution layers."""

from __future__ import annotations

import concurrent.futures
import math
import os
from multiprocessing import cpu_count

import torch
from qiskit.quantum_info import SparsePauliOp, Statevector
from torch import nn

from src.quantum.encoder import patch_to_2qubits, patch_to_3qubits
from src.quantum.qconv import qconv_block, qconv_block3

_Z = SparsePauliOp.from_list([("Z", 1)])
_CPUS_ENV = "QCNN_CPUS"


def _resolve_max_workers() -> int:
    """Resolve process pool width from ``QCNN_CPUS`` or CPU count."""
    raw = os.getenv(_CPUS_ENV)
    if raw is None:
        return cpu_count()
    try:
        value = int(raw)
    except ValueError:
        return cpu_count()
    return max(1, value)


def _simulate_patch(
    patch_np: list[float], theta0: float, theta1: float
) -> tuple[float, float, float]:
    """Evaluate output and parameter-shift gradients for one 2x2 patch.

    Parameters
    ----------
    patch_np : list[float]
        Flattened 2x2 patch values.
    theta0 : float
        First variational parameter.
    theta1 : float
        Second variational parameter.

    Returns
    -------
    tuple[float, float, float]
        ``(f, df_dtheta0, df_dtheta1)`` for expectation value ``<Z0>``.
    """
    patch = torch.tensor(patch_np, dtype=torch.float32)

    def _expect(t0: float, t1: float) -> float:
        circuit = patch_to_2qubits(patch)
        block, params = qconv_block()
        circuit.compose(block.assign_parameters({params[0]: t0, params[1]: t1}), inplace=True)
        state = Statevector.from_instruction(circuit)
        return float(state.expectation_value(_Z, qargs=[0]).real)

    center = _expect(theta0, theta1)
    shift = math.pi / 2.0

    grad_theta0 = 0.5 * (_expect(theta0 + shift, theta1) - _expect(theta0 - shift, theta1))
    grad_theta1 = 0.5 * (_expect(theta0, theta1 + shift) - _expect(theta0, theta1 - shift))
    return center, grad_theta0, grad_theta1


def _simulate_patch3(
    patch_np: list[float],
    theta0: float,
    theta1: float,
    theta2: float,
) -> tuple[float, float, float, float]:
    """Evaluate output and parameter-shift gradients for one 3x3 patch.

    Parameters
    ----------
    patch_np : list[float]
        Flattened 3x3 patch values.
    theta0 : float
        First variational parameter.
    theta1 : float
        Second variational parameter.
    theta2 : float
        Third variational parameter.

    Returns
    -------
    tuple[float, float, float, float]
        ``(f, df_dtheta0, df_dtheta1, df_dtheta2)`` for expectation value ``<Z0>``.
    """
    patch = torch.tensor(patch_np, dtype=torch.float32)

    def _expect(t0: float, t1: float, t2: float) -> float:
        circuit = patch_to_3qubits(patch)
        block, params = qconv_block3()
        circuit.compose(
            block.assign_parameters({params[0]: t0, params[1]: t1, params[2]: t2}),
            inplace=True,
        )
        state = Statevector.from_instruction(circuit)
        return float(state.expectation_value(_Z, qargs=[0]).real)

    center = _expect(theta0, theta1, theta2)
    shift = math.pi / 2.0

    grad_theta0 = 0.5 * (
        _expect(theta0 + shift, theta1, theta2) - _expect(theta0 - shift, theta1, theta2)
    )
    grad_theta1 = 0.5 * (
        _expect(theta0, theta1 + shift, theta2) - _expect(theta0, theta1 - shift, theta2)
    )
    grad_theta2 = 0.5 * (
        _expect(theta0, theta1, theta2 + shift) - _expect(theta0, theta1, theta2 - shift)
    )
    return center, grad_theta0, grad_theta1, grad_theta2


class _BatchQuantumConv(torch.autograd.Function):
    """Autograd function for batched two-parameter quantum convolution."""

    @staticmethod
    def forward(ctx, patches: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta0, theta1 = map(float, theta.detach().cpu())
        patches_np = patches.detach().cpu().numpy().tolist()

        with concurrent.futures.ProcessPoolExecutor(max_workers=_resolve_max_workers()) as executor:
            results = list(
                executor.map(
                    _simulate_patch,
                    patches_np,
                    [theta0] * len(patches_np),
                    [theta1] * len(patches_np),
                )
            )

        outputs = torch.tensor(
            [row[0] for row in results], dtype=patches.dtype, device=patches.device
        )
        grad_theta0 = torch.tensor([row[1] for row in results], dtype=patches.dtype)
        grad_theta1 = torch.tensor([row[2] for row in results], dtype=patches.dtype)
        ctx.save_for_backward(grad_theta0, grad_theta1)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_theta0, grad_theta1 = ctx.saved_tensors
        grad_out_cpu = grad_output.detach().cpu()

        dtheta0 = torch.dot(grad_out_cpu, grad_theta0)
        dtheta1 = torch.dot(grad_out_cpu, grad_theta1)

        grad_patches = None
        grad_theta = torch.stack([dtheta0, dtheta1]).to(grad_output.device)
        return grad_patches, grad_theta


class _BatchQuantumConv3(torch.autograd.Function):
    """Autograd function for batched three-parameter quantum convolution."""

    @staticmethod
    def forward(ctx, patches: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        theta0, theta1, theta2 = map(float, theta.detach().cpu())
        patches_np = patches.detach().cpu().numpy().tolist()

        with concurrent.futures.ProcessPoolExecutor(max_workers=_resolve_max_workers()) as executor:
            results = list(
                executor.map(
                    _simulate_patch3,
                    patches_np,
                    [theta0] * len(patches_np),
                    [theta1] * len(patches_np),
                    [theta2] * len(patches_np),
                )
            )

        outputs = torch.tensor(
            [row[0] for row in results], dtype=patches.dtype, device=patches.device
        )
        grad_theta0 = torch.tensor([row[1] for row in results], dtype=patches.dtype)
        grad_theta1 = torch.tensor([row[2] for row in results], dtype=patches.dtype)
        grad_theta2 = torch.tensor([row[3] for row in results], dtype=patches.dtype)
        ctx.save_for_backward(grad_theta0, grad_theta1, grad_theta2)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_theta0, grad_theta1, grad_theta2 = ctx.saved_tensors
        grad_out_cpu = grad_output.detach().cpu()

        dtheta0 = torch.dot(grad_out_cpu, grad_theta0)
        dtheta1 = torch.dot(grad_out_cpu, grad_theta1)
        dtheta2 = torch.dot(grad_out_cpu, grad_theta2)

        grad_patches = None
        grad_theta = torch.stack([dtheta0, dtheta1, dtheta2]).to(grad_output.device)
        return grad_patches, grad_theta


class QuantumConvLayer(nn.Module):
    """Two-parameter quantum convolution layer.

    Notes
    -----
    Input patches are expected with shape ``(num_patches, 4)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(2 * math.pi * torch.rand(2))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Evaluate quantum convolution on a batch of 2x2 patches."""
        return _BatchQuantumConv.apply(patches, self.theta)


class QuantumConvLayer3(nn.Module):
    """Three-parameter quantum convolution layer.

    Notes
    -----
    Input patches are expected with shape ``(num_patches, 9)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(2 * math.pi * torch.rand(3))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Evaluate quantum convolution on a batch of 3x3 patches."""
        return _BatchQuantumConv3.apply(patches, self.theta)
