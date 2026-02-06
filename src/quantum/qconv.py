"""Parameterized quantum convolution blocks used by QCNN layers."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def qconv_block(name: str = "QConv"):
    """Build the two-qubit variational convolution block.

    Parameters
    ----------
    name : str, default="QConv"
        Prefix used for circuit and parameter naming.

    Returns
    -------
    tuple[qiskit.QuantumCircuit, qiskit.circuit.ParameterVector]
        Parametric circuit and parameter vector of length 2.
    """
    theta = ParameterVector(f"{name}_theta", 2)

    qc = QuantumCircuit(2, name=name)
    qc.h([0, 1])
    qc.ry(theta[0], 0)
    qc.cx(0, 1)
    qc.ry(theta[1], 1)
    return qc, theta


def qconv_block3(name: str = "QConv3"):
    """Build the three-qubit variational convolution block.

    Parameters
    ----------
    name : str, default="QConv3"
        Prefix used for circuit and parameter naming.

    Returns
    -------
    tuple[qiskit.QuantumCircuit, qiskit.circuit.ParameterVector]
        Parametric circuit and parameter vector of length 3.
    """
    theta = ParameterVector(f"{name}_theta", 3)

    qc = QuantumCircuit(3, name=name)
    qc.h([0, 1, 2])
    qc.ry(theta[0], 0)
    qc.cx(0, 1)
    qc.ry(theta[1], 1)
    qc.cx(1, 2)
    qc.ry(theta[2], 2)
    return qc, theta
