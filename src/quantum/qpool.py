"""Quantum pooling blocks based on measurement-conditioned correction."""

from qiskit import QuantumCircuit


def qpool_block() -> QuantumCircuit:
    """Build a two-qubit pooling circuit with one classical bit.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit that measures qubit 1 and conditionally applies ``X`` on qubit 0
        when the classical outcome is 1.
    """
    qc = QuantumCircuit(2, 1, name="QPool")
    qc.measure(1, 0)
    qc.x(0).c_if(qc.cregs[0], 1)
    return qc


def qpool_block3() -> QuantumCircuit:
    """Build a three-qubit pooling circuit with two classical bits.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit that measures qubits 1 and 2 and conditionally applies ``X`` on qubit 0
        when at least one measured classical bit is 1.
    """
    qc = QuantumCircuit(3, 2, name="QPool3")
    qc.measure(1, 0)
    qc.measure(2, 1)
    creg = qc.cregs[0]
    qc.x(0).c_if(creg, 1)
    qc.x(0).c_if(creg, 2)
    qc.x(0).c_if(creg, 3)
    return qc
