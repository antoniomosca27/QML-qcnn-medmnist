from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.quantum.encoder import image_to_patches, patch_grid_size
from src.training import trainer as trainer_module
from src.training.evaluate import classification_metrics
from src.utils.seed import set_global_seed

QISKIT_AVAILABLE = importlib.util.find_spec("qiskit") is not None


def test_image_to_patches_shapes() -> None:
    img = torch.zeros(1, 30, 30, dtype=torch.float32)

    patches_stride_3 = image_to_patches(img, stride=3)
    assert patches_stride_3.shape == (100, 9)

    patches_stride_2 = image_to_patches(img, stride=2)
    assert patches_stride_2.shape == (196, 9)

    patches_stride_4 = image_to_patches(img, stride=4)
    assert patches_stride_4.shape == (49, 9)

    assert patch_grid_size(stride=4) == 7
    with pytest.raises(ValueError, match=">= 1"):
        image_to_patches(img, stride=0)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="qiskit is not installed")
def test_qconv_block_qubits_and_parameter_counts() -> None:
    from src.quantum.qconv import qconv_block, qconv_block3

    circuit2, theta2 = qconv_block()
    circuit3, theta3 = qconv_block3()

    assert circuit2.num_qubits == 2
    assert circuit3.num_qubits == 3
    assert len(theta2) == 2
    assert len(theta3) == 3


def test_classification_metrics_auc_numeric() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.5, 0.1],
            [3.5, 0.4, 0.2],
            [0.3, 3.8, 0.2],
            [0.4, 3.2, 0.6],
            [0.2, 0.5, 3.7],
            [0.1, 0.4, 4.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    metrics = classification_metrics(logits, labels)

    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["auc_micro"] == pytest.approx(1.0)
    assert metrics["auc_macro"] == pytest.approx(1.0)


def test_set_global_seed_no_crash() -> None:
    set_global_seed(123)


def test_trainer_run_epoch_single_forward_per_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyHybridQCNN(nn.Module):
        def __init__(self, n_classes: int, stride: int) -> None:
            super().__init__()
            self.linear = nn.Linear(4, n_classes)
            self.forward_calls = 0
            self.qconv = nn.Linear(1, 1)  # shape compatibility for freeze_q path

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            flat = x.view(x.size(0), -1)
            return self.linear(flat)

    def fake_load_split(path: Path, batch: int, shuffle: bool) -> DataLoader:
        x = torch.randn(12, 1, 2, 2)
        y = torch.randint(0, 2, (12,))
        return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=shuffle)

    monkeypatch.setattr(trainer_module, "HybridQCNN", DummyHybridQCNN)
    monkeypatch.setattr(trainer_module, "_load_split", fake_load_split)

    split_root = tmp_path / "data" / "processed" / "dummy"
    split_root.mkdir(parents=True, exist_ok=True)
    dummy_payload = {"x": torch.zeros(1, 1, 2, 2), "y": torch.zeros(1, dtype=torch.long)}
    for split_name in ("train", "val", "test"):
        torch.save(dummy_payload, split_root / f"{split_name}.pt")

    trainer = trainer_module.Trainer(
        dataset_name="dummy",
        out_dir=tmp_path / "logs" / "dummy_run_001",
        data_dir=tmp_path / "data",
        batch_size=3,
        epochs=1,
    )

    expected_train_batches = len(trainer.train_ld)
    trainer._run_epoch(trainer.train_ld, train=True)
    assert trainer.model.forward_calls == expected_train_batches


@pytest.mark.parametrize(
    ("module_name", "program_name"),
    [
        ("src.scripts.preprocess", "qcnn-preprocess"),
        ("src.scripts.train", "qcnn-train"),
        ("src.scripts.report", "qcnn-report"),
        ("src.scripts.heatmap", "qcnn-heatmap"),
        ("src.scripts.plot_curves", "qcnn-plot-curves"),
    ],
)
def test_cli_help_invocation(
    module_name: str, program_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not QISKIT_AVAILABLE and module_name != "src.scripts.preprocess":
        pytest.skip("qiskit is not installed")

    module = __import__(module_name, fromlist=["cli"])
    monkeypatch.setattr(sys, "argv", [program_name, "--help"])

    with pytest.raises(SystemExit) as excinfo:
        module.cli()
    assert excinfo.value.code == 0
