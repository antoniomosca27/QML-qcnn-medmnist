"""CLI for prediction reports and confusion matrix generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.hybrid_qcnn import HybridQCNN
from src.quantum.qconv import qconv_block3
from src.training.trainer import _load_split
from src.utils.logging import get_logger
from src.utils.paths import (
    resolve_data_dir,
    resolve_logs_dir,
    resolve_processed_data_dir,
    resolve_project_root,
    resolve_reports_dir,
    resolve_run_dir,
)

log = get_logger(__name__)


def generate_report(
    dataset: str,
    logdir: Path,
    stride: int,
    *,
    data_dir: Path | None = None,
    logs_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> None:
    """Generate prediction table, confusion matrix, and circuit metadata.

    Parameters
    ----------
    dataset : str
        medMNIST dataset name.
    logdir : pathlib.Path
        Run directory path or run directory name.
    stride : int
        Patch extraction stride used by the model.
    data_dir : pathlib.Path | None, optional
        Data directory override.
    logs_dir : pathlib.Path | None, optional
        Logs directory used when ``logdir`` is relative.
    reports_dir : pathlib.Path | None, optional
        Reports directory root.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1.")

    run_dir = resolve_run_dir(logdir, logs_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    reports_root = resolve_reports_dir(reports_dir)
    report_root = reports_root / run_dir.name
    figs_dir = report_root / "figures"
    tables_dir = report_root / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    processed_root = resolve_processed_data_dir(data_dir)
    test_path = processed_root / dataset / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test split does not exist: {test_path}")

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    test_loader = _load_split(test_path, batch=64, shuffle=False)
    n_classes = int(test_loader.dataset.tensors[1].max().item()) + 1

    model = HybridQCNN(n_classes=n_classes, stride=stride)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            y_true.append(y)
            y_pred.append(logits.argmax(1))

    y_true_np = torch.cat(y_true).view(-1).numpy()
    y_pred_np = torch.cat(y_pred).view(-1).numpy()

    predictions = pd.DataFrame({"true": y_true_np, "pred": y_pred_np})
    predictions_path = tables_dir / "preds.csv"
    predictions.to_csv(predictions_path, index=False)
    log.info(f"Saved predictions to {predictions_path}.")

    matrix = confusion_matrix(y_true_np, y_pred_np, normalize="true")
    display = ConfusionMatrixDisplay(matrix)
    display.plot(include_values=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset}")
    fig_path = figs_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved confusion matrix to {fig_path}.")

    circuit, _ = qconv_block3()
    metadata = {
        "dataset": dataset,
        "stride": stride,
        "num_qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
    }
    metadata_path = report_root / "report_meta.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log.info(f"Saved circuit metadata to {metadata_path}.")


def cli() -> None:
    """Entry point for ``qcnn-report``."""
    parser = argparse.ArgumentParser(
        prog="qcnn-report",
        description="Generate predictions, confusion matrix, and circuit metadata for a run.",
    )
    parser.add_argument("--dataset", required=True, help="medMNIST dataset name.")
    parser.add_argument(
        "--logdir",
        required=True,
        help="Run directory path or run folder name inside --logs-dir.",
    )
    parser.add_argument("--stride", type=int, default=3, help="Patch extraction stride.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root override. Defaults to QCNN_PROJECT_ROOT or repository root.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory override. Defaults to <project_root>/data.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Logs directory override. Defaults to <project_root>/logs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Reports directory override. Defaults to <project_root>/reports.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1.")

    project_root = resolve_project_root(args.project_root)
    data_dir = resolve_data_dir(args.data_dir, project_root=project_root)
    logs_dir = resolve_logs_dir(args.logs_dir, project_root=project_root)
    reports_dir = resolve_reports_dir(args.reports_dir, project_root=project_root)

    if not data_dir.exists():
        parser.error(f"data directory does not exist: {data_dir}")
    if not logs_dir.exists() and not Path(args.logdir).is_absolute():
        parser.error(f"logs directory does not exist: {logs_dir}")

    generate_report(
        dataset=args.dataset,
        logdir=Path(args.logdir),
        stride=args.stride,
        data_dir=data_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    cli()
