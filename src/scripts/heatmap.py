"""CLI for generating patch-level quantum activation heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.hybrid_qcnn import HybridQCNN
from src.quantum.encoder import image_to_patches, patch_grid_size
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


def generate_heatmap(
    dataset: str,
    logdir: Path,
    *,
    idx: int = 0,
    stride: int = 3,
    data_dir: Path | None = None,
    logs_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Generate a patch-level heatmap from the trained quantum layer.

    Parameters
    ----------
    dataset : str
        medMNIST dataset name.
    logdir : pathlib.Path
        Run directory path or run directory name.
    idx : int, default=0
        Index of the test sample.
    stride : int, default=3
        Patch extraction stride.
    data_dir : pathlib.Path | None, optional
        Data directory override.
    logs_dir : pathlib.Path | None, optional
        Logs directory used when ``logdir`` is relative.
    reports_dir : pathlib.Path | None, optional
        Reports directory root.

    Returns
    -------
    pathlib.Path
        Saved heatmap image path.
    """
    if idx < 0:
        raise ValueError("idx must be >= 0.")
    n_steps = patch_grid_size(stride=stride)

    run_dir = resolve_run_dir(logdir, logs_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    reports_root = resolve_reports_dir(reports_dir)
    processed_root = resolve_processed_data_dir(data_dir)
    test_path = processed_root / dataset / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test split does not exist: {test_path}")

    test_payload = torch.load(test_path, map_location="cpu")
    if idx >= len(test_payload["x"]):
        raise IndexError(f"idx={idx} is out of range for test split of size {len(test_payload['x'])}.")

    image = test_payload["x"][idx]
    n_classes = int(test_payload["y"].max().item()) + 1

    model = HybridQCNN(n_classes=n_classes, stride=stride)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    patches = image_to_patches(image, stride=stride)
    with torch.no_grad():
        activations = model.qconv(patches)
    activation_map = activations.view(n_steps, n_steps).cpu().numpy()

    fig_dir = reports_root / run_dir.name / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"heatmap_index_{idx}.png"

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(activation_map, cmap="viridis", square=True)
    plt.title(f"Patch Activation Heatmap - {dataset} (index={idx})")
    plt.xlabel("Patch Column")
    plt.ylabel("Patch Row")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Saved heatmap to {output_path}.")
    return output_path


def cli() -> None:
    """Entry point for ``qcnn-heatmap``."""
    parser = argparse.ArgumentParser(
        prog="qcnn-heatmap",
        description="Generate a patch-level activation heatmap for one test sample.",
    )
    parser.add_argument("--dataset", required=True, help="medMNIST dataset name.")
    parser.add_argument(
        "--logdir",
        required=True,
        help="Run directory path or run folder name inside --logs-dir.",
    )
    parser.add_argument("--idx", type=int, default=0, help="Index of the test sample.")
    parser.add_argument("--stride", type=int, default=3, help="Patch extraction stride (>= 1).")
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

    generate_heatmap(
        dataset=args.dataset,
        logdir=Path(args.logdir),
        idx=args.idx,
        stride=args.stride,
        data_dir=data_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    cli()
