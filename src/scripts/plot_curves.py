"""CLI for plotting training and validation learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.paths import (
    resolve_logs_dir,
    resolve_project_root,
    resolve_reports_dir,
    resolve_run_dir,
)

log = get_logger(__name__)


def generate_learning_curves(
    logdir: Path,
    *,
    logs_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Generate loss and accuracy curves from ``metrics.csv``.

    Parameters
    ----------
    logdir : pathlib.Path
        Run directory path or run directory name.
    logs_dir : pathlib.Path | None, optional
        Logs directory used when ``logdir`` is relative.
    reports_dir : pathlib.Path | None, optional
        Reports directory root.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to ``learning_curve_loss.png`` and ``learning_curve_accuracy.png``.
    """
    run_dir = resolve_run_dir(logdir, logs_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_path)
    required = {"epoch", "train_loss", "val_loss", "val_acc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metrics.csv: {sorted(missing)}")

    reports_root = resolve_reports_dir(reports_dir)
    figures_dir = reports_root / run_dir.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    loss_path = figures_dir / "learning_curve_loss.png"
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    accuracy_path = figures_dir / "learning_curve_accuracy.png"
    plt.figure()
    plt.plot(df["epoch"], df["val_acc"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Saved learning curves to {figures_dir}.")
    return loss_path, accuracy_path


def cli() -> None:
    """Entry point for ``qcnn-plot-curves``."""
    parser = argparse.ArgumentParser(
        prog="qcnn-plot-curves",
        description="Plot learning curves from metrics.csv for a training run.",
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Run directory path or run folder name inside --logs-dir.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root override. Defaults to QCNN_PROJECT_ROOT or repository root.",
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

    project_root = resolve_project_root(args.project_root)
    logs_dir = resolve_logs_dir(args.logs_dir, project_root=project_root)
    reports_dir = resolve_reports_dir(args.reports_dir, project_root=project_root)

    if not logs_dir.exists() and not Path(args.logdir).is_absolute():
        parser.error(f"logs directory does not exist: {logs_dir}")

    generate_learning_curves(Path(args.logdir), logs_dir=logs_dir, reports_dir=reports_dir)


if __name__ == "__main__":
    cli()
