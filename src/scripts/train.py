"""CLI for hybrid QCNN training."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.paths import resolve_data_dir, resolve_logs_dir, resolve_project_root
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def _next_run_id(dataset: str, logs_dir: Path) -> int:
    """Return the next run index for ``<dataset>_run_XXX``.

    Parameters
    ----------
    dataset : str
        Dataset identifier used in run naming.
    logs_dir : pathlib.Path
        Root directory containing run folders.

    Returns
    -------
    int
        Next run identifier in ascending order.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(dataset)}_run_(\d{{3}})")
    run_ids = [
        int(match.group(1))
        for candidate in logs_dir.glob(f"{dataset}_run_*")
        if (match := pattern.fullmatch(candidate.name))
    ]
    return (max(run_ids) + 1) if run_ids else 1


def cli() -> None:
    """Entry point for ``qcnn-train``."""
    parser = argparse.ArgumentParser(
        prog="qcnn-train",
        description="Train a hybrid QCNN model from preprocessed medMNIST tensors.",
    )
    parser.add_argument("--dataset", default="pathmnist", help="medMNIST dataset name.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of the training split to use in (0, 1].",
    )
    parser.add_argument(
        "--subset-val",
        type=float,
        default=None,
        help="Fraction of the validation split. Defaults to --subset.",
    )
    parser.add_argument(
        "--subset-test",
        type=float,
        default=None,
        help="Fraction of the test split. Defaults to --subset.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Patch extraction stride for 3x3 patches (>= 1).",
    )
    parser.add_argument(
        "--freeze-q",
        action="store_true",
        help="Freeze quantum convolution parameters during optimization.",
    )
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
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1.")

    set_global_seed(args.seed)

    project_root = resolve_project_root(args.project_root)
    data_dir = resolve_data_dir(args.data_dir, project_root=project_root)
    logs_dir = resolve_logs_dir(args.logs_dir, project_root=project_root)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = _next_run_id(args.dataset, logs_dir)
    out_dir = logs_dir / f"{args.dataset}_run_{run_id:03d}"
    log.info(f"Writing run outputs to {out_dir}.")

    trainer = Trainer(
        dataset_name=args.dataset,
        out_dir=out_dir,
        data_dir=data_dir,
        batch_size=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        subset=args.subset,
        subset_val=args.subset_val,
        subset_test=args.subset_test,
        stride=args.stride,
        freeze_q=args.freeze_q,
    )
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    cli()
