"""CLI for medMNIST preprocessing into normalized grayscale tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.color2gray import Color2GrayNet
from src.datasets.loader import download_medmnist, split_train_val
from src.utils.logging import get_logger
from src.utils.paths import (
    resolve_data_dir,
    resolve_processed_data_dir,
    resolve_project_root,
    resolve_raw_data_dir,
)
from src.utils.seed import set_global_seed

log = get_logger(__name__)


def _ensure_rgb_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure input batch has three channels.

    Parameters
    ----------
    x : torch.Tensor
        Batch tensor of shape ``(batch, channels, height, width)``.

    Returns
    -------
    torch.Tensor
        Batch tensor with three channels.
    """
    channels = x.size(1)
    if channels == 3:
        return x
    if channels == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, received {channels}.")


def _luma_target(x_rgb: torch.Tensor) -> torch.Tensor:
    """Compute luminance targets from RGB batches."""
    return (0.299 * x_rgb[:, 0] + 0.587 * x_rgb[:, 1] + 0.114 * x_rgb[:, 2]).unsqueeze(1)


def train_color2gray(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-2,
    epochs: int = 3,
    device: str = "cpu",
) -> nn.Module:
    """Train the grayscale projection network with MSE luminance targets.

    Parameters
    ----------
    model : torch.nn.Module
        Model that maps RGB tensors to one channel.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    lr : float, default=1e-2
        Adam learning rate.
    epochs : int, default=3
        Number of epochs.
    device : str, default="cpu"
        Torch device string.

    Returns
    -------
    torch.nn.Module
        Model loaded with the best validation checkpoint.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_state = model.state_dict()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_running = 0.0
        for x, _ in tqdm(train_loader, desc=f"preprocess train epoch {epoch}/{epochs}"):
            x = _ensure_rgb_batch(x.to(device))
            target = _luma_target(x)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_running += loss.item() * x.size(0)
        train_loss = train_running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = _ensure_rgb_batch(x.to(device))
                target = _luma_target(x)
                prediction = model(x)
                val_running += criterion(prediction, target).item() * x.size(0)
        val_loss = val_running / len(val_loader.dataset)

        log.info(
            f"[Epoch {epoch}/{epochs}] " f"train_loss={train_loss:.6f} " f"val_loss={val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model


def apply_and_save(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    out_dir: Path,
    *,
    device: str = "cpu",
) -> Path:
    """Apply grayscale transform and persist one split as a tensor file.

    Parameters
    ----------
    model : torch.nn.Module
        Trained grayscale conversion network.
    loader : torch.utils.data.DataLoader
        Data loader for one split.
    split : str
        Split label, for example ``"train"``.
    out_dir : pathlib.Path
        Output directory where ``<split>.pt`` is written.
    device : str, default="cpu"
        Torch device string.

    Returns
    -------
    pathlib.Path
        Path to the saved tensor file.
    """
    x_batches: list[torch.Tensor] = []
    y_batches: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"convert {split}"):
            x = _ensure_rgb_batch(x.to(device))
            x_gray = model(x).cpu()
            x_gray = torch.nn.functional.pad(x_gray, (1, 1, 1, 1))
            x_batches.append(x_gray)
            y_batches.append(y)

    x_tensor = torch.cat(x_batches).to(torch.float32)
    y_tensor = torch.cat(y_batches).to(torch.long)

    output_path = out_dir / f"{split}.pt"
    torch.save({"x": x_tensor, "y": y_tensor}, output_path)
    log.info(f"Saved split '{split}' to {output_path} with shape={tuple(x_tensor.shape)}.")
    return output_path


def cli() -> None:
    """Entry point for ``qcnn-preprocess``."""
    parser = argparse.ArgumentParser(
        prog="qcnn-preprocess",
        description=(
            "Download medMNIST, fit an RGB-to-grayscale projection, and save "
            "processed tensors under data/processed/<dataset>."
        ),
    )
    parser.add_argument("--dataset", default="pathmnist", help="medMNIST dataset name.")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for preprocessing.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for Color2GrayNet.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Color2GrayNet.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
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
    args = parser.parse_args()

    set_global_seed(args.seed)

    project_root = resolve_project_root(args.project_root)
    data_dir = resolve_data_dir(args.data_dir, project_root=project_root)
    raw_dir = resolve_raw_data_dir(data_dir)
    processed_dir = resolve_processed_data_dir(data_dir) / args.dataset
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Using data directory: {data_dir}")
    train_ds, test_ds = download_medmnist(args.dataset, root=raw_dir)
    train_ds, val_ds = split_train_val(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = Color2GrayNet(init="random")
    model = train_color2gray(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )

    apply_and_save(model, train_loader, "train", processed_dir, device=args.device)
    apply_and_save(model, val_loader, "val", processed_dir, device=args.device)
    apply_and_save(model, test_loader, "test", processed_dir, device=args.device)

    log.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    cli()
