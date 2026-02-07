"""medMNIST dataset loading and train/validation splitting utilities."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.utils.paths import resolve_raw_data_dir

_DEFAULT_DATASET = "pathmnist"
_MEAN = 0.5
_STD = 0.5


def _ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert single-channel tensors to pseudo-RGB.

    Parameters
    ----------
    x : torch.Tensor
        Tensor produced by ``ToTensor`` with shape ``(channels, height, width)``.

    Returns
    -------
    torch.Tensor
        Tensor with either three channels or unchanged dimensions if already
        three-channel.
    """
    if x.dim() != 3:
        return x
    channels = x.size(0)
    if channels == 3:
        return x
    if channels == 1:
        return x.repeat(3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels after ToTensor, received {channels}.")


def _get_transform() -> transforms.Compose:
    """Build tensor conversion and normalization transforms.

    Returns
    -------
    torchvision.transforms.Compose
        Transform pipeline that converts images to normalized tensors in
        approximately ``[-1, 1]``.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(_ensure_three_channels),
            transforms.Normalize(mean=[_MEAN] * 3, std=[_STD] * 3),
        ]
    )


def download_medmnist(
    name: str = _DEFAULT_DATASET,
    root: str | Path | None = None,
):
    """Download and instantiate medMNIST train and test datasets.

    Parameters
    ----------
    name : str, default="pathmnist"
        medMNIST dataset identifier.
    root : str | Path | None, optional
        Download root directory (raw data directory). If omitted, uses
        ``<project_root>/data/raw``. When provided, this path is used as-is
        without appending additional subdirectories.

    Returns
    -------
    tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
        Train and test datasets.
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
        raise ImportError(
            "medmnist is required for preprocessing. Install dependencies with "
            "`pip install -e .`."
        ) from exc

    if name not in INFO:
        available = ", ".join(sorted(INFO))
        raise ValueError(f"Unknown medMNIST dataset '{name}'. Available datasets: {available}.")

    root_path = resolve_raw_data_dir() if root is None else Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    data_class = getattr(medmnist, INFO[name]["python_class"])
    kwargs: dict[str, object] = {
        "transform": _get_transform(),
        "download": True,
        "root": str(root_path),
    }
    try:
        if "as_rgb" in inspect.signature(data_class.__init__).parameters:
            kwargs["as_rgb"] = True
    except (TypeError, ValueError):
        pass

    train_ds = data_class(split="train", **kwargs)
    test_ds = data_class(split="test", **kwargs)
    return train_ds, test_ds


def split_train_val(
    train_ds,
    val_frac: float = 0.15,
    seed: int = 42,
):
    """Split the training dataset into train and validation subsets.

    Notes
    -----
    This split uses ``torch.utils.data.random_split`` and is not stratified.

    Parameters
    ----------
    train_ds : torch.utils.data.Dataset
        Full training dataset.
    val_frac : float, default=0.15
        Fraction assigned to validation.
    seed : int, default=42
        Random split seed.

    Returns
    -------
    tuple[torch.utils.data.Subset, torch.utils.data.Subset]
        Training subset and validation subset.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in the open interval (0, 1).")

    val_len = int(len(train_ds) * val_frac)
    train_len = len(train_ds) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_ds, [train_len, val_len], generator=generator)
    return train_subset, val_subset


def get_dataloaders(
    batch_size: int = 128,
    dataset_name: str = _DEFAULT_DATASET,
    num_workers: int = 2,
):
    """Build train, validation, and test data loaders for medMNIST.

    Parameters
    ----------
    batch_size : int, default=128
        Data loader batch size.
    dataset_name : str, default="pathmnist"
        medMNIST dataset identifier.
    num_workers : int, default=2
        Number of worker processes for each data loader.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Train, validation, and test data loaders.
    """
    train_ds, test_ds = download_medmnist(dataset_name)
    train_ds, val_ds = split_train_val(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
