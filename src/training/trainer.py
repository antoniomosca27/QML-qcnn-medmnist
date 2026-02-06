"""Training loop for the hybrid QCNN model."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as tud
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.hybrid_qcnn import HybridQCNN
from src.training.evaluate import classification_metrics
from src.utils.logging import get_logger
from src.utils.paths import resolve_processed_data_dir

log = get_logger(__name__)


def _load_split(path: Path, batch: int, shuffle: bool) -> DataLoader:
    """Load a preprocessed split file into a data loader.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.pt`` file containing keys ``"x"`` and ``"y"``.
    batch : int
        Batch size.
    shuffle : bool
        Whether to shuffle samples.

    Returns
    -------
    torch.utils.data.DataLoader
        Data loader wrapping a tensor dataset.
    """
    payload = torch.load(path, map_location="cpu")
    dataset = TensorDataset(payload["x"], payload["y"])
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle)


def _subsample(loader: DataLoader, fraction: float, split_name: str) -> DataLoader:
    """Subsample a data loader dataset by deterministic prefix slicing.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Source loader.
    fraction : float
        Fraction in ``(0, 1]``.
    split_name : str
        Split identifier used in logs.

    Returns
    -------
    torch.utils.data.DataLoader
        Original loader if ``fraction >= 1`` otherwise a subsampled loader.
    """
    if fraction >= 1.0:
        return loader
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"subset fraction for '{split_name}' must be in (0, 1].")

    sample_count = max(1, int(len(loader.dataset) * fraction))
    subset = tud.Subset(loader.dataset, list(range(sample_count)))
    log.warning(
        f"FAST MODE enabled for {split_name}: fraction={fraction:.3f}, samples={sample_count}."
    )

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, tud.RandomSampler),
    )


class Trainer:
    """Model trainer for preprocessed medMNIST tensors.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier, for example ``"bloodmnist"``.
    out_dir : pathlib.Path
        Run output directory where logs and checkpoints are written.
    data_dir : pathlib.Path | None, optional
        Data root directory. Defaults to ``<project_root>/data``.
    batch_size : int, default=64
        Batch size for all splits.
    lr : float, default=1e-3
        Learning rate for Adam.
    epochs : int, default=5
        Number of training epochs.
    device : str, default="cpu"
        Torch device string.
    subset : float, default=1.0
        Fraction for training split.
    subset_val : float | None, optional
        Fraction for validation split. Uses ``subset`` if ``None``.
    subset_test : float | None, optional
        Fraction for test split. Uses ``subset`` if ``None``.
    stride : int, default=3
        Patch stride for the quantum encoder.
    freeze_q : bool, default=False
        If ``True``, quantum parameters are frozen.
    """

    def __init__(
        self,
        dataset_name: str,
        out_dir: Path,
        data_dir: Path | None = None,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 5,
        device: str = "cpu",
        subset: float = 1.0,
        subset_val: float | None = None,
        subset_test: float | None = None,
        stride: int = 3,
        freeze_q: bool = False,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        processed_root = resolve_processed_data_dir(data_dir)
        split_root = processed_root / dataset_name
        train_path = split_root / "train.pt"
        val_path = split_root / "val.pt"
        test_path = split_root / "test.pt"

        for split_path in (train_path, val_path, test_path):
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Missing preprocessed split file: {split_path}. "
                    "Run qcnn-preprocess before training."
                )

        self.train_ld = _load_split(train_path, batch=batch_size, shuffle=True)
        self.val_ld = _load_split(val_path, batch=batch_size, shuffle=False)
        self.test_ld = _load_split(test_path, batch=batch_size, shuffle=False)

        val_fraction = subset if subset_val is None else subset_val
        test_fraction = subset if subset_test is None else subset_test
        self.train_ld = _subsample(self.train_ld, subset, "train")
        self.val_ld = _subsample(self.val_ld, val_fraction, "val")
        self.test_ld = _subsample(self.test_ld, test_fraction, "test")

        base_dataset = (
            self.train_ld.dataset.dataset
            if isinstance(self.train_ld.dataset, tud.Subset)
            else self.train_ld.dataset
        )
        labels = base_dataset.tensors[1]
        n_classes = int(labels.max().item() + 1)
        log.info(f"Detected {n_classes} classes from preprocessed tensors.")

        self.model: nn.Module = HybridQCNN(n_classes=n_classes, stride=stride).to(device)
        if freeze_q:
            for parameter in self.model.qconv.parameters():
                parameter.requires_grad = False
            log.warning("Quantum convolution parameters are frozen.")

        self.opt = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=lr,
        )

        self.csv_path = self.out_dir / "metrics.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])

        self.best_val_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, dict[str, float]]:
        """Run one epoch over a data loader.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Split data loader.
        train : bool
            Whether to perform gradient updates.

        Returns
        -------
        tuple[float, dict[str, float]]
            Epoch loss and computed metrics.
        """
        self.model.train(mode=train)

        total_loss = 0.0
        y_true: list[torch.Tensor] = []
        y_pred: list[torch.Tensor] = []
        mode_name = "train" if train else "eval"

        for x, y in tqdm(loader, desc=mode_name, leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            if train:
                self.opt.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                if hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(x, y, logits=logits)  # type: ignore[attr-defined]
                else:
                    loss = F.cross_entropy(logits, y.view(-1).long())

            if train:
                loss.backward()
                self.opt.step()

            total_loss += loss.item() * x.size(0)
            y_true.append(y)
            y_pred.append(logits.detach())

        epoch_loss = total_loss / len(loader.dataset)
        metrics = classification_metrics(torch.cat(y_pred), torch.cat(y_true))
        metrics["loss"] = float(epoch_loss)
        return epoch_loss, metrics

    def fit(self) -> None:
        """Run the full training loop and persist the best model checkpoint."""
        log.info(f"Starting training for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            train_loss, _ = self._run_epoch(self.train_ld, train=True)
            val_loss, val_metrics = self._run_epoch(self.val_ld, train=False)

            log.info(
                f"[Epoch {epoch}/{self.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        val_metrics["acc"],
                        val_metrics["f1"],
                    ]
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                torch.save(self.best_state, self.out_dir / "best_model.pt")
                log.info(f"Saved new best model with val_loss={val_loss:.4f}.")

        if self.best_state is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

    def test(self) -> dict[str, float]:
        """Evaluate the best checkpoint on the test split.

        Returns
        -------
        dict[str, float]
            Test metrics dictionary.
        """
        if self.best_state is None:
            checkpoint = self.out_dir / "best_model.pt"
            if not checkpoint.exists():
                raise RuntimeError("No best model available. Run fit() before test().")
            self.best_state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(self.best_state)
        _, metrics = self._run_epoch(self.test_ld, train=False)
        log.info(f"Test metrics: acc={metrics['acc']:.4f}, f1={metrics['f1']:.4f}.")
        return metrics
