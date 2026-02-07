"""Top-level package for the src project."""

from importlib.metadata import PackageNotFoundError, version

from src.utils.seed import set_global_seed

try:
    __version__ = version("QML-qcnn-medmnist")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "set_global_seed"]
