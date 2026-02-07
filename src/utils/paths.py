"""Path resolution helpers for project-level runtime directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def resolve_project_root(project_root: PathLike | None = None) -> Path:
    """Resolve the repository root for command execution.

    Parameters
    ----------
    project_root : str | Path | None, optional
        Explicit project root override. If omitted, the function checks
        ``QCNN_PROJECT_ROOT`` and then falls back to repository inference.

    Returns
    -------
    Path
        Absolute path to the project root. The fallback repository inference
        resolves to the repository directory that contains ``pyproject.toml``.
    """
    if project_root is not None:
        return Path(project_root).expanduser().resolve()

    env_root = os.getenv("QCNN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(__file__).resolve().parents[2]


def resolve_data_dir(
    data_dir: PathLike | None = None,
    *,
    project_root: PathLike | None = None,
) -> Path:
    """Resolve the data directory.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        Explicit data directory.
    project_root : str | Path | None, optional
        Optional project root used when ``data_dir`` is omitted.

    Returns
    -------
    Path
        Absolute data directory path.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    return resolve_project_root(project_root) / "data"


def resolve_logs_dir(
    logs_dir: PathLike | None = None,
    *,
    project_root: PathLike | None = None,
) -> Path:
    """Resolve the logs directory.

    Parameters
    ----------
    logs_dir : str | Path | None, optional
        Explicit logs directory.
    project_root : str | Path | None, optional
        Optional project root used when ``logs_dir`` is omitted.

    Returns
    -------
    Path
        Absolute logs directory path.
    """
    if logs_dir is not None:
        return Path(logs_dir).expanduser().resolve()
    return resolve_project_root(project_root) / "logs"


def resolve_reports_dir(
    reports_dir: PathLike | None = None,
    *,
    project_root: PathLike | None = None,
) -> Path:
    """Resolve the reports directory.

    Parameters
    ----------
    reports_dir : str | Path | None, optional
        Explicit reports directory.
    project_root : str | Path | None, optional
        Optional project root used when ``reports_dir`` is omitted.

    Returns
    -------
    Path
        Absolute reports directory path.
    """
    if reports_dir is not None:
        return Path(reports_dir).expanduser().resolve()
    return resolve_project_root(project_root) / "reports"


def resolve_raw_data_dir(
    data_dir: PathLike | None = None,
    *,
    project_root: PathLike | None = None,
) -> Path:
    """Resolve the raw dataset directory."""
    return resolve_data_dir(data_dir, project_root=project_root) / "raw"


def resolve_processed_data_dir(
    data_dir: PathLike | None = None,
    *,
    project_root: PathLike | None = None,
) -> Path:
    """Resolve the processed tensor directory."""
    return resolve_data_dir(data_dir, project_root=project_root) / "processed"


def resolve_run_dir(logdir: PathLike, logs_dir: PathLike | None = None) -> Path:
    """Resolve a run directory path with deterministic precedence.

    Parameters
    ----------
    logdir : str | Path
        Run directory path or name.

        Supported forms:
        - absolute path (for example ``/abs/path/to/logs/bloodmnist_run_001``);
        - run folder name (for example ``bloodmnist_run_001``);
        - relative path prefixed by the logs directory name
          (for example ``logs/bloodmnist_run_001`` when ``logs_dir`` ends with
          ``/logs``);
        - other relative paths with subdirectories.
    logs_dir : str | Path | None, optional
        Base logs directory used when ``logdir`` is relative. If provided,
        relative ``logdir`` values are resolved using this precedence:

        1. If the first component of ``logdir`` matches ``logs_dir.name``,
           interpret ``logdir`` as relative to ``logs_dir.parent``.
        2. Else if ``logdir`` is a simple folder name, append it to
           ``logs_dir``.
        3. Else resolve ``logdir`` from the current working directory without
           prepending ``logs_dir``.

    Returns
    -------
    Path
        Absolute run directory path.
    """
    candidate = Path(logdir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if logs_dir is not None:
        base_logs = Path(logs_dir).expanduser().resolve()

        if candidate.parts and candidate.parts[0] == base_logs.name:
            return (base_logs.parent / candidate).resolve()

        if candidate.parent == Path("."):
            return (base_logs / candidate).resolve()

        return candidate.resolve()

    return candidate.resolve()
