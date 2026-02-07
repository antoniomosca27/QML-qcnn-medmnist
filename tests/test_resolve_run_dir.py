from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_run_dir


def test_resolve_run_dir_run_name_inside_logs_dir(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    resolved = resolve_run_dir("bloodmnist_run_001", logs_dir=logs_dir)
    assert resolved == tmp_path / "logs" / "bloodmnist_run_001"


def test_resolve_run_dir_logs_prefixed_relative_path_not_duplicated(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    resolved = resolve_run_dir("logs/bloodmnist_run_001", logs_dir=logs_dir)
    assert resolved == tmp_path / "logs" / "bloodmnist_run_001"


def test_resolve_run_dir_custom_logs_dir_prefix_not_duplicated(tmp_path: Path) -> None:
    logs_dir = tmp_path / "runs"
    resolved = resolve_run_dir("runs/abc_run_001", logs_dir=logs_dir)
    assert resolved == tmp_path / "runs" / "abc_run_001"


def test_resolve_run_dir_absolute_path_is_preserved(tmp_path: Path) -> None:
    logs_dir = tmp_path / "runs"
    run_path = tmp_path / "any" / "run"
    resolved = resolve_run_dir(str(run_path), logs_dir=logs_dir)
    assert resolved == run_path


def test_resolve_run_dir_generic_relative_subpath_uses_cwd(tmp_path: Path, monkeypatch) -> None:
    logs_dir = tmp_path / "logs"
    monkeypatch.chdir(tmp_path)
    resolved = resolve_run_dir("artifacts/abc_run_001", logs_dir=logs_dir)
    assert resolved == tmp_path / "artifacts" / "abc_run_001"
