from __future__ import annotations

from src.utils.paths import resolve_project_root


def test_resolve_project_root_points_to_repository_root() -> None:
    root = resolve_project_root()
    assert (root / "pyproject.toml").is_file()
