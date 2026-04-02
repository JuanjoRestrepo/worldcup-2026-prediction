"""Unit tests for the dbt wrapper script."""

from pathlib import Path

import run_dbt


def test_ensure_profiles_file_copies_example(monkeypatch):
    target = Path("dbt/profiles.yml")
    template = Path("dbt/profiles.yml.example")
    copied: dict[str, Path] = {}

    monkeypatch.setattr(run_dbt, "DEFAULT_PROFILE", target)
    monkeypatch.setattr(run_dbt, "PROFILE_TEMPLATE", template)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: self == template,
    )
    monkeypatch.setattr(
        run_dbt.shutil,
        "copyfile",
        lambda src, dst: copied.update({"src": src, "dst": dst}),
    )

    resolved = run_dbt._ensure_profiles_file()

    assert resolved == target
    assert copied == {"src": template, "dst": target}


def test_resolve_dbt_executable_uses_project_venv(monkeypatch):
    base_dir = Path("workspace")
    executable = base_dir / ".venv" / "Scripts" / "dbt.exe"

    monkeypatch.setattr(run_dbt, "BASE_DIR", base_dir)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: self == executable,
    )

    assert run_dbt._resolve_dbt_executable() == executable
