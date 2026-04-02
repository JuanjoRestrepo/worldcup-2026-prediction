#!/usr/bin/env python
"""Run dbt commands using the same .env-backed configuration as the Python pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DBT_DIR = BASE_DIR / "dbt"
DEFAULT_PROFILE = DBT_DIR / "profiles.yml"
PROFILE_TEMPLATE = DBT_DIR / "profiles.yml.example"


def _ensure_profiles_file() -> Path:
    """Create a local dbt profile from the example template if it is missing."""
    if DEFAULT_PROFILE.exists():
        return DEFAULT_PROFILE

    if not PROFILE_TEMPLATE.exists():
        raise FileNotFoundError(
            f"dbt profile template not found at '{PROFILE_TEMPLATE}'."
        )

    shutil.copyfile(PROFILE_TEMPLATE, DEFAULT_PROFILE)
    return DEFAULT_PROFILE


def _resolve_dbt_executable() -> Path:
    candidate = BASE_DIR / ".venv" / "Scripts" / "dbt.exe"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "dbt executable not found in .venv. Install project dependencies first."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dbt with project .env settings and the local dbt profile."
    )
    parser.add_argument(
        "dbt_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed directly to dbt, for example: run --select gold",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    os.environ.setdefault("DBT_THREADS", "1")
    args = _parse_args()
    dbt_args = args.dbt_args or ["debug"]

    _ensure_profiles_file()
    dbt_executable = _resolve_dbt_executable()

    command = [
        str(dbt_executable),
        *dbt_args,
        "--project-dir",
        str(DBT_DIR),
        "--profiles-dir",
        str(DBT_DIR),
    ]
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
