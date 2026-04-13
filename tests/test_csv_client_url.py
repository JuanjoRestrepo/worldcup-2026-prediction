"""Tests for the updated csv_client.py URL download behaviour.

These tests verify:
- ``force_local=True`` skips the network and reads the local file directly.
- A ``URLError`` during download triggers a clean fallback to the local file.
- The returned DataFrame satisfies the expected schema contract.
- A missing local file after a failed download raises ``FileNotFoundError``.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.clients.csv_client import (
    EXPECTED_COLUMNS,
    load_historical_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_CSV_CONTENT = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2024-01-01,Brazil,Argentina,2,1,Friendly,Brasilia,Brazil,FALSE\n"
    "2024-03-15,Germany,France,1,1,Friendly,Berlin,Germany,FALSE\n"
)


@pytest.fixture()
def local_csv(tmp_path: Path) -> Path:
    """Write a minimal valid CSV to a temp file and return its path."""
    csv_path = tmp_path / "international_results.csv"
    csv_path.write_text(_MINIMAL_CSV_CONTENT, encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Tests: force_local=True skips download
# ---------------------------------------------------------------------------


class TestForceLocal:
    """When force_local=True the download is skipped unconditionally."""

    def test_load_returns_dataframe_from_local_file(self, local_csv: Path) -> None:
        """The function reads the local CSV without touching the network."""
        with patch("src.ingestion.clients.csv_client._download_csv") as mock_dl:
            df = load_historical_data(local_path=local_csv, force_local=True)

        mock_dl.assert_not_called()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # noqa: PLR2004 — two fixture rows

    def test_load_has_expected_columns(self, local_csv: Path) -> None:
        """All required schema columns are present."""
        df = load_historical_data(local_path=local_csv, force_local=True)
        missing = EXPECTED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"


# ---------------------------------------------------------------------------
# Tests: network fallback on URLError
# ---------------------------------------------------------------------------


class TestNetworkFallback:
    """When download raises URLError the function falls back to the local file."""

    def test_falls_back_to_local_on_url_error(
        self, local_csv: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A URLError during download must not crash the pipeline."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("simulated timeout")
            with caplog.at_level("WARNING", logger="src.ingestion.clients.csv_client"):
                df = load_historical_data(
                    url="https://example.invalid/results.csv",
                    local_path=local_csv,
                )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # noqa: PLR2004 — local fixture rows
        assert any("falling back" in record.message.lower() for record in caplog.records)

    def test_falls_back_to_local_on_os_error(self, local_csv: Path) -> None:
        """An OSError (e.g. DNS failure) also triggers the local fallback."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = OSError("network unreachable")
            df = load_historical_data(local_path=local_csv)

        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: successful download path
# ---------------------------------------------------------------------------


class TestSuccessfulDownload:
    """When the download succeeds the local file is updated with fresh content."""

    def test_successful_download_overwrites_local(self, tmp_path: Path) -> None:
        """A successful download persists the content to the local path."""
        local_csv = tmp_path / "international_results.csv"
        # Write stale content so we can verify it gets replaced.
        local_csv.write_text(
            "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
            "2020-01-01,OldTeam,AnotherOld,0,0,Friendly,City,Country,FALSE\n",
            encoding="utf-8",
        )

        fresh_bytes = _MINIMAL_CSV_CONTENT.encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = fresh_bytes
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            df = load_historical_data(local_path=local_csv)

        # The fresh fixture content has 2 rows (Brazil and Germany).
        assert len(df) == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Tests: missing file error
# ---------------------------------------------------------------------------


class TestMissingFile:
    """FileNotFoundError is raised when neither download nor local file exists."""

    def test_raises_when_no_local_file_and_download_fails(self, tmp_path: Path) -> None:
        """If the local file is absent and download fails, raise FileNotFoundError."""
        nonexistent = tmp_path / "not_here.csv"
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("offline")
            with pytest.raises(FileNotFoundError, match="No local CSV found"):
                load_historical_data(local_path=nonexistent)


# ---------------------------------------------------------------------------
# Tests: schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """ValueError is raised when downloaded content is missing required columns."""

    def test_raises_on_bad_schema(self, tmp_path: Path) -> None:
        """A CSV missing required columns raises ValueError on load."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing required columns"):
            load_historical_data(local_path=bad_csv, force_local=True)
