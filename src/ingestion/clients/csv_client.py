"""CSV client for loading historical international football results.

Supports downloading the latest dataset from the canonical GitHub source
(martj42/international_results) with automatic fallback to the local file
when the network is unavailable. The download occurs only at *pipeline run
time* — never at API serving time — so there is zero latency impact on the
production endpoint.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical raw CSV source maintained by martj42 on GitHub.
# This is the single source of truth for the historical results dataset.
MARTJ42_CSV_URL: str = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/refs/heads/master/results.csv"
)

# Local path where the raw CSV is persisted after each successful download.
RAW_PATH: Path = settings.RAW_DIR / "international_results.csv"

# Minimum expected columns in the source CSV schema.
# Used to detect corrupted or misformatted downloads early.
EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_csv(url: str, local_path: Path, timeout_seconds: int) -> bool:
    """Download *url* to *local_path*, returning True on success.

    A temporary file is written first so that an interrupted download does not
    corrupt the existing local copy.

    Args:
        url: Remote CSV URL to fetch.
        local_path: Destination path (parent directory must exist).
        timeout_seconds: Network timeout in seconds.

    Returns:
        True if the file was downloaded and persisted successfully.
    """
    tmp_path = local_path.with_suffix(".tmp")
    try:
        logger.info("Downloading historical CSV from %s ...", url)
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            content = response.read()
        tmp_path.write_bytes(content)
        # Rename atomically (best-effort on Windows).
        tmp_path.replace(local_path)
        logger.info(
            "Download complete — saved %s bytes → %s",
            len(content),
            local_path,
        )
        return True
    except urllib.error.URLError as exc:
        logger.warning(
            "Network download failed (%s). Falling back to local file: %s",
            exc,
            local_path,
        )
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unexpected error during CSV download (%s). Falling back to local file.",
            exc,
        )
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def _validate_schema(df: pd.DataFrame, source: str) -> None:
    """Raise ValueError if the DataFrame is missing required columns.

    Args:
        df: Loaded DataFrame to validate.
        source: Human-readable source label for error messages.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV from '{source}' is missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_historical_data(
    *,
    url: str = MARTJ42_CSV_URL,
    local_path: Path = RAW_PATH,
    force_local: bool = False,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Load the historical international football results dataset.

    Download strategy (executed at *pipeline run time*, not at serving time):
    1. If ``force_local=False`` (default), attempt to download the latest CSV
       from *url* and persist it to *local_path*.
    2. If the download fails for any reason (timeout, network error, etc.),
       fall back silently to the existing *local_path* file.
    3. If ``force_local=True``, skip the download entirely.

    In all cases the returned DataFrame contains the raw, un-standardized
    contents of the CSV including future fixture rows with ``NA`` scores.
    Downstream pipeline stages (ingestion_pipeline.py) handle null-score
    filtering before the bronze validation contract.

    Args:
        url: Remote CSV URL. Defaults to the martj42 GitHub source.
        local_path: Local file path for persisting the download and fallback.
        force_local: When True, skip the network download unconditionally.
        timeout_seconds: HTTP request timeout in seconds.

    Returns:
        Raw historical matches DataFrame.

    Raises:
        FileNotFoundError: If neither a successful download nor a local file
            is available.
        ValueError: If the downloaded/local file is missing required columns.
    """
    source_used: str

    if force_local:
        logger.info("force_local=True — skipping download, using local file.")
        source_used = "local"
    else:
        downloaded = _download_csv(url, local_path, timeout_seconds)
        source_used = "github_url" if downloaded else "local_fallback"

    if not local_path.exists():
        raise FileNotFoundError(
            f"No local CSV found at {local_path} and download was not successful. "
            "Ensure the file exists or check your network connection."
        )

    try:
        df = pd.read_csv(local_path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CSV at {local_path}: {exc}") from exc

    pre_drop_len = len(df)
    if "home_score" in df.columns and "away_score" in df.columns:
        df = df.dropna(subset=["home_score", "away_score"])
    dropped = pre_drop_len - len(df)
    if dropped > 0:
        logger.info(
            "csv_client: Dropped %s future-fixture rows (null scores) early.",
            dropped,
        )

    _validate_schema(df, source=source_used)

    date_min = df["date"].min()
    date_max = df["date"].max()
    logger.info(
        "Loaded historical dataset: %s rows × %s cols | source=%s | date_range=[%s, %s]",
        df.shape[0],
        df.shape[1],
        source_used,
        date_min,
        date_max,
    )
    return df
