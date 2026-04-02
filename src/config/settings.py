import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.BASE_DIR: Path = BASE_DIR
        self.DATA_DIR: Path = Path(os.getenv("DATA_DIR", self.BASE_DIR / "data"))
        self.RAW_DIR: Path = self.DATA_DIR / "raw"
        self.BRONZE_DIR: Path = self.DATA_DIR / "bronze"
        self.SILVER_DIR: Path = self.DATA_DIR / "silver"
        self.GOLD_DIR: Path = self.DATA_DIR / "gold"
        self.MODELS_DIR: Path = self.BASE_DIR / "models"

        self.DB_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
        self.DB_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
        self.DB_NAME: str = os.getenv("POSTGRES_DB", "wc_db")
        self.DB_USER: str = os.getenv("POSTGRES_USER", "wc_user")
        self.DB_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "wc_password")
        self.DBT_BASE_SCHEMA: str = os.getenv("DBT_BASE_SCHEMA", "analytics").strip()
        self.PERSIST_TO_DB: bool = os.getenv("PERSIST_TO_DB", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.PREDICTION_FEATURE_SOURCE: str = os.getenv(
            "PREDICTION_FEATURE_SOURCE",
            "auto",
        ).strip().lower()
        if self.PREDICTION_FEATURE_SOURCE not in {"auto", "dbt", "postgres", "csv"}:
            raise ValueError(
                "PREDICTION_FEATURE_SOURCE must be one of: auto, dbt, postgres, csv."
            )
        self.MONITORING_SOURCE: str = os.getenv(
            "MONITORING_SOURCE",
            "auto",
        ).strip().lower()
        if self.MONITORING_SOURCE not in {"auto", "dbt", "postgres"}:
            raise ValueError(
                "MONITORING_SOURCE must be one of: auto, dbt, postgres."
            )

        self.FOOTBALL_API_KEY: str | None = os.getenv("FOOTBALL_API_KEY")
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
        self.MODEL_ARTIFACT_PATH: Path = Path(
            os.getenv(
                "MODEL_ARTIFACT_PATH",
                self.MODELS_DIR / "match_predictor.joblib",
            )
        )

    def ensure_project_dirs(self) -> None:
        """Create the standard project directories if they do not exist."""
        for path in (
            self.RAW_DIR,
            self.BRONZE_DIR,
            self.SILVER_DIR,
            self.GOLD_DIR,
            self.MODELS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
