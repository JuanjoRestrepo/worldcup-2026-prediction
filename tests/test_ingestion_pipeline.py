from unittest.mock import MagicMock, patch

import pandas as pd

from backend.ingestion.pipelines.ingestion_pipeline import run_ingestion_pipeline


@patch("backend.ingestion.pipelines.ingestion_pipeline.load_historical_data")
@patch("backend.ingestion.pipelines.ingestion_pipeline.validate_schema")
@patch("backend.ingestion.pipelines.ingestion_pipeline.standardize_csv")
@patch(
    "backend.ingestion.pipelines.ingestion_pipeline.validate_standardized_matches_contract"
)
@patch("backend.ingestion.pipelines.ingestion_pipeline.persist_dataframe")
@patch("backend.ingestion.pipelines.ingestion_pipeline.FootballAPIClient")
@patch("backend.ingestion.pipelines.ingestion_pipeline.filter_international_matches")
@patch("backend.ingestion.pipelines.ingestion_pipeline.standardize_api")
def test_run_ingestion_pipeline(
    mock_std_api,
    mock_filter,
    mock_api_client_cls,
    mock_persist,
    mock_validate_contract,
    mock_std_csv,
    mock_validate_schema,
    mock_load_hist,
    tmp_path,
    monkeypatch,
):
    from backend.config.settings import settings

    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path)
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    monkeypatch.setattr(settings, "FOOTBALL_API_KEY", "test_key")

    mock_load_hist.return_value = pd.DataFrame(
        {"home_score": [1, None], "away_score": [2, None]}
    )

    mock_std_csv.return_value = pd.DataFrame(
        {"homeTeam": ["A"], "awayTeam": ["B"], "homeGoals": [1], "awayGoals": [2]}
    )

    mock_api_client = MagicMock()
    mock_api_client_cls.return_value = mock_api_client
    mock_api_client.get_matches.return_value = {"matches": [{"id": 1}]}

    mock_filter.return_value = [{"id": 1}]

    mock_std_api.return_value = pd.DataFrame(
        {
            "homeTeam": ["C", "D"],
            "awayTeam": ["E", "F"],
            "homeGoals": [1, None],
            "awayGoals": [1, None],
        }
    )

    run_ingestion_pipeline(persist_to_db=True, pipeline_run_id="run_123")

    assert mock_load_hist.called
    assert mock_validate_schema.called
    assert mock_std_csv.called
    assert mock_validate_contract.call_count >= 1
    assert mock_persist.call_count == 2
    assert mock_api_client.get_matches.called
    assert mock_filter.called
    assert mock_std_api.called


@patch("backend.ingestion.pipelines.ingestion_pipeline.load_historical_data")
@patch("backend.ingestion.pipelines.ingestion_pipeline.standardize_csv")
def test_run_ingestion_pipeline_no_api_key(
    mock_std_csv, mock_load_hist, tmp_path, monkeypatch
):
    from backend.config.settings import settings

    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path)
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    monkeypatch.setattr(settings, "FOOTBALL_API_KEY", "")

    mock_load_hist.return_value = pd.DataFrame({"home_score": [], "away_score": []})
    mock_std_csv.return_value = pd.DataFrame()

    with (
        patch("backend.ingestion.pipelines.ingestion_pipeline.validate_schema"),
        patch(
            "backend.ingestion.pipelines.ingestion_pipeline.validate_standardized_matches_contract"
        ),
    ):
        run_ingestion_pipeline()

    assert mock_load_hist.called
