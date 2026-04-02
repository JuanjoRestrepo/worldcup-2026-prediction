"""Unit tests for the end-to-end pipeline orchestrator."""

import pandas as pd

import run_pipeline


def test_run_full_pipeline_passes_db_persistence_flag(monkeypatch):
    calls: dict[str, object] = {}

    def fake_ingestion(*, persist_to_db, pipeline_run_id):
        calls["ingestion"] = {
            "persist_to_db": persist_to_db,
            "pipeline_run_id": pipeline_run_id,
        }

    def fake_processing(*, use_api_data, persist_to_db, pipeline_run_id):
        calls["processing"] = {
            "use_api_data": use_api_data,
            "persist_to_db": persist_to_db,
            "pipeline_run_id": pipeline_run_id,
        }
        return pd.DataFrame({"col": [1, 2], "target_multiclass": [1, 0]})

    def fake_training(*, data_path, artifact_path, test_size, persist_to_db, pipeline_run_id):
        calls["training"] = {
            "data_path": data_path,
            "artifact_path": artifact_path,
            "test_size": test_size,
            "persist_to_db": persist_to_db,
            "pipeline_run_id": pipeline_run_id,
        }
        return {
            "metrics": {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "weighted_f1": 0.45,
                "log_loss": 1.0,
            },
            "training_rows": 10,
            "test_rows": 2,
            "feature_count": 3,
        }

    monkeypatch.setattr(run_pipeline, "run_ingestion_pipeline", fake_ingestion)
    monkeypatch.setattr(run_pipeline, "run_processing_pipeline", fake_processing)
    monkeypatch.setattr(run_pipeline, "train_and_export_model", fake_training)

    summary = run_pipeline.run_full_pipeline(
        run_ingestion=True,
        run_processing=True,
        run_training=True,
        use_api_data=False,
        persist_to_db=True,
    )

    assert calls["ingestion"]["persist_to_db"] is True
    assert calls["processing"]["persist_to_db"] is True
    assert calls["processing"]["use_api_data"] is False
    assert calls["training"]["persist_to_db"] is True
    assert calls["ingestion"]["pipeline_run_id"] == summary["pipeline_run_id"]
    assert calls["processing"]["pipeline_run_id"] == summary["pipeline_run_id"]
    assert calls["training"]["pipeline_run_id"] == summary["pipeline_run_id"]
    assert summary["stages"]["training"]["persisted_to_db"] is True
