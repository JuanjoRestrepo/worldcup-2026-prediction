{{ config(materialized='view') }}

select
    pipeline_run_id,
    artifact_path,
    data_path,
    training_rows,
    test_rows,
    feature_count,
    train_date_start::date as train_date_start,
    train_date_end::date as train_date_end,
    test_date_start::date as test_date_start,
    test_date_end::date as test_date_end,
    accuracy,
    macro_f1,
    weighted_f1,
    log_loss,
    feature_columns_json::jsonb as feature_columns_json,
    class_distribution_train_json::jsonb as class_distribution_train_json,
    class_distribution_test_json::jsonb as class_distribution_test_json,
    classification_report_json::jsonb as classification_report_json,
    trained_at_utc::timestamptz as trained_at_utc,
    persisted_at_utc::timestamptz as persisted_at_utc
from {{ source('gold', 'training_runs') }}
