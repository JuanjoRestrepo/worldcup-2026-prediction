{{ config(materialized='view') }}

with training_runs as (
    select * from {{ ref('gold_training_runs_curated') }}
),
ranked_training_runs as (
    select
        *,
        row_number() over (
            order by trained_at_utc desc, persisted_at_utc desc
        ) as row_num
    from training_runs
)
select
    pipeline_run_id,
    artifact_path,
    data_path,
    training_rows,
    test_rows,
    feature_count,
    train_date_start,
    train_date_end,
    test_date_start,
    test_date_end,
    accuracy,
    macro_f1,
    weighted_f1,
    log_loss,
    feature_columns_json,
    class_distribution_train_json,
    class_distribution_test_json,
    classification_report_json,
    trained_at_utc,
    persisted_at_utc
from ranked_training_runs
where row_num = 1
