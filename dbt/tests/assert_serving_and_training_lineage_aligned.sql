with latest_training_run as (
    select pipeline_run_id
    from {{ ref('gold_latest_training_run') }}
),
latest_serving_snapshots as (
    select distinct pipeline_run_id
    from {{ ref('gold_latest_team_snapshots') }}
)
select
    training.pipeline_run_id as training_pipeline_run_id,
    serving.pipeline_run_id as serving_pipeline_run_id
from latest_training_run as training
left join latest_serving_snapshots as serving
    on training.pipeline_run_id = serving.pipeline_run_id
where serving.pipeline_run_id is null
