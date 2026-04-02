select
    count(distinct pipeline_run_id) as distinct_pipeline_runs
from {{ ref('gold_latest_team_snapshots') }}
having count(distinct pipeline_run_id) > 1
