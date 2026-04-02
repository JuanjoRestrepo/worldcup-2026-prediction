select
    team,
    team_role,
    count(*) as row_count
from {{ ref('gold_latest_team_snapshots') }}
group by 1, 2
having count(*) > 1
