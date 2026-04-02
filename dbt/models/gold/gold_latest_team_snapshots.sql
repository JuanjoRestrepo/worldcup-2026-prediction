{{ config(materialized='view') }}

with team_feature_snapshots as (
    select * from {{ ref('gold_team_feature_snapshots') }}
),
latest_role_snapshots as (
    select
        snapshot_date,
        team,
        opponent,
        team_role,
        elo,
        opponent_elo,
        avg_goals_last5,
        avg_goals_conceded_last5,
        global_avg_goals_last5,
        global_avg_conceded_last5,
        win_rate_last5,
        global_win_rate_last5,
        avg_opponent_elo_last5,
        weighted_win_rate_last5,
        opponent_elo_form,
        elo_form,
        home_advantage_effect,
        is_friendly,
        is_world_cup,
        is_qualifier,
        is_continental,
        pipeline_run_id,
        persisted_at_utc
    from (
        select
            *,
            row_number() over (
                partition by team, team_role
                order by snapshot_date desc, persisted_at_utc desc
            ) as row_num
        from team_feature_snapshots
    ) ranked
    where row_num = 1
),
latest_overall_snapshots as (
    select
        snapshot_date,
        team,
        opponent,
        'overall' as team_role,
        elo,
        opponent_elo,
        avg_goals_last5,
        avg_goals_conceded_last5,
        global_avg_goals_last5,
        global_avg_conceded_last5,
        win_rate_last5,
        global_win_rate_last5,
        avg_opponent_elo_last5,
        weighted_win_rate_last5,
        opponent_elo_form,
        elo_form,
        home_advantage_effect,
        is_friendly,
        is_world_cup,
        is_qualifier,
        is_continental,
        pipeline_run_id,
        persisted_at_utc
    from (
        select
            *,
            row_number() over (
                partition by team
                order by snapshot_date desc, persisted_at_utc desc
            ) as row_num
        from team_feature_snapshots
    ) ranked
    where row_num = 1
)
select * from latest_role_snapshots
union all
select * from latest_overall_snapshots
