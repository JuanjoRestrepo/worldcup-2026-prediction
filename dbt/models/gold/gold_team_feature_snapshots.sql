{{ config(materialized='view') }}

with feature_dataset as (
    select * from {{ ref('gold_feature_dataset_curated') }}
),
home_snapshots as (
    select
        match_date as snapshot_date,
        home_team as team,
        away_team as opponent,
        'home' as team_role,
        elo_home as elo,
        elo_away as opponent_elo,
        home_avg_goals_last5 as avg_goals_last5,
        home_avg_goals_conceded_last5 as avg_goals_conceded_last5,
        home_global_avg_goals_last5 as global_avg_goals_last5,
        home_global_avg_conceded_last5 as global_avg_conceded_last5,
        home_win_rate_last5 as win_rate_last5,
        home_global_win_rate_last5 as global_win_rate_last5,
        home_avg_opponent_elo_last5 as avg_opponent_elo_last5,
        home_weighted_win_rate_last5 as weighted_win_rate_last5,
        home_opponent_elo_form as opponent_elo_form,
        home_elo_form as elo_form,
        home_advantage_effect,
        is_friendly,
        is_world_cup,
        is_qualifier,
        is_continental,
        pipeline_run_id,
        persisted_at_utc
    from feature_dataset
),
away_snapshots as (
    select
        match_date as snapshot_date,
        away_team as team,
        home_team as opponent,
        'away' as team_role,
        elo_away as elo,
        elo_home as opponent_elo,
        away_avg_goals_last5 as avg_goals_last5,
        away_avg_goals_conceded_last5 as avg_goals_conceded_last5,
        away_global_avg_goals_last5 as global_avg_goals_last5,
        away_global_avg_conceded_last5 as global_avg_conceded_last5,
        away_win_rate_last5 as win_rate_last5,
        away_global_win_rate_last5 as global_win_rate_last5,
        away_avg_opponent_elo_last5 as avg_opponent_elo_last5,
        away_weighted_win_rate_last5 as weighted_win_rate_last5,
        away_opponent_elo_form as opponent_elo_form,
        away_elo_form as elo_form,
        home_advantage_effect,
        is_friendly,
        is_world_cup,
        is_qualifier,
        is_continental,
        pipeline_run_id,
        persisted_at_utc
    from feature_dataset
)
select * from home_snapshots
union all
select * from away_snapshots
