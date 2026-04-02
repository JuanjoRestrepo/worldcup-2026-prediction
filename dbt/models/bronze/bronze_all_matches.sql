{{ config(materialized='view') }}

with historical_matches as (
    select
        nullif("date", '')::date as match_date,
        "homeTeam" as home_team,
        "awayTeam" as away_team,
        "homeGoals"::bigint as home_goals,
        "awayGoals"::bigint as away_goals,
        tournament,
        city,
        country,
        neutral::boolean as neutral,
        'historical' as source_stream,
        pipeline_run_id,
        persisted_at_utc
    from {{ source('bronze', 'historical_matches') }}
),
api_matches as (
    select
        nullif("date", '')::date as match_date,
        "homeTeam" as home_team,
        "awayTeam" as away_team,
        nullif("homeGoals", '')::bigint as home_goals,
        nullif("awayGoals", '')::bigint as away_goals,
        tournament,
        city,
        country,
        case
            when lower(coalesce(nullif(neutral, ''), 'false')) in ('true', 't', '1', 'yes') then true
            when lower(coalesce(nullif(neutral, ''), 'false')) in ('false', 'f', '0', 'no') then false
            else null
        end as neutral,
        'api' as source_stream,
        pipeline_run_id,
        persisted_at_utc
    from {{ source('bronze', 'api_matches') }}
)
select * from historical_matches
union all
select * from api_matches
