{{ config(materialized='view') }}

select
    md5(
        concat_ws(
            '||',
            "date"::text,
            coalesce("homeTeam", ''),
            coalesce("awayTeam", ''),
            coalesce(tournament, ''),
            coalesce(city, ''),
            coalesce(country, '')
        )
    ) as match_key,
    "date"::date as match_date,
    "homeTeam" as home_team,
    "awayTeam" as away_team,
    "homeGoals" as home_goals,
    "awayGoals" as away_goals,
    tournament,
    city,
    country,
    neutral,
    case
        when "homeGoals" > "awayGoals" then 1
        else 0
    end as target,
    case
        when "homeGoals" > "awayGoals" then 1
        when "homeGoals" = "awayGoals" then 0
        else -1
    end as target_multiclass,
    case
        when "homeGoals" > "awayGoals" then 'home_win'
        when "homeGoals" = "awayGoals" then 'draw'
        else 'away_win'
    end as target_label,
    pipeline_run_id,
    persisted_at_utc
from {{ source('silver', 'matches_cleaned') }}
