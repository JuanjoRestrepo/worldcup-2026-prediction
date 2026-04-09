-- Minimal fixture for API testing
-- Provides sample team snapshots for /predict endpoint

CREATE TABLE IF NOT EXISTS analytics_gold.gold_latest_team_snapshots (
    snapshot_date DATE,
    team VARCHAR,
    opponent VARCHAR,
    team_role VARCHAR,
    elo FLOAT,
    opponent_elo FLOAT,
    avg_goals_last5 FLOAT,
    avg_goals_conceded_last5 FLOAT,
    global_avg_goals_last5 FLOAT,
    global_avg_conceded_last5 FLOAT,
    win_rate_last5 FLOAT,
    global_win_rate_last5 FLOAT,
    avg_opponent_elo_last5 FLOAT,
    weighted_win_rate_last5 FLOAT,
    opponent_elo_form FLOAT,
    elo_form FLOAT,
    home_advantage_effect FLOAT,
    is_friendly BOOLEAN,
    is_world_cup BOOLEAN,
    is_qualifier BOOLEAN,
    is_continental BOOLEAN,
    pipeline_run_id VARCHAR,
    persisted_at_utc TIMESTAMP
);

-- Insert sample data for Argentina and Brazil
INSERT INTO analytics_gold.gold_latest_team_snapshots
VALUES
    (CURRENT_DATE, 'Argentina', 'Brazil', 'home', 1850.0, 1790.0, 2.1, 0.8, 1.95, 0.85, 0.75, 0.72, 1750.0, 0.73, 0.80, 0.85, 0.20, false, false, false, false, 'test_run', NOW()),
    (CURRENT_DATE, 'Argentina', null, 'away', 1850.0, 1790.0, 2.1, 0.8, 1.95, 0.85, 0.75, 0.72, 1750.0, 0.73, 0.80, 0.85, 0.20, false, false, false, false, 'test_run', NOW()),
    (CURRENT_DATE, 'Brazil', 'Argentina', 'away', 1790.0, 1850.0, 2.0, 0.9, 1.95, 0.85, 0.70, 0.72, 1750.0, 0.71, 0.78, 0.80, 0.20, false, false, false, false, 'test_run', NOW()),
    (CURRENT_DATE, 'Brazil', null, 'home', 1790.0, 1850.0, 2.0, 0.9, 1.95, 0.85, 0.70, 0.72, 1750.0, 0.71, 0.78, 0.80, 0.20, false, false, false, false, 'test_run', NOW());
