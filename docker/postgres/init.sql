CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Inference logging table for observability and monitoring
CREATE TABLE IF NOT EXISTS monitoring.inference_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    timestamp_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    neutral BOOLEAN NOT NULL,
    tournament VARCHAR(255),
    predicted_class INTEGER NOT NULL,
    predicted_outcome VARCHAR(50) NOT NULL,
    class_probabilities_json JSONB NOT NULL,
    feature_snapshot_dates_json JSONB NOT NULL,
    feature_source VARCHAR(100) NOT NULL,
    model_artifact_path TEXT NOT NULL,
    model_version VARCHAR(100),
    persisted_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_inference_logs_timestamp 
    ON monitoring.inference_logs(timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_inference_logs_teams 
    ON monitoring.inference_logs(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_inference_logs_outcome 
    ON monitoring.inference_logs(predicted_outcome);
CREATE INDEX IF NOT EXISTS idx_inference_logs_model_version 
    ON monitoring.inference_logs(model_version);
