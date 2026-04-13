from src.processing.transformers.match_standardizer import standardize_api


def test_standardize_api():
    sample = [
        {
            "utcDate": "2026-03-21",
            "homeTeam": {"name": "Brazil"},
            "awayTeam": {"name": "Argentina"},
            "score": {"fullTime": {"home": 2, "away": 1}},
            "competition": {"name": "Friendly"},
            "area": {"name": "South America"},
        }
    ]

    df = standardize_api(sample)

    assert len(df) == 1
    assert "homeTeam" in df.columns
    assert "homeGoals" in df.columns
