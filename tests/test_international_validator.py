"""Test international validator."""

import pytest

from src.ingestion.utils.international_validator import (
    CLUB_LEAGUES,
    INTERNATIONAL_COMPETITIONS,
    filter_international_matches,
    validate_international_match,
)


def test_validate_international_world_cup():
    """Test that World Cup matches are recognized as international."""
    match = {"competition": {"code": "WC", "name": "FIFA World Cup"}}
    assert validate_international_match(match) is True


def test_validate_international_euro():
    """Test that Euro matches are recognized as international."""
    match = {"competition": {"code": "EC", "name": "European Championship"}}
    assert validate_international_match(match) is True


def test_validate_international_friendly():
    """Test that friendly internationals are recognized."""
    match = {"competition": {"code": "FR", "name": "Friendly International"}}
    assert validate_international_match(match) is True


def test_exclude_club_league_premier():
    """Test that Premier League matches are excluded."""
    match = {"competition": {"code": "PL", "name": "Premier League"}}
    assert validate_international_match(match) is False


def test_exclude_club_league_la_liga():
    """Test that La Liga matches are excluded."""
    match = {"competition": {"code": "PD", "name": "Primera Division"}}
    assert validate_international_match(match) is False


def test_exclude_club_league_bundesliga():
    """Test that Bundesliga matches are excluded."""
    match = {"competition": {"code": "BL1", "name": "Bundesliga"}}
    assert validate_international_match(match) is False


def test_exclude_club_league_serie_a():
    """Test that Serie A matches are excluded."""
    match = {"competition": {"code": "SA", "name": "Serie A"}}
    assert validate_international_match(match) is False


def test_filter_international_matches():
    """Test filtering mixed matches."""
    matches = [
        {"competition": {"code": "WC", "name": "World Cup"}},  # Keep
        {"competition": {"code": "PL", "name": "Premier League"}},  # Remove
        {"competition": {"code": "EC", "name": "Euro"}},  # Keep
        {"competition": {"code": "BL1", "name": "Bundesliga"}},  # Remove
        {"competition": {"code": "FR", "name": "Friendly"}},  # Keep
        {"competition": {"code": "PD", "name": "La Liga"}},  # Remove
    ]

    filtered = filter_international_matches(matches)

    assert len(filtered) == 3
    assert all(m["competition"]["code"] in INTERNATIONAL_COMPETITIONS for m in filtered)


def test_no_club_leagues_in_filtered():
    """Verify no club league codes in filtered results."""
    matches = [
        {"competition": {"code": "WC", "name": "World Cup"}},
        {"competition": {"code": "PL", "name": "Premier League"}},
        {"competition": {"code": "BL1", "name": "Bundesliga"}},
        {"competition": {"code": "ECQ", "name": "Euro Qualifiers"}},
        {"competition": {"code": "SA", "name": "Serie A"}},
    ]

    filtered = filter_international_matches(matches)

    # Check no club league codes exist
    filtered_codes = {m["competition"]["code"] for m in filtered}
    assert not any(code in CLUB_LEAGUES for code in filtered_codes)
    assert len(filtered) == 2  # Only WC and ECQ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
