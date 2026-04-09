"""Unit tests for confederation lookup used in model evaluation reports."""

from src.modeling.team_confederations import get_team_confederation


def test_get_team_confederation_resolves_major_nations():
    assert get_team_confederation("Brazil") == "CONMEBOL"
    assert get_team_confederation("United States") == "CONCACAF"
    assert get_team_confederation("Japan") == "AFC"
    assert get_team_confederation("Germany") == "UEFA"


def test_get_team_confederation_flags_regional_sides():
    assert get_team_confederation("Basque Country") == "Regional/NonFIFA"
    assert get_team_confederation("Biafra") == "Regional/NonFIFA"
