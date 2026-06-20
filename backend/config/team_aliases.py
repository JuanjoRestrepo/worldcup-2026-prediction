"""Team name aliases and normalization for prediction requests."""

# Mapping of common aliases/abbreviations to canonical team names.
# All keys MUST be lowercase (spaces replaced by underscores) for case-insensitive matching.
# Canonical values must match exactly the names in the gold feature dataset.
TEAM_ALIASES: dict[str, str] = {
    # ── North America ─────────────────────────────────────────────────────────
    "usa": "United States",
    "us": "United States",
    "united_states": "United States",
    "usmnt": "United States",
    # ── Europe ────────────────────────────────────────────────────────────────
    "uk": "England",
    "britain": "England",
    "czechoslovakia": "Czech Republic",
    "czech": "Czech Republic",
    "czechia": "Czech Republic",  # FIFA 2026 uses "Czechia"
    "holland": "Netherlands",
    "turkiye": "Turkey",  # FIFA 2026 uses "Turkiye"
    "turkey": "Turkey",
    "bosnia": "Bosnia and Herzegovina",
    "bih": "Bosnia and Herzegovina",
    # ── Asia ──────────────────────────────────────────────────────────────────
    "korea": "South Korea",
    "south_korea": "South Korea",
    "southkorea": "South Korea",
    "sk": "South Korea",
    "korea_republic": "South Korea",  # FIFA 2026 uses "Korea Republic"
    "republic_of_korea": "South Korea",
    "ksa": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "saudi_arabia": "Saudi Arabia",
    "ir_iran": "Iran",  # FIFA 2026 uses "IR Iran"
    "iran_islamic_republic": "Iran",
    # ── Africa ────────────────────────────────────────────────────────────────
    # CRITICAL: model feature dataset uses "Ivory Coast", NOT "Cote d'Ivoire"
    "ivory_coast": "Ivory Coast",
    "cote_d_ivoire": "Ivory Coast",
    "coted_ivoire": "Ivory Coast",
    "cote_divoire": "Ivory Coast",
    "cote_d'ivoire": "Ivory Coast",  # FIFA 2026 uses "Cote d'Ivoire"
    "cabo_verde": "Cape Verde",  # FIFA 2026 uses "Cabo Verde"
    "cape_verde": "Cape Verde",
    "congo_dr": "DR Congo",  # FIFA 2026 uses "Congo DR"
    "dr_congo": "DR Congo",
    "democratic_republic_congo": "DR Congo",
    "curacao": "Curaçao",  # FIFA 2026 uses "Curacao"
    # ── South America ─────────────────────────────────────────────────────────
    "brasil": "Brazil",
    "br": "Brazil",
    "arg": "Argentina",
    "ch": "Chile",
    "col": "Colombia",
    "urug": "Uruguay",
    "par": "Paraguay",
    # ── Oceania ───────────────────────────────────────────────────────────────
    "aus": "Australia",
    "nz": "New Zealand",
    "new_zealand": "New Zealand",
}


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name using alias mapping.

    Converts common abbreviations and FIFA official name variants to canonical
    team names that match the gold feature dataset.
    Case-insensitive matching. Returns original stripped name if no alias found.

    Args:
        team_name: Raw team name from request

    Returns:
        Normalized team name (or original if no alias matches)

    Examples:
        >>> normalize_team_name("USA")
        'United States'
        >>> normalize_team_name("Turkiye")
        'Turkey'
        >>> normalize_team_name("Korea Republic")
        'South Korea'
        >>> normalize_team_name("Brazil")
        'Brazil'
    """
    normalized_input = team_name.strip().lower().replace(" ", "_")
    return TEAM_ALIASES.get(normalized_input, team_name.strip())
