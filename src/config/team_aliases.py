"""Team name aliases and normalization for prediction requests."""

# Mapping of common aliases/abbreviations to canonical team names
# All keys MUST be lowercase for case-insensitive matching
TEAM_ALIASES = {
    # North America
    "usa": "United States",
    "us": "United States",
    "united_states": "United States",
    "usmnt": "United States",  # US Men's National Team
    
    # Europe
    "england": "England",
    "uk": "England",
    "britain": "England",
    "czechoslovakia": "Czech Republic",
    "czech": "Czech Republic",
    "czechia": "Czech Republic",
    "holland": "Netherlands",
    "netherlands": "Netherlands",
    
    # Asia
    "korea": "South Korea",
    "south_korea": "South Korea",
    "southkorea": "South Korea",
    "sk": "South Korea",
    "ksa": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "saudi_arabia": "Saudi Arabia",
    
    # Africa
    "ivory_coast": "Côte d'Ivoire",
    "cote_d_ivoire": "Côte d'Ivoire",
    "coted_ivoire": "Côte d'Ivoire",
    
    # South America
    "brasil": "Brazil",
    "br": "Brazil",
    "argentina": "Argentina",
    "arg": "Argentina",
    "chile": "Chile",
    "ch": "Chile",
    "colombia": "Colombia",
    "col": "Colombia",
    "urug": "Uruguay",
    "uruguay": "Uruguay",
    "paraguay": "Paraguay",
    "par": "Paraguay",
    
    # Oceania
    "australia": "Australia",
    "aus": "Australia",
    "nz": "New Zealand",
    "new_zealand": "New Zealand",
}


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name using alias mapping.
    
    Converts common abbreviations and variations to canonical team names.
    Case-insensitive matching. Returns original if no alias found.
    
    Args:
        team_name: Raw team name from request
        
    Returns:
        Normalized team name (or original if no alias matches)
        
    Examples:
        >>> normalize_team_name("USA")
        "United States"
        >>> normalize_team_name("Brasil")
        "Brazil"
        >>> normalize_team_name("Brazil")
        "Brazil"
    """
    normalized_input = team_name.strip().lower().replace(" ", "_")
    result = TEAM_ALIASES.get(normalized_input, team_name.strip())
    return result
