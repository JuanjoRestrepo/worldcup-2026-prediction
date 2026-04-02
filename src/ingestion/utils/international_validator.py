"""Validator for international match data (selecciones nacionales only)."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# International competitions (selecciones nacionales)
INTERNATIONAL_COMPETITIONS = {
    # World Cup
    "WC",      # FIFA World Cup
    "WCQ",     # World Cup Qualifiers
    
    # Continental Championships
    "COPA",    # Copa América
    "COPAAQ",  # Copa América Qualifiers
    "EC",      # European Championship (UEFA Euro)
    "ECQ",     # Euro Qualifiers
    "ACN",     # African Cup of Nations
    "ANC",     # African Nations Cup Qualifiers
    "ACNQ",    # Africa Cup of Nations Qualifiers
    "AFC",     # AFC Asian Cup
    "AFCQ",    # AFC Asian Cup Qualifiers
    "OFC",     # OFC Nations Cup
    "OFCQ",    # OFC Qualifiers
    "CONCACAF",# CONCACAF Champions League (can include national teams)
    "CNL",     # CONCACAF Nations League
    
    # Qualifiers
    "UEFAQ",   # UEFA Qualifiers
    "CONMEBOLQ", # CONMEBOL Qualifiers
    
    # Friendlies
    "FR",      # Friendly Internationals
}

# Club league codes to EXCLUDE
CLUB_LEAGUES = {
    "PL",      # Premier League (England)
    "BL1",     # Bundesliga (Germany)
    "SA",      # Serie A (Italy)
    "PD",      # La Liga (Spain)
    "FL1",     # Ligue 1 (France)
    "PPL",     # Primeira Liga (Portugal)
    "DED",     # Eredivisie (Netherlands)
    "ELC",     # Championship (England)
    "BSA",     # Campeonato Brasileiro
    "RFPL",    # Russian Premier League
    "JPL",     # Japanese J-League
    "MLS",     # Major League Soccer
    "CL",      # UEFA Champions League
    "EL",      # UEFA Europa League
    "ECL",     # UEFA Europa Conference League
    "CLI",     # Copa Libertadores (South American clubs)
    "SUD",     # Sudamericana (South American clubs)
}


def validate_international_match(match: Dict) -> bool:
    """
    Validate if a match is from an international competition (selecciones nacionales).
    
    Args:
        match: Match dictionary from API
        
    Returns:
        True if match is international, False if it's a club league
    """
    try:
        # Get competition code
        competition_code = match.get("competition", {}).get("code", "")
        
        # Check if it's an international competition
        if competition_code in INTERNATIONAL_COMPETITIONS:
            return True
        
        # If code is in club leagues, it's NOT international
        if competition_code in CLUB_LEAGUES:
            return False
        
        # Unknown competition - log and skip
        competition_name = match.get("competition", {}).get("name", "Unknown")
        logger.warning(f"Unknown competition: {competition_name} ({competition_code})")
        return False
        
    except Exception as e:
        logger.error(f"Error validating match: {e}")
        return False


def filter_international_matches(matches: List[Dict]) -> List[Dict]:
    """
    Filter matches to keep only international competitions.
    
    Args:
        matches: List of match dictionaries
        
    Returns:
        Filtered list with only international matches
    """
    international_matches = []
    excluded_count = 0
    
    for match in matches:
        if validate_international_match(match) and is_national_team(match):
            international_matches.append(match)
        else:
            excluded_count += 1
    
    logger.info(f"Filtered {excluded_count} club league matches")
    logger.info(f"Kept {len(international_matches)} international matches")
    
    return international_matches

def is_national_team(match: Dict) -> bool:
    """
    Heuristic to detect national teams (avoid clubs).
    """
    try:
        home = match.get("homeTeam", {}).get("name", "")
        away = match.get("awayTeam", {}).get("name", "")

        return (
            "FC" not in home and
            "FC" not in away and
            "Club" not in home and
            "Club" not in away and
            "U" not in home and   # evita U21, U23
            "U" not in away and
            len(home.split()) <= 3 and
            len(away.split()) <= 3
        )
    except Exception:
        return False