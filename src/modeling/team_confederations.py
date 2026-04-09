"""Best-effort confederation lookup for national teams used in evaluation reports."""

from __future__ import annotations

import unicodedata

from src.config.team_aliases import normalize_team_name


def _normalize_team_key(team_name: str) -> str:
    normalized = normalize_team_name(team_name).strip()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.casefold()


CONFEDERATION_TEAMS: dict[str, set[str]] = {
    "AFC": {
        "afghanistan", "australia", "bahrain", "bangladesh", "bhutan", "brunei",
        "cambodia", "china pr", "chinese taipei", "taiwan", "guam", "hong kong",
        "india", "indonesia", "iran", "iraq", "japan", "jordan", "kuwait",
        "kyrgyzstan", "laos", "lebanon", "macau", "macao", "malaysia",
        "maldives", "mongolia", "myanmar", "nepal", "north korea", "oman",
        "pakistan", "palestine", "philippines", "qatar", "saudi arabia",
        "singapore", "south korea", "sri lanka", "syria", "tajikistan",
        "thailand", "timor-leste", "turkmenistan", "united arab emirates",
        "uzbekistan", "vietnam", "yemen",
    },
    "CAF": {
        "algeria", "angola", "benin", "botswana", "burkina faso", "burundi",
        "cameroon", "cape verde", "central african republic", "chad", "comoros",
        "congo", "dr congo", "djibouti", "egypt", "equatorial guinea", "eritrea",
        "eswatini", "ethiopia", "gabon", "gambia", "ghana", "guinea",
        "guinea-bissau", "ivory coast", "cote d'ivoire", "kenya", "lesotho",
        "liberia", "libya", "madagascar", "malawi", "mali", "mauritania",
        "mauritius", "morocco", "mozambique", "namibia", "niger", "nigeria",
        "reunion", "rwanda", "sao tome and principe", "senegal", "seychelles",
        "sierra leone", "somalia", "south africa", "south sudan", "sudan",
        "tanzania", "togo", "tunisia", "uganda", "zambia", "zimbabwe",
    },
    "CONCACAF": {
        "anguilla", "antigua and barbuda", "aruba", "bahamas", "barbados",
        "belize", "bermuda", "bonaire", "british virgin islands", "canada",
        "cayman islands", "costa rica", "cuba", "curacao", "curaçao", "dominica",
        "dominican republic", "el salvador", "french guiana", "grenada",
        "guadeloupe", "guatemala", "guyana", "haiti", "honduras", "jamaica",
        "martinique", "mexico", "montserrat", "nicaragua", "panama",
        "puerto rico", "saint kitts and nevis", "saint lucia", "saint martin",
        "saint vincent and the grenadines", "suriname", "trinidad and tobago",
        "turks and caicos islands", "united states", "us virgin islands",
        "greenland",
    },
    "CONMEBOL": {
        "argentina", "bolivia", "brazil", "chile", "colombia", "ecuador",
        "paraguay", "peru", "uruguay", "venezuela",
    },
    "OFC": {
        "american samoa", "cook islands", "fiji", "new caledonia", "new zealand",
        "papua new guinea", "samoa", "solomon islands", "tahiti", "tonga",
        "vanuatu",
    },
    "UEFA": {
        "albania", "andorra", "armenia", "austria", "azerbaijan", "belarus",
        "belgium", "bosnia and herzegovina", "bulgaria", "croatia", "cyprus",
        "czech republic", "denmark", "england", "estonia", "faroe islands",
        "finland", "france", "georgia", "germany", "gibraltar", "greece",
        "guernsey", "hungary", "iceland", "israel", "italy", "jersey",
        "kazakhstan", "kosovo", "latvia", "liechtenstein", "lithuania",
        "luxembourg", "malta", "moldova", "montenegro", "netherlands",
        "north macedonia", "northern ireland", "norway", "poland", "portugal",
        "republic of ireland", "romania", "russia", "san marino", "scotland",
        "serbia", "slovakia", "slovenia", "spain", "sweden", "switzerland",
        "turkey", "ukraine", "wales", "isle of man", "alderney",
    },
}

REGIONAL_NON_FIFA_TEAMS = {
    "aymara", "basque country", "biafra", "chagos islands", "chameria",
    "elba island", "falkland islands", "froya", "frøya", "gozo", "hitra",
    "hmong", "isle of wight",
}


def get_team_confederation(team_name: str) -> str:
    """Resolve a team to its confederation, or a fallback bucket."""
    normalized_key = _normalize_team_key(team_name)
    if normalized_key in REGIONAL_NON_FIFA_TEAMS:
        return "Regional/NonFIFA"

    for confederation, teams in CONFEDERATION_TEAMS.items():
        if normalized_key in teams:
            return confederation
    return "Unknown"
