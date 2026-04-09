"""Tests for team alias normalization."""

from __future__ import annotations


from src.config.team_aliases import normalize_team_name, TEAM_ALIASES


class TestTeamAliases:
    """Test team name alias functionality."""
    
    def test_normalize_usa(self):
        """Test USA alias."""
        assert normalize_team_name("USA") == "United States"
        assert normalize_team_name("usa") == "United States"
        assert normalize_team_name("uS") == "United States"
    
    def test_normalize_brazilian_teams(self):
        """Test Brazilian team aliases."""
        assert normalize_team_name("Brasil") == "Brazil"
        assert normalize_team_name("br") == "Brazil"
        assert normalize_team_name("BR") == "Brazil"
    
    def test_normalize_european_teams(self):
        """Test European team aliases."""
        assert normalize_team_name("Holland") == "Netherlands"
        assert normalize_team_name("Czech") == "Czech Republic"
        assert normalize_team_name("Czechia") == "Czech Republic"
    
    def test_normalize_asian_teams(self):
        """Test Asian team aliases."""
        assert normalize_team_name("Korea") == "South Korea"
        assert normalize_team_name("South_Korea") == "South Korea"
        assert normalize_team_name("SK") == "South Korea"
    
    def test_normalize_african_teams(self):
        """Test African team aliases."""
        assert normalize_team_name("Ivory_Coast") == "Côte d'Ivoire"
        assert normalize_team_name("Cote_d_Ivoire") == "Côte d'Ivoire"
    
    def test_normalize_no_alias_returns_original(self):
        """Test that teams without aliases return original name."""
        # These have no aliases defined, so should return as-is
        assert normalize_team_name("Brazil") == "Brazil"
        assert normalize_team_name("England") == "England"
        assert normalize_team_name("France") == "France"
    
    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_team_name("USA") == normalize_team_name("usa")
        assert normalize_team_name("Holland") == normalize_team_name("HOLLAND")
        assert normalize_team_name("Korea") == normalize_team_name("korea")
    
    def test_normalize_preserves_spacing(self):
        """Test that leading/trailing spaces are stripped."""
        assert normalize_team_name(" USA ") == "United States"
        assert normalize_team_name("  Brazil  ") == "Brazil"
    
    def test_normalize_underscores_to_spaces(self):
        """Test that underscores are handled in lookup."""
        result = normalize_team_name("South_Korea")
        assert result == "South Korea"
    
    def test_all_aliases_have_valid_targets(self):
        """Ensure all alias mappings point to valid strings."""
        for alias, target in TEAM_ALIASES.items():
            assert isinstance(alias, str), f"Alias key must be string: {alias}"
            assert isinstance(target, str), f"Alias target must be string: {target}"
            assert len(alias) > 0, "Alias cannot be empty"
            assert len(target) > 0, f"Target for alias {alias} cannot be empty"
    
    def test_no_alias_to_empty_string(self):
        """Ensure no alias maps to empty string."""
        for target in TEAM_ALIASES.values():
            assert target.strip() != "", "No alias should map to empty or whitespace-only string"
    
    def test_common_misspellings(self):
        """Test support for common misspellings."""
        # US instead of USA
        result = normalize_team_name("us")
        assert result == "United States"
        
        # UK instead of England
        result = normalize_team_name("uk")
        assert result == "England"
    
    def test_alias_case_variations(self):
        """Test various case combinations."""
        test_cases = [
            ("USA", "United States"),
            ("usa", "United States"),
            ("Usa", "United States"),
            ("uSa", "United States"),
        ]
        for input_name, expected in test_cases:
            assert normalize_team_name(input_name) == expected, f"Failed for {input_name}"
