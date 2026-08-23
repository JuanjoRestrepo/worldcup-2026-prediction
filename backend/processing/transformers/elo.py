"""ELO rating calculation for team strength estimation with time-decay.

Design decisions:
- Time-decay: Teams that are inactive for extended periods have their ELO
  regressed toward the mean (1500). This prevents stale ratings from inflating
  or deflating predictions for teams that return after long absences.
- Decay formula: ELO_decayed = mean + (ELO - mean) * decay_factor^(days/90)
  where decay_factor=0.97 means ~28% reversion over a year of inactivity.
- Leakage prevention: ELO stored BEFORE the match result is processed, so
  elo_home / elo_away represent pre-match ratings (no look-ahead bias).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

INITIAL_ELO: float = 1500.0
ELO_MEAN: float = 1500.0  # Global mean for regression target
DEFAULT_K_FACTOR: float = 20.0
ELO_DECAY_FACTOR: float = 0.97  # Per-90-day decay multiplier
ELO_DECAY_PERIOD_DAYS: int = 90  # Decay reference period (≈ one international window)


def get_tournament_k_factor(tournament: str | None, is_knockout: bool = False) -> float:
    """
    Determine FIFA-style ELO K-factor based on tournament importance.

    - World Cup Knockouts: 60.0
    - World Cup Group Stage: 50.0
    - Continental Finals (Euro, Copa America, AFCON): 40.0
    - World Cup & Continental Qualifiers / Nations League: 30.0
    - Friendlies & minor cups: 15.0
    """
    if not tournament:
        return DEFAULT_K_FACTOR

    t_lower = tournament.strip().lower()

    if any(token in t_lower for token in ("qualification", "qualifier", "wcq", "ecq", "nations league")):
        return 30.0

    is_wc = any(token in t_lower for token in ("world cup", "wc", "fifa"))
    if is_wc:
        return 60.0 if is_knockout else 50.0

    if any(token in t_lower for token in ("euro", "copa america", "africa cup", "asian cup", "championship")):
        return 40.0

    if "friendly" in t_lower:
        return 15.0

    return DEFAULT_K_FACTOR


def get_goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
    """Calculate ELO margin of victory multiplier based on goal difference."""
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    # For diff >= 3: 1.75 + (diff - 3) / 8.0
    return float(1.75 + (diff - 3) / 8.0)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for team A against team B using logistic ELO."""
    return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = DEFAULT_K_FACTOR,
    goal_margin_multiplier: float = 1.0,
) -> tuple[float, float]:
    """
    Update ELO ratings after a match result.

    Args:
        rating_a: Team A's current ELO.
        rating_b: Team B's current ELO.
        score_a: Match result for team A (1=win, 0.5=draw, 0=loss).
        k_factor: Match importance factor.
        goal_margin_multiplier: Multiplier for victory margin.

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    exp_a = expected_score(rating_a, rating_b)
    effective_k = k_factor * goal_margin_multiplier
    new_a = rating_a + effective_k * (score_a - exp_a)
    new_b = rating_b + effective_k * ((1.0 - score_a) - (1.0 - exp_a))
    return new_a, new_b


def _apply_inactivity_decay(
    rating: float,
    days_since_last_match: float,
) -> float:
    """
    Regress ELO toward the mean for inactive teams.

    A team that hasn't played in N days sees its rating pulled toward ELO_MEAN
    by decay_factor^(N / ELO_DECAY_PERIOD_DAYS). After ~365 days of inactivity
    the rating is ~28% reverted toward the mean, which avoids extreme outlier ELOs
    persisting for squads that are effectively rebuilding.

    Args:
        rating: Current ELO rating.
        days_since_last_match: Number of days since the team's last match.

    Returns:
        Decayed ELO rating.
    """
    if days_since_last_match <= 0:
        return rating
    periods = days_since_last_match / ELO_DECAY_PERIOD_DAYS
    decay_multiplier = ELO_DECAY_FACTOR**periods
    return float(ELO_MEAN + (rating - ELO_MEAN) * decay_multiplier)


def compute_elo(df: pd.DataFrame, apply_decay: bool = True) -> pd.DataFrame:
    """
    Compute pre-match ELO ratings for all teams across match history.

    Leakage is prevented by recording each team's ELO *before* the match outcome
    is processed. When apply_decay=True, each team's rating is regressed toward
    the mean proportional to their inactivity gap since the previous fixture.

    Args:
        df: DataFrame with columns: date, homeTeam, awayTeam, homeGoals, awayGoals.
            Optionally includes 'tournament' and 'is_knockout' for dynamic K-factors.
        apply_decay: Whether to apply time-decay for inactive teams (default: True).

    Returns:
        DataFrame with added columns: elo_home, elo_away, elo_diff.
    """
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    last_match_date: dict[str, pd.Timestamp] = {}
    elo_home_list: list[float] = []
    elo_away_list: list[float] = []

    has_tournament = "tournament" in df.columns
    has_knockout = "is_knockout" in df.columns

    for idx, row in df.iterrows():
        home: str = str(row["homeTeam"])
        away: str = str(row["awayTeam"])
        match_date: pd.Timestamp = pd.Timestamp(row["date"])

        # Retrieve or initialize ELO
        rating_home = ratings.get(home, INITIAL_ELO)
        rating_away = ratings.get(away, INITIAL_ELO)

        # Apply time-decay for inactivity before recording pre-match ELO
        if apply_decay:
            if home in last_match_date:
                days_inactive = (match_date - last_match_date[home]).days
                rating_home = _apply_inactivity_decay(rating_home, float(days_inactive))
            if away in last_match_date:
                days_inactive = (match_date - last_match_date[away]).days
                rating_away = _apply_inactivity_decay(rating_away, float(days_inactive))

        # Record pre-match ELO (leakage-free)
        elo_home_list.append(rating_home)
        elo_away_list.append(rating_away)

        # Determine match outcome
        home_goals: int = int(row["homeGoals"])
        away_goals: int = int(row["awayGoals"])
        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals < away_goals:
            score_home = 0.0
        else:
            score_home = 0.5

        tournament_val = str(row["tournament"]) if has_tournament else None
        is_ko_val = bool(row["is_knockout"]) if has_knockout else False

        k_factor = get_tournament_k_factor(tournament_val, is_knockout=is_ko_val)
        margin_mult = get_goal_margin_multiplier(home_goals, away_goals)

        # Update post-match ratings
        new_home, new_away = update_elo(
            rating_home,
            rating_away,
            score_home,
            k_factor=k_factor,
            goal_margin_multiplier=margin_mult,
        )
        ratings[home] = new_home
        ratings[away] = new_away
        last_match_date[home] = match_date
        last_match_date[away] = match_date

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    if apply_decay:
        logger.debug(
            "ELO computed with dynamic K-factors & time-decay for %d matches",
            len(df),
        )

    return df

