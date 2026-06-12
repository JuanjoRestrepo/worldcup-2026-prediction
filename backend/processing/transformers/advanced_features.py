"""Advanced football analytics features for the World Cup prediction engine.

Implements industry-standard football analytics features derivable from
score-level historical data, following practices used by Opta, FBRef, and
professional sports analytics teams:

  - Time-decayed form (EWMA, α=0.3) — exponential recency weighting
  - Head-to-head (H2H) record — direct confrontation history
  - Clean sheet rate — defensive solidity proxy
  - Tournament pressure score — contextual importance weighting
  - Rest days — fatigue proxy from match cadence
  - Confederation strength index — regional competitive context
  - Goals scoring variance — attacking volatility signal

Leakage prevention: All rolling/EWMA computations use .shift(1) prior
to the aggregation step, identical to the convention in rolling_features.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exponential decay factor for EWMA form (industry standard α = 0.3)
EWMA_ALPHA: float = 0.3

# Rolling window for clean-sheet rate and variance (10 matches → stable estimate)
LONG_WINDOW: int = 10

# Short window used by upstream rolling_features.py
SHORT_WINDOW: int = 5

# Tournament pressure scores — calibrated against historical draw-rate studies
# (Caley 2015; FiveThirtyEight WC model 2018)
TOURNAMENT_PRESSURE: dict[str, float] = {
    "fifa world cup": 1.0,
    "copa america": 0.90,
    "uefa euro": 0.90,
    "africa cup of nations": 0.85,
    "afc asian cup": 0.80,
    "concacaf gold cup": 0.75,
    "fifa confederations cup": 0.75,
    "uefa nations league": 0.65,
    "fifa world cup qualification": 0.70,
    "uefa euro qualification": 0.65,
    "friendly": 0.20,
}

# Confederation membership — used for confederation strength index
CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Germany": "UEFA",
    "France": "UEFA",
    "Spain": "UEFA",
    "Italy": "UEFA",
    "England": "UEFA",
    "Netherlands": "UEFA",
    "Portugal": "UEFA",
    "Belgium": "UEFA",
    "Croatia": "UEFA",
    "Denmark": "UEFA",
    "Switzerland": "UEFA",
    "Poland": "UEFA",
    "Czech Republic": "UEFA",
    "Serbia": "UEFA",
    "Hungary": "UEFA",
    "Romania": "UEFA",
    "Turkey": "UEFA",
    "Ukraine": "UEFA",
    "Austria": "UEFA",
    "Scotland": "UEFA",
    "Wales": "UEFA",
    "Sweden": "UEFA",
    "Norway": "UEFA",
    "Russia": "UEFA",
    "Greece": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Albania": "UEFA",
    "North Macedonia": "UEFA",
    "Montenegro": "UEFA",
    "Kosovo": "UEFA",
    "Georgia": "UEFA",
    "Finland": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL",
    "Argentina": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF",
    "Mexico": "CONCACAF",
    "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Panama": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "El Salvador": "CONCACAF",
    "Haiti": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    # CAF
    "Morocco": "CAF",
    "Senegal": "CAF",
    "Nigeria": "CAF",
    "Cameroon": "CAF",
    "Ghana": "CAF",
    "Algeria": "CAF",
    "Ivory Coast": "CAF",
    "Egypt": "CAF",
    "South Africa": "CAF",
    "Tunisia": "CAF",
    "Mali": "CAF",
    "Burkina Faso": "CAF",
    "Democratic Republic of the Congo": "CAF",
    # AFC
    "Japan": "AFC",
    "South Korea": "AFC",
    "Australia": "AFC",
    "Iran": "AFC",
    "Saudi Arabia": "AFC",
    "Qatar": "AFC",
    "China": "AFC",
    "United Arab Emirates": "AFC",
    "Iraq": "AFC",
    "Jordan": "AFC",
    # OFC
    "New Zealand": "OFC",
}


def _tournament_pressure_score(tournament: str) -> float:
    """Assign a tournament pressure scalar to a raw tournament string."""
    t = tournament.strip().lower()
    for key, score in TOURNAMENT_PRESSURE.items():
        if key in t:
            return score
    # Default for unrecognised tournaments: qualifier-level pressure
    return 0.55


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all advanced football analytics features.

    Args:
        df: Gold-layer DataFrame already containing ELO and basic rolling
            features (output of compute_rolling_features + compute_elo).
            Must be sorted ascending by date.

    Returns:
        DataFrame with additional advanced feature columns appended.

    Note:
        All temporal computations use .shift(1) before aggregation to
        prevent any current-match data leakage into the feature vectors.
    """
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Computing advanced football analytics features...")

    # ------------------------------------------------------------------ #
    # Feature A: Tournament Pressure Score                                 #
    # (static per-match contextual weight — no rolling, no leakage)        #
    # ------------------------------------------------------------------ #
    df["tournament_pressure_score"] = df["tournament"].apply(_tournament_pressure_score)
    logger.info("  ✅ Tournament pressure score computed")

    # ------------------------------------------------------------------ #
    # Feature B: Rest Days (days since last match — fatigue proxy)         #
    # ------------------------------------------------------------------ #
    home_last = df.groupby("homeTeam")["date"].shift(1)
    away_last = df.groupby("awayTeam")["date"].shift(1)
    df["home_days_since_last_match"] = (df["date"] - home_last).dt.days.fillna(30.0)
    df["away_days_since_last_match"] = (df["date"] - away_last).dt.days.fillna(30.0)
    logger.info("  ✅ Rest days features computed")

    # ------------------------------------------------------------------ #
    # Feature C: EWMA Time-Decayed Form (α = 0.3)                         #
    # Industry standard recency-weighted form (FBRef, Opta analytics)      #
    # ------------------------------------------------------------------ #

    # Build long-format unified view for global EWMA
    home_long = df[["date", "homeTeam", "homeGoals", "awayGoals"]].rename(
        columns={
            "homeTeam": "team",
            "homeGoals": "goals_for",
            "awayGoals": "goals_against",
        }
    )
    away_long = df[["date", "awayTeam", "awayGoals", "homeGoals"]].rename(
        columns={
            "awayTeam": "team",
            "awayGoals": "goals_for",
            "homeGoals": "goals_against",
        }
    )
    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    # EWMA with shift(1) — leakage prevention
    long_df["ewma_goals_for"] = long_df.groupby("team")["goals_for"].transform(
        lambda s: s.shift(1).ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    )
    long_df["ewma_goals_against"] = long_df.groupby("team")["goals_against"].transform(
        lambda s: s.shift(1).ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    )

    # Merge EWMA back onto the original match-level frame
    # Home team EWMA
    home_ewma = long_df.iloc[: len(df)][
        ["ewma_goals_for", "ewma_goals_against"]
    ].reset_index(drop=True)
    away_ewma = long_df.iloc[len(df) :][
        ["ewma_goals_for", "ewma_goals_against"]
    ].reset_index(drop=True)

    df["home_ewma_goals"] = home_ewma["ewma_goals_for"].values
    df["home_ewma_conceded"] = home_ewma["ewma_goals_against"].values
    df["away_ewma_goals"] = away_ewma["ewma_goals_for"].values
    df["away_ewma_conceded"] = away_ewma["ewma_goals_against"].values
    logger.info("  ✅ EWMA time-decayed form features (α=%.1f) computed", EWMA_ALPHA)

    # ------------------------------------------------------------------ #
    # Feature D: Clean Sheet Rate (last 10 matches)                        #
    # Defensive solidity — direct draw-propensity signal                   #
    # ------------------------------------------------------------------ #
    long_df["is_clean_sheet"] = (long_df["goals_against"] == 0).astype(float)
    long_df["clean_sheet_rate"] = long_df.groupby("team")["is_clean_sheet"].transform(
        lambda s: s.shift(1).rolling(LONG_WINDOW, min_periods=1).mean()
    )

    home_cs = long_df.iloc[: len(df)][["clean_sheet_rate"]].reset_index(drop=True)
    away_cs = long_df.iloc[len(df) :][["clean_sheet_rate"]].reset_index(drop=True)

    df["home_clean_sheet_rate_last10"] = home_cs["clean_sheet_rate"].values
    df["away_clean_sheet_rate_last10"] = away_cs["clean_sheet_rate"].values
    logger.info("  ✅ Clean sheet rate (last %d) computed", LONG_WINDOW)

    # ------------------------------------------------------------------ #
    # Feature E: Goals Scoring Variance (last 10)                          #
    # High variance → attacking/unpredictable; low variance → defensive    #
    # ------------------------------------------------------------------ #
    long_df["goals_variance"] = long_df.groupby("team")["goals_for"].transform(
        lambda s: s.shift(1).rolling(LONG_WINDOW, min_periods=2).std().fillna(0.0)
    )

    home_var = long_df.iloc[: len(df)][["goals_variance"]].reset_index(drop=True)
    away_var = long_df.iloc[len(df) :][["goals_variance"]].reset_index(drop=True)

    df["home_goals_variance_last10"] = home_var["goals_variance"].values
    df["away_goals_variance_last10"] = away_var["goals_variance"].values
    logger.info("  ✅ Goals scoring variance (last %d) computed", LONG_WINDOW)

    # ------------------------------------------------------------------ #
    # Feature F: Head-to-Head (H2H) Record (last 10 meetings)             #
    # Strongest contextual signal for tournament prediction                 #
    # ------------------------------------------------------------------ #
    df = _compute_h2h_features(df, window=10)
    logger.info("  ✅ Head-to-head record features computed (window=10)")

    # ------------------------------------------------------------------ #
    # Feature G: Confederation Strength Index                              #
    # Avg ELO of all teams in the same confederation (regional context)    #
    # ------------------------------------------------------------------ #
    df = _compute_confederation_strength(df)
    logger.info("  ✅ Confederation strength index computed")

    # ------------------------------------------------------------------ #
    # Feature H: Talent Differential (Transfermarkt)                       #
    # log(home_squad_value + 1) - log(away_squad_value + 1)                #
    # ------------------------------------------------------------------ #
    df = _compute_talent_differential(df)
    logger.info("  ✅ Talent differential feature computed")

    logger.info("Advanced feature engineering complete. Added 17 new columns.")
    return df


def _compute_h2h_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute head-to-head historical record between the two teams.

    For each match, looks back at the last `window` historical meetings
    between the same pair of teams (regardless of home/away role) and
    computes:
      - h2h_home_win_rate: win rate for the current home team
      - h2h_draw_rate: draw rate in H2H meetings
      - h2h_avg_goals: average combined goals per H2H match

    Leakage is prevented by only including matches with index < current.

    Args:
        df: Sorted DataFrame with homeTeam, awayTeam, homeGoals, awayGoals.
        window: Number of most recent H2H meetings to consider.

    Returns:
        DataFrame with H2H feature columns added.
    """
    # Normalise pair key so (A, B) and (B, A) map to the same key
    df["_pair_key"] = df.apply(
        lambda r: tuple(sorted([r["homeTeam"], r["awayTeam"]])), axis=1
    )

    h2h_home_win_rates: list[float] = []
    h2h_draw_rates: list[float] = []
    h2h_avg_goals: list[float] = []

    for idx, row in df.iterrows():
        pair = row["_pair_key"]
        home_team = row["homeTeam"]

        # All prior meetings between the same pair
        prior = df.loc[(df.index < idx) & (df["_pair_key"] == pair)].tail(window)

        if prior.empty:
            h2h_home_win_rates.append(float("nan"))
            h2h_draw_rates.append(float("nan"))
            h2h_avg_goals.append(float("nan"))
            continue

        # Compute from home_team's perspective
        def _result(r: pd.Series) -> float:
            if r["homeTeam"] == home_team:
                return float(np.sign(r["homeGoals"] - r["awayGoals"]))
            return float(np.sign(r["awayGoals"] - r["homeGoals"]))

        results = prior.apply(_result, axis=1)
        wins = (results == 1).sum()
        draws = (results == 0).sum()
        total_goals = (prior["homeGoals"] + prior["awayGoals"]).mean()

        h2h_home_win_rates.append(float(wins) / len(prior))
        h2h_draw_rates.append(float(draws) / len(prior))
        h2h_avg_goals.append(float(total_goals))

    df["h2h_home_win_rate"] = h2h_home_win_rates
    df["h2h_draw_rate"] = h2h_draw_rates
    df["h2h_avg_goals"] = h2h_avg_goals
    df = df.drop(columns=["_pair_key"], errors="ignore")
    return df


def _compute_confederation_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the confederation average ELO for home and away teams.

    Uses the `elo_home` / `elo_away` columns already present in `df` to
    derive a running average ELO per confederation, computed as a
    cross-sectional aggregate at the time of each match (no leakage —
    we use the values already present in the row, which reflect ELO
    ratings *before* the match outcome is known).

    Args:
        df: DataFrame with elo_home, elo_away, homeTeam, awayTeam.

    Returns:
        DataFrame with home_confederation_avg_elo and
        away_confederation_avg_elo added.
    """
    # Map teams to confederations
    home_conf = df["homeTeam"].map(CONFEDERATION_MAP).fillna("OTHER")
    away_conf = df["awayTeam"].map(CONFEDERATION_MAP).fillna("OTHER")

    # Compute per-match confederation average ELO using only teams
    # present in the dataset at that moment (broad cross-sectional approx)
    # We combine all elo values per confederation at the match date
    long_elo = pd.concat(
        [
            df[["date", "homeTeam", "elo_home"]].rename(
                columns={"homeTeam": "team", "elo_home": "elo"}
            ),
            df[["date", "awayTeam", "elo_away"]].rename(
                columns={"awayTeam": "team", "elo_away": "elo"}
            ),
        ],
        ignore_index=True,
    )
    long_elo["confederation"] = long_elo["team"].map(CONFEDERATION_MAP).fillna("OTHER")

    # Rolling mean ELO per confederation (expanding window for stability)
    conf_elo = (
        long_elo.sort_values("date")
        .groupby("confederation")["elo"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    long_elo["conf_avg_elo"] = conf_elo.values

    # Take the latest confederation ELO per match date
    conf_snapshot = (
        long_elo.sort_values("date")
        .groupby(["date", "confederation"])["conf_avg_elo"]
        .last()
        .reset_index()
    )
    conf_lookup = conf_snapshot.set_index(["date", "confederation"])["conf_avg_elo"]

    def _lookup_conf_elo(date: pd.Timestamp, conf: str) -> float:
        try:
            return float(conf_lookup.loc[(date, conf)])
        except KeyError:
            return float("nan")

    df["home_confederation_avg_elo"] = [
        _lookup_conf_elo(row["date"], home_conf.iloc[i])
        for i, (_, row) in enumerate(df.iterrows())
    ]
    df["away_confederation_avg_elo"] = [
        _lookup_conf_elo(row["date"], away_conf.iloc[i])
        for i, (_, row) in enumerate(df.iterrows())
    ]
    return df


def _compute_talent_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the log talent differential using static Transfermarkt squad values.

    Reads data/bronze/transfermarkt_static.csv, fuzzy matches to canonical teams,
    and applies log(home_value + 1) - log(away_value + 1).
    """
    from pathlib import Path

    from thefuzz import process

    tm_path = Path("data/bronze/transfermarkt_static.csv")
    if not tm_path.exists():
        logger.warning(
            "transfermarkt_static.csv not found. Filling talent features with 0."
        )
        df["home_squad_value"] = 0.0
        df["away_squad_value"] = 0.0
        df["talent_differential"] = 0.0
        return df

    tm_df = pd.read_csv(tm_path)
    # Basic data cleaning and prep
    tm_df["Total_Value_Num"] = pd.to_numeric(
        tm_df["Total_Value_Num"], errors="coerce"
    ).fillna(0.0)

    # Extract unique canonical teams from the dataframe
    canonical_teams = list(set(df["homeTeam"].unique()) | set(df["awayTeam"].unique()))
    source_teams = tm_df["Nation"].unique().tolist()

    # Fuzzy match TM teams to Kaggle canonical teams
    mapping = {}
    for src in source_teams:
        match, score = process.extractOne(src, canonical_teams)
        if score >= 85:
            mapping[src] = match

    tm_df["canonical_team"] = tm_df["Nation"].map(mapping)
    # Keep only matched teams and set index
    tm_df = tm_df.dropna(subset=["canonical_team"]).set_index("canonical_team")

    # Lookup dictionary
    val_dict = tm_df["Total_Value_Num"].to_dict()

    # Map to df using a baseline minimal value (e.g., 1M) for unmatched teams
    BASELINE_VALUE = 1_000_000.0

    df["home_squad_value"] = df["homeTeam"].map(val_dict).fillna(BASELINE_VALUE)
    df["away_squad_value"] = df["awayTeam"].map(val_dict).fillna(BASELINE_VALUE)

    # Calculate Log Talent Differential
    # log1p safely handles log(1 + x)
    df["talent_differential"] = np.log1p(df["home_squad_value"]) - np.log1p(
        df["away_squad_value"]
    )

    return df
