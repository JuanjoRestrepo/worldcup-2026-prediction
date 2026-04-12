"""Canonical tournament-to-segment routing for training and inference.

This is the **single source of truth** for segment detection across the
entire pipeline.  Both ``train.py`` (temporal backtesting) and
``predict.py`` (production inference) must import from here — never
duplicate the routing logic.

The mapping encodes domain knowledge from the global 0.45 uncertainty
threshold analysis: draw prediction noise concentrates in Friendlies
and some continental fixtures, while World Cup / Qualifiers are
structurally cleaner for the generalist.
"""

from __future__ import annotations

import pandas as pd

# ============================================================================
# Segment IDs — canonical string constants
# ============================================================================

SEGMENT_WORLDCUP = "worldcup"
SEGMENT_FRIENDLIES = "friendlies"
SEGMENT_QUALIFIERS = "qualifiers"
SEGMENT_CONTINENTAL = "continental"

ALL_SEGMENTS: frozenset[str] = frozenset(
    {SEGMENT_WORLDCUP, SEGMENT_FRIENDLIES, SEGMENT_QUALIFIERS, SEGMENT_CONTINENTAL}
)

# Metadata columns required for segment-aware ensemble routing.
# These ride alongside features during backtesting prediction but
# are excluded from model training (the ensemble's feature_names_in_
# filtering strips them automatically).
SEGMENT_METADATA_COLUMNS: list[str] = ["tournament"]

# ============================================================================
# Qualifier keywords — matched via substring in lowercase tournament name
# ============================================================================

_QUALIFIER_KEYWORDS: tuple[str, ...] = (
    "qualification",
    "qualifier",
    "playoff",
    "repechage",
)

# ============================================================================
# Continental tournament keywords — substring match in lowercase
# ============================================================================

_CONTINENTAL_KEYWORDS: tuple[str, ...] = (
    "copa am",
    "euro",
    "africa cup",
    "african cup",
    "asian cup",
    "confederations",
    "gold cup",
    "nations league",
)


def tournament_segment_detector(row: pd.Series) -> str | None:
    """Route a fixture to its segment based on tournament name.

    Returns a segment_id (e.g. ``"worldcup"``, ``"friendlies"``) used by
    ``SegmentAwareHybridDrawOverrideEnsemble`` to select per-segment
    uncertainty and conviction thresholds.

    Args:
        row: A pandas Series representing a single fixture.
             Must contain a ``tournament`` field (string or None).

    Returns:
        One of the canonical segment IDs, or ``None`` for unmatched
        tournaments that will fall back to default thresholds.

    Examples:
        >>> import pandas as pd
        >>> tournament_segment_detector(pd.Series({"tournament": "FIFA World Cup"}))
        'worldcup'
        >>> tournament_segment_detector(pd.Series({"tournament": "FIFA World Cup qualification"}))
        'qualifiers'
        >>> tournament_segment_detector(pd.Series({"tournament": "Friendly"}))
        'friendlies'
    """
    tournament = row.get("tournament")
    if not tournament or not isinstance(tournament, str):
        return None

    tournament_lower = tournament.lower()

    # --- World Cup (final tournament) — MUST exclude qualifications ------
    if "world cup" in tournament_lower and "qualification" not in tournament_lower:
        return SEGMENT_WORLDCUP

    # --- Friendlies -------------------------------------------------------
    if "friendly" in tournament_lower:
        return SEGMENT_FRIENDLIES

    # --- Qualifiers (any confederation) ------------------------------------
    if any(kw in tournament_lower for kw in _QUALIFIER_KEYWORDS):
        return SEGMENT_QUALIFIERS

    # --- Continental tournaments -------------------------------------------
    if any(kw in tournament_lower for kw in _CONTINENTAL_KEYWORDS):
        return SEGMENT_CONTINENTAL

    return None


def detect_match_segment(tournament: str | None) -> str | None:
    """Convenience wrapper for production inference (string-in, string-out).

    Unlike :func:`tournament_segment_detector` (which takes a pandas Series
    for scikit-learn compatibility), this function accepts a plain string —
    the format available in the API request layer.

    Both functions use the **exact same routing logic** — this wrapper
    simply constructs the Series internally.

    Args:
        tournament: Tournament name from the API request, or None.

    Returns:
        Canonical segment ID, or None for unmatched tournaments.
    """
    if not tournament:
        return None
    return tournament_segment_detector(pd.Series({"tournament": tournament}))
