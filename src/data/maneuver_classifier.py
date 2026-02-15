"""Classify detected satellite maneuvers into avoidance vs routine.

Enriches each maneuver with:
  - magnitude_class: micro/small/medium/large based on delta-v
  - constellation: starlink/oneweb/iridium/other
  - is_stationkeeping: regularity-based detection from maneuver history
  - likely_avoidance: heuristic combining all signals

These enrichments improve training label quality for PI-TFT fine-tuning
without changing the model's feature space.
"""

import re
import numpy as np
from datetime import datetime


# Delta-v magnitude bins (m/s)
MAGNITUDE_BINS = [
    ("micro", 0.0, 0.5),
    ("small", 0.5, 2.0),
    ("medium", 2.0, 10.0),
    ("large", 10.0, float("inf")),
]

# Constellation name patterns
CONSTELLATION_PATTERNS = [
    ("starlink", re.compile(r"STARLINK", re.IGNORECASE)),
    ("oneweb", re.compile(r"ONEWEB", re.IGNORECASE)),
    ("iridium", re.compile(r"IRIDIUM", re.IGNORECASE)),
]

# Stationkeeping regularity threshold (coefficient of variation of intervals)
STATIONKEEPING_CV_THRESHOLD = 0.3
MIN_HISTORY_FOR_SK = 3  # Need at least 3 past maneuvers to detect pattern


def classify_magnitude(delta_v_m_s: float) -> str:
    """Bin delta-v into magnitude class."""
    dv = abs(delta_v_m_s)
    for label, lo, hi in MAGNITUDE_BINS:
        if lo <= dv < hi:
            return label
    return "large"


def detect_constellation(name: str) -> str:
    """Identify constellation from satellite name."""
    for constellation, pattern in CONSTELLATION_PATTERNS:
        if pattern.search(name):
            return constellation
    return "other"


def detect_stationkeeping(history: list[dict]) -> bool:
    """Detect stationkeeping from regularity of past maneuver intervals.

    If the coefficient of variation (std/mean) of time intervals between
    consecutive maneuvers is below threshold, it's likely stationkeeping.

    Args:
        history: Past maneuver records for this NORAD ID, each with
                 'detected_at' ISO timestamp.

    Returns:
        True if maneuver pattern suggests stationkeeping.
    """
    if not history or len(history) < MIN_HISTORY_FOR_SK:
        return False

    # Parse timestamps and sort
    timestamps = []
    for h in history:
        ts_str = h.get("detected_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps.append(ts.timestamp())
        except (ValueError, TypeError):
            continue

    if len(timestamps) < MIN_HISTORY_FOR_SK:
        return False

    timestamps.sort()
    intervals = np.diff(timestamps)

    if len(intervals) < 2:
        return False

    mean_interval = np.mean(intervals)
    if mean_interval <= 0:
        return False

    cv = np.std(intervals) / mean_interval
    return cv < STATIONKEEPING_CV_THRESHOLD


def classify_maneuver(maneuver: dict, history: list[dict] = None) -> dict:
    """Classify a detected maneuver with enrichment flags.

    Args:
        maneuver: Maneuver dict from detect_maneuvers() with keys:
                  norad_id, name, delta_v_m_s, delta_a_m, etc.
        history: Past maneuver records for same NORAD ID (optional).

    Returns:
        Dict with enrichment fields added to the original maneuver.
    """
    delta_v = maneuver.get("delta_v_m_s", 0.0)
    name = maneuver.get("name", "")

    magnitude_class = classify_magnitude(delta_v)
    constellation = detect_constellation(name)
    is_sk = detect_stationkeeping(history) if history else False

    # Likely avoidance heuristic
    likely_avoidance = False

    if not is_sk and magnitude_class in ("micro", "small") and delta_v < 5.0:
        likely_avoidance = True

    # Starlink CAMs are typically very small (< 1 m/s)
    if constellation == "starlink" and delta_v < 1.0:
        likely_avoidance = True

    enriched = dict(maneuver)
    enriched.update({
        "magnitude_class": magnitude_class,
        "constellation": constellation,
        "is_stationkeeping": is_sk,
        "likely_avoidance": likely_avoidance,
        "enrichment_version": 1,
        # Phase B/C defaults — overwritten later if data is available
        "has_cdm": False,
        "cdm_pc": None,
        "cdm_miss_distance_km": None,
        "counterfactual_min_distance_km": None,
        "would_have_collided": False,
        "counterfactual_closest_norad": None,
    })
    return enriched
