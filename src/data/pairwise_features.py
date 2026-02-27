"""Pairwise TLE feature engineering for CDM-supervised risk prediction.

Computes ~30 features from a pair of CelesTrak TLE JSON records. These
features are derived entirely from free, public TLE data — no CDM access
needed at inference time.

Feature groups:
  - Object properties (altitude, eccentricity, inclination, etc.)
  - Pairwise geometry (deltas in orbital elements)
  - Collision geometry (coplanar angle, relative velocity estimate)
  - Shell density (nearby object counts from full TLE catalog)
  - Object classification (debris, payload, rocket body flags)
  - TLE freshness (age of each TLE in hours)

Used by:
  - scripts/train_pairwise.py (training data generation)
  - scripts/daily_predictions.py (inference scoring)
  - app/main.py (API endpoint)
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from src.data.maneuver_detector import (
    extract_orbital_elements,
    mean_motion_to_sma,
    EARTH_RADIUS_KM,
    MU_EARTH,
)

# Object type keywords for classification
_DEBRIS_KEYWORDS = {"DEB", "DEBRIS", "FRAG", "FRAGMENT"}
_ROCKET_BODY_KEYWORDS = {"R/B", "ROCKET", "STAGE", "KICK", "MOTOR", "BOOSTER"}


def _classify_object(name: str) -> tuple[bool, bool, bool]:
    """Classify a satellite by name into (is_debris, is_payload, is_rocket_body)."""
    upper = name.upper()
    is_debris = any(kw in upper for kw in _DEBRIS_KEYWORDS)
    is_rb = any(kw in upper for kw in _ROCKET_BODY_KEYWORDS)
    is_payload = not is_debris and not is_rb
    return is_debris, is_payload, is_rb


def _rcs_code_to_numeric(rcs: str) -> float:
    """Convert RCS size code to approximate cross-section in m^2."""
    rcs = str(rcs).upper().strip()
    if rcs == "LARGE":
        return 10.0
    elif rcs == "MEDIUM":
        return 1.0
    elif rcs == "SMALL":
        return 0.1
    return 0.5  # unknown


def _tle_age_hours(tle: dict, ref_time: Optional[datetime] = None) -> float:
    """Compute TLE age in hours from epoch to reference time."""
    epoch_str = tle.get("EPOCH", "")
    if not epoch_str:
        return 48.0  # default: 2 days old
    try:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                epoch = datetime.strptime(epoch_str, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return 48.0
        ref = ref_time or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        age = (ref - epoch).total_seconds() / 3600.0
        return max(0.0, age)
    except Exception:
        return 48.0


def _vis_viva_velocity(sma_km: float) -> float:
    """Circular orbital velocity (km/s) from vis-viva equation."""
    if sma_km <= 0:
        return 0.0
    return math.sqrt(MU_EARTH / sma_km)


def compute_pairwise_features(
    sat1: dict,
    sat2: dict,
    all_tles: Optional[list[dict]] = None,
    ref_time: Optional[datetime] = None,
) -> dict:
    """Compute ~30 features from two CelesTrak TLE JSON records.

    Args:
        sat1: First satellite TLE (CelesTrak JSON format).
        sat2: Second satellite TLE (CelesTrak JSON format).
        all_tles: Full TLE catalog for shell density features. If None,
                  density features are set to 0.
        ref_time: Reference time for TLE age computation.

    Returns:
        Dict of feature name -> float value.
    """
    e1 = extract_orbital_elements(sat1)
    e2 = extract_orbital_elements(sat2)

    alt1 = e1["altitude_km"]
    alt2 = e2["altitude_km"]
    inc1 = e1["inclination"]
    inc2 = e2["inclination"]
    raan1 = e1["raan"]
    raan2 = e2["raan"]
    ecc1 = e1["eccentricity"]
    ecc2 = e2["eccentricity"]
    mm1 = e1["mean_motion"]
    mm2 = e2["mean_motion"]
    sma1 = e1["sma_km"]
    sma2 = e2["sma_km"]

    argp1 = float(sat1.get("ARG_OF_PERICENTER", 0))
    argp2 = float(sat2.get("ARG_OF_PERICENTER", 0))

    # --- Object properties (10) ---
    features = {
        "altitude_1": alt1,
        "altitude_2": alt2,
        "eccentricity_1": ecc1,
        "eccentricity_2": ecc2,
        "inclination_1": inc1,
        "inclination_2": inc2,
        "mean_motion_1": mm1,
        "mean_motion_2": mm2,
    }

    # RCS codes
    rcs1 = sat1.get("RCS_SIZE", sat1.get("SAT1_RCS", ""))
    rcs2 = sat2.get("RCS_SIZE", sat2.get("SAT2_RCS", ""))
    features["rcs_size_1"] = _rcs_code_to_numeric(rcs1)
    features["rcs_size_2"] = _rcs_code_to_numeric(rcs2)

    # --- Pairwise geometry (6) ---
    features["delta_altitude_km"] = abs(alt1 - alt2)
    features["delta_inclination_deg"] = abs(inc1 - inc2)

    raan_diff = abs(raan1 - raan2)
    raan_diff = min(raan_diff, 360.0 - raan_diff)
    features["delta_raan_deg"] = raan_diff

    features["delta_ecc"] = abs(ecc1 - ecc2)

    argp_diff = abs(argp1 - argp2)
    argp_diff = min(argp_diff, 360.0 - argp_diff)
    features["delta_argp_deg"] = argp_diff

    features["delta_mean_motion"] = abs(mm1 - mm2)

    # --- Collision geometry (3) ---
    # Coplanar angle: combined measure of orbital plane difference
    features["coplanar_angle"] = abs(inc1 - inc2) + raan_diff

    # Relative velocity estimate from vis-viva
    v1 = _vis_viva_velocity(sma1)
    v2 = _vis_viva_velocity(sma2)
    # For crossing orbits, relative velocity depends on crossing angle
    # Approximation: v_rel ~ sqrt(v1^2 + v2^2 - 2*v1*v2*cos(theta))
    # where theta is the crossing angle from inclination + RAAN difference
    theta_rad = math.radians(min(features["coplanar_angle"], 180.0))
    v_rel = math.sqrt(max(0, v1**2 + v2**2 - 2 * v1 * v2 * math.cos(theta_rad)))
    features["relative_velocity_est_km_s"] = v_rel

    # Crossing angle (simplified: from inclination difference)
    features["crossing_angle_deg"] = min(
        abs(inc1 - inc2),
        abs(inc1 + inc2),
        abs(180 - inc1 - inc2),
    )

    # --- Shell density (3) ---
    if all_tles is not None:
        avg_alt = (alt1 + alt2) / 2.0
        n_within_10 = 0
        n_within_50 = 0
        for tle in all_tles:
            mm = float(tle.get("MEAN_MOTION", 0))
            if mm <= 0:
                continue
            sma = mean_motion_to_sma(mm)
            alt = sma - EARTH_RADIUS_KM
            d = abs(alt - avg_alt)
            if d < 10.0:
                n_within_10 += 1
            if d < 50.0:
                n_within_50 += 1
        features["n_objects_within_10km_alt"] = float(n_within_10)
        features["n_objects_within_50km_alt"] = float(n_within_50)
        # Shell congestion index: objects per km of altitude band
        features["shell_congestion_index"] = float(n_within_50) / 50.0
    else:
        features["n_objects_within_10km_alt"] = 0.0
        features["n_objects_within_50km_alt"] = 0.0
        features["shell_congestion_index"] = 0.0

    # --- Object classification (6) ---
    name1 = sat1.get("OBJECT_NAME", sat1.get("sat1_name", ""))
    name2 = sat2.get("OBJECT_NAME", sat2.get("sat2_name", ""))
    deb1, pay1, rb1 = _classify_object(name1)
    deb2, pay2, rb2 = _classify_object(name2)
    features["is_debris_1"] = float(deb1)
    features["is_debris_2"] = float(deb2)
    features["is_payload_1"] = float(pay1)
    features["is_payload_2"] = float(pay2)
    features["is_rocket_body_1"] = float(rb1)
    features["is_rocket_body_2"] = float(rb2)

    # --- TLE freshness (2) ---
    features["tle_age_hours_1"] = _tle_age_hours(sat1, ref_time)
    features["tle_age_hours_2"] = _tle_age_hours(sat2, ref_time)

    return features


# Canonical feature ordering for model training/inference
PAIRWISE_FEATURE_COLS = [
    "altitude_1", "altitude_2",
    "eccentricity_1", "eccentricity_2",
    "inclination_1", "inclination_2",
    "mean_motion_1", "mean_motion_2",
    "rcs_size_1", "rcs_size_2",
    "delta_altitude_km", "delta_inclination_deg", "delta_raan_deg",
    "delta_ecc", "delta_argp_deg", "delta_mean_motion",
    "coplanar_angle", "relative_velocity_est_km_s", "crossing_angle_deg",
    "n_objects_within_10km_alt", "n_objects_within_50km_alt", "shell_congestion_index",
    "is_debris_1", "is_debris_2",
    "is_payload_1", "is_payload_2",
    "is_rocket_body_1", "is_rocket_body_2",
    "tle_age_hours_1", "tle_age_hours_2",
]


def compute_batch_features(
    pairs: list[tuple[dict, dict]],
    all_tles: Optional[list[dict]] = None,
    ref_time: Optional[datetime] = None,
    precomputed_alts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Vectorized batch computation of pairwise features.

    For shell density features, precomputes the altitude array from all_tles
    once, then uses numpy broadcasting instead of per-pair Python loops.

    Args:
        pairs: List of (sat1_tle, sat2_tle) tuples.
        all_tles: Full TLE catalog for density features.
        ref_time: Reference time for TLE age.
        precomputed_alts: Pre-extracted altitude array for all_tles (optimization).

    Returns:
        DataFrame with PAIRWISE_FEATURE_COLS columns, one row per pair.
    """
    if not pairs:
        return pd.DataFrame(columns=PAIRWISE_FEATURE_COLS)

    # Precompute altitude array for density features
    if all_tles is not None and precomputed_alts is None:
        catalog_alts = np.zeros(len(all_tles))
        for i, tle in enumerate(all_tles):
            mm = float(tle.get("MEAN_MOTION", 0))
            if mm > 0:
                sma = mean_motion_to_sma(mm)
                catalog_alts[i] = sma - EARTH_RADIUS_KM
    elif precomputed_alts is not None:
        catalog_alts = precomputed_alts
    else:
        catalog_alts = None

    rows = []
    for sat1, sat2 in pairs:
        feats = compute_pairwise_features(sat1, sat2, all_tles=None, ref_time=ref_time)

        # Compute density features using vectorized altitude comparison
        if catalog_alts is not None:
            avg_alt = (feats["altitude_1"] + feats["altitude_2"]) / 2.0
            diffs = np.abs(catalog_alts - avg_alt)
            n10 = int(np.sum(diffs < 10.0))
            n50 = int(np.sum(diffs < 50.0))
            feats["n_objects_within_10km_alt"] = float(n10)
            feats["n_objects_within_50km_alt"] = float(n50)
            feats["shell_congestion_index"] = float(n50) / 50.0

        rows.append(feats)

    df = pd.DataFrame(rows, columns=PAIRWISE_FEATURE_COLS)
    # Fill any NaN with 0
    df = df.fillna(0.0)
    return df
