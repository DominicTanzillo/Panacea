"""SGP4 counterfactual propagation — "what if no maneuver?" simulation.

For each likely-avoidance maneuver, propagates the pre-maneuver TLE forward
to estimate whether a close approach would have occurred. This generates
counterfactual "would-have-collided" labels for training enrichment.

Uses the sgp4 library for efficient satellite propagation.
"""

import math
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    from sgp4.api import Satrec, WGS72
    from sgp4 import exporter
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

# Earth parameters
EARTH_RADIUS_KM = 6378.137

# Counterfactual thresholds
COLLISION_THRESHOLD_KM = 1.0    # "Would have collided" if closer than this
NEARBY_ALT_BAND_KM = 50.0      # Altitude proximity for neighbor selection
NEARBY_RAAN_BAND_DEG = 30.0    # RAAN proximity for neighbor selection


def celestrak_json_to_satrec(tle_json: dict) -> "Satrec":
    """Convert a CelesTrak GP JSON record to an sgp4 Satrec object.

    CelesTrak JSON includes TLE_LINE1/TLE_LINE2 when available. Falls
    back to constructing from orbital elements via sgp4init().

    Args:
        tle_json: CelesTrak GP JSON dict with orbital elements.

    Returns:
        sgp4 Satrec object ready for propagation.

    Raises:
        ImportError: If sgp4 is not installed.
        ValueError: If TLE data is insufficient.
    """
    if not SGP4_AVAILABLE:
        raise ImportError("sgp4 library is required: pip install sgp4")

    # Prefer TLE lines if available (most reliable)
    line1 = tle_json.get("TLE_LINE1", "")
    line2 = tle_json.get("TLE_LINE2", "")
    if line1 and line2:
        return Satrec.twoline2rv(line1, line2)

    # Construct from JSON orbital elements using sgp4init
    satrec = Satrec()

    # Parse epoch
    epoch_str = tle_json.get("EPOCH", "")
    if not epoch_str:
        raise ValueError("No EPOCH in TLE JSON")

    epoch_dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
    if epoch_dt.tzinfo is None:
        epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)

    # Convert to Julian date pair for sgp4
    year = epoch_dt.year
    mon = epoch_dt.month
    day = epoch_dt.day
    hr = epoch_dt.hour
    minute = epoch_dt.minute
    sec = epoch_dt.second + epoch_dt.microsecond / 1e6

    # sgp4init expects elements in specific units
    no_kozai = float(tle_json.get("MEAN_MOTION", 0)) * (2.0 * math.pi / 1440.0)  # rev/day -> rad/min
    ecco = float(tle_json.get("ECCENTRICITY", 0))
    inclo = math.radians(float(tle_json.get("INCLINATION", 0)))
    nodeo = math.radians(float(tle_json.get("RA_OF_ASC_NODE", 0)))
    argpo = math.radians(float(tle_json.get("ARG_OF_PERICENTER", 0)))
    mo = math.radians(float(tle_json.get("MEAN_ANOMALY", 0)))
    bstar = float(tle_json.get("BSTAR", 0))
    norad_id = int(tle_json.get("NORAD_CAT_ID", 0))

    # Epoch in Julian date
    jd_base = _datetime_to_jd(epoch_dt)
    epoch_jd = jd_base
    # sgp4init epoch is minutes since 1949-12-31 00:00 UTC
    # But the Python API uses (jdsatepoch, jdsatepochF) pair
    jd_whole = int(epoch_jd)
    jd_frac = epoch_jd - jd_whole

    satrec.sgp4init(
        WGS72,           # gravity model
        'i',             # 'a' = old AFSPC mode, 'i' = improved
        norad_id,        # NORAD catalog number
        (epoch_jd - 2433281.5),  # epoch in days since 1949 Dec 31 00:00 UT
        bstar,           # BSTAR drag term
        0.0,             # ndot (not used in sgp4init 'i' mode)
        0.0,             # nddot (not used)
        ecco,            # eccentricity
        argpo,           # argument of perigee (radians)
        inclo,           # inclination (radians)
        mo,              # mean anomaly (radians)
        no_kozai,        # mean motion (radians/minute)
        nodeo,           # RAAN (radians)
    )

    return satrec


def _datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
    return jd


def _propagate_positions(satrec: "Satrec", start_jd: float, hours: float, step_min: float) -> np.ndarray:
    """Propagate a satellite and return position array (N x 3) in km.

    Returns empty array if propagation fails.
    """
    n_steps = int(hours * 60 / step_min) + 1
    positions = []

    for i in range(n_steps):
        minutes_since_epoch = (start_jd - satrec.jdsatepoch - satrec.jdsatepochF) * 1440.0 + i * step_min
        e, r, v = satrec.sgp4(satrec.jdsatepoch, satrec.jdsatepochF + minutes_since_epoch / 1440.0)
        if e != 0:
            continue
        positions.append(r)

    if not positions:
        return np.array([]).reshape(0, 3)
    return np.array(positions)


def find_nearby_satellites(
    maneuvered_tle: dict,
    all_tles: list[dict],
    alt_band_km: float = NEARBY_ALT_BAND_KM,
    raan_band_deg: float = NEARBY_RAAN_BAND_DEG,
) -> list[dict]:
    """Find satellites in similar orbital shell to the maneuvered object."""
    from src.data.maneuver_detector import mean_motion_to_sma, sma_to_altitude

    norad_id = int(maneuvered_tle.get("NORAD_CAT_ID", 0))
    mm = float(maneuvered_tle.get("MEAN_MOTION", 0))
    target_alt = sma_to_altitude(mean_motion_to_sma(mm))
    target_raan = float(maneuvered_tle.get("RA_OF_ASC_NODE", 0))

    nearby = []
    for tle in all_tles:
        tid = int(tle.get("NORAD_CAT_ID", 0))
        if tid == norad_id or tid <= 0:
            continue

        t_mm = float(tle.get("MEAN_MOTION", 0))
        t_alt = sma_to_altitude(mean_motion_to_sma(t_mm))
        t_raan = float(tle.get("RA_OF_ASC_NODE", 0))

        alt_diff = abs(t_alt - target_alt)
        raan_diff = abs(t_raan - target_raan)
        raan_diff = min(raan_diff, 360.0 - raan_diff)

        if alt_diff < alt_band_km and raan_diff < raan_band_deg:
            nearby.append(tle)

    return nearby


def propagate_counterfactual(
    pre_maneuver_tle: dict,
    nearby_tles: list[dict],
    hours_forward: float = 24.0,
    step_minutes: float = 10.0,
) -> dict:
    """Simulate "what if no maneuver?" using SGP4 propagation.

    Propagates the pre-maneuver TLE (before orbit change) forward and
    checks for close approaches with nearby satellites.

    Args:
        pre_maneuver_tle: Yesterday's TLE for the maneuvered satellite.
        nearby_tles: Current TLEs for nearby satellites.
        hours_forward: How far to propagate (hours).
        step_minutes: Time step for propagation (minutes).

    Returns:
        Dict with: min_distance_km, time_of_closest_approach,
                   would_have_collided, closest_norad_id, n_neighbors_checked.
    """
    if not SGP4_AVAILABLE:
        return {
            "min_distance_km": None,
            "would_have_collided": False,
            "error": "sgp4 not installed",
        }

    try:
        target_sat = celestrak_json_to_satrec(pre_maneuver_tle)
    except (ValueError, Exception) as e:
        return {
            "min_distance_km": None,
            "would_have_collided": False,
            "error": f"target TLE parse failed: {e}",
        }

    # Use current time as propagation start
    now = datetime.now(timezone.utc)
    start_jd = _datetime_to_jd(now)

    # Propagate maneuvered satellite (pre-maneuver orbit)
    target_positions = _propagate_positions(target_sat, start_jd, hours_forward, step_minutes)
    if len(target_positions) == 0:
        return {
            "min_distance_km": None,
            "would_have_collided": False,
            "error": "target propagation failed",
        }

    global_min_dist = float("inf")
    closest_norad = 0
    closest_time_offset_min = 0.0
    n_checked = 0

    for neighbor_tle in nearby_tles:
        try:
            neighbor_sat = celestrak_json_to_satrec(neighbor_tle)
        except (ValueError, Exception):
            continue

        neighbor_positions = _propagate_positions(neighbor_sat, start_jd, hours_forward, step_minutes)
        if len(neighbor_positions) == 0:
            continue

        n_checked += 1

        # Compute distances at each timestep (use min of overlapping steps)
        n_common = min(len(target_positions), len(neighbor_positions))
        diffs = target_positions[:n_common] - neighbor_positions[:n_common]
        distances = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < global_min_dist:
            global_min_dist = min_dist
            closest_norad = int(neighbor_tle.get("NORAD_CAT_ID", 0))
            closest_time_offset_min = min_idx * step_minutes

    if global_min_dist == float("inf"):
        return {
            "min_distance_km": None,
            "would_have_collided": False,
            "n_neighbors_checked": n_checked,
            "error": "no valid neighbors propagated",
        }

    tca_dt = now + timedelta(minutes=closest_time_offset_min)

    return {
        "min_distance_km": round(global_min_dist, 3),
        "time_of_closest_approach": tca_dt.isoformat(),
        "would_have_collided": global_min_dist < COLLISION_THRESHOLD_KM,
        "closest_norad_id": closest_norad,
        "n_neighbors_checked": n_checked,
    }
