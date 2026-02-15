"""Cross-reference detected maneuvers with Space-Track.org CDM data.

Queries the CDM_PUBLIC class for recent conjunction data messages
involving maneuvered satellites. CDM confirmation is the strongest
signal that a maneuver was collision-avoidance.

Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.
Fails silently if credentials are not set (purely enrichment).
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Rate limiting: max 30 requests/min to Space-Track
MAX_REQUESTS_PER_MIN = 30
BATCH_SIZE = 100  # Max NORAD IDs per query
CACHE_EXPIRY_DAYS = 7

SPACETRACK_BASE = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE}/ajaxauth/login"
CDM_QUERY_URL = f"{SPACETRACK_BASE}/basicspacedata/query/class/cdm_public"


def _get_credentials() -> tuple[str, str]:
    """Get Space-Track credentials from environment."""
    user = os.environ.get("SPACETRACK_USER", "")
    passwd = os.environ.get("SPACETRACK_PASS", "")
    return user, passwd


def _load_cache(cache_path: Path) -> dict:
    """Load CDM cache, filtering expired entries."""
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    # Filter expired entries
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_EXPIRY_DAYS)).isoformat()
    return {
        k: v for k, v in cache.items()
        if v.get("cached_at", "") > cutoff
    }


def _save_cache(cache: dict, cache_path: Path):
    """Save CDM cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def check_cdm_for_norad_ids(
    norad_ids: list[int],
    lookback_days: int = 7,
    min_pc: float = 1e-7,
    cache_dir: Path = None,
) -> dict[int, list[dict]]:
    """Query Space-Track CDM_PUBLIC for recent CDMs involving given satellites.

    Args:
        norad_ids: NORAD catalog IDs to check.
        lookback_days: How far back to search for CDMs.
        min_pc: Minimum probability of collision to include.
        cache_dir: Directory for CDM cache file. Defaults to data/prediction_logs/.

    Returns:
        Map of norad_id -> list of CDM records with PC, TCA, MISS_DISTANCE.
        Empty dict if credentials not set or query fails.
    """
    user, passwd = _get_credentials()
    if not user or not passwd:
        return {}

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "prediction_logs"

    cache_path = cache_dir / "cdm_cache.json"
    cache = _load_cache(cache_path)

    # Check which IDs need fresh queries
    results = {}
    uncached_ids = []

    for nid in norad_ids:
        key = str(nid)
        if key in cache:
            results[nid] = cache[key].get("cdms", [])
        else:
            uncached_ids.append(nid)

    if not uncached_ids:
        return results

    # Authenticate with Space-Track
    try:
        session = requests.Session()
        resp = session.post(LOGIN_URL, data={
            "identity": user,
            "password": passwd,
        }, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Space-Track login failed: {e}")
        return results

    # Query in batches
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lookback_str = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    for batch_start in range(0, len(uncached_ids), BATCH_SIZE):
        batch = uncached_ids[batch_start:batch_start + BATCH_SIZE]
        ids_str = ",".join(str(nid) for nid in batch)

        query_url = (
            f"{CDM_QUERY_URL}"
            f"/SAT1_NORAD_CAT_ID/{ids_str}"
            f"/TCA/>{lookback_str}"
            f"/orderby/TCA desc"
            f"/format/json"
        )

        try:
            resp = session.get(query_url, timeout=60)
            resp.raise_for_status()
            cdm_records = resp.json()
        except Exception as e:
            print(f"  Space-Track CDM query failed: {e}")
            # Cache empty results for failed IDs to avoid re-querying
            for nid in batch:
                cache[str(nid)] = {
                    "cdms": [],
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }
            continue

        # Process CDM records
        batch_results: dict[int, list[dict]] = {nid: [] for nid in batch}

        for cdm in cdm_records:
            try:
                pc = float(cdm.get("PC", 0) or 0)
                if pc < min_pc:
                    continue

                sat1_id = int(cdm.get("SAT1_NORAD_CAT_ID", 0))
                record = {
                    "tca": cdm.get("TCA", ""),
                    "pc": pc,
                    "miss_distance_km": float(cdm.get("MISS_DISTANCE", 0) or 0) / 1000.0,
                    "sat1_name": cdm.get("SAT1_NAME", ""),
                    "sat2_name": cdm.get("SAT2_NAME", ""),
                    "sat2_norad": int(cdm.get("SAT2_NORAD_CAT_ID", 0) or 0),
                }

                if sat1_id in batch_results:
                    batch_results[sat1_id].append(record)
            except (ValueError, TypeError):
                continue

        # Update cache and results
        for nid in batch:
            cdms = batch_results.get(nid, [])
            results[nid] = cdms
            cache[str(nid)] = {
                "cdms": cdms,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

        # Rate limiting between batches
        if batch_start + BATCH_SIZE < len(uncached_ids):
            time.sleep(60.0 / MAX_REQUESTS_PER_MIN)

    # Save updated cache
    _save_cache(cache, cache_path)

    return results
