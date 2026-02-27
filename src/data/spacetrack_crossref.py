"""Space-Track.org CDM client for conjunction data with real Pc values.

This module is the PRIMARY label source for the daily pipeline, replacing
TLE-based counterfactual propagation which is limited by TLE position
accuracy (~3.3 km combined 1-sigma, see writeup/paper.md Section 6.4).

Space-Track's CDM_PUBLIC class provides:
  - Probability of Collision (Pc) computed with proper covariance
  - Miss distance in meters (not km-scale TLE approximations)
  - Full 6x6 RTN covariance matrices for both objects
  - Object identification (NORAD IDs, names, types)
  - TCA (Time of Closest Approach)

RATE LIMITS (government-mandated, account suspension if violated):
  - CDM class: MAX 3 queries per day (once every 8 hours)
  - General: <30 requests/min, <300 requests/hour
  - Must store downloaded data locally, not re-download

Architecture:
  - ONE bulk CDM query per daily pipeline run (uses 1 of 3 daily quota)
  - Results cached to local JSONL (persistent, not re-downloaded)
  - Cross-reference against maneuvered satellites happens locally
  - check_cdm_for_norad_ids() reads from local cache, never queries API

Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.
Register at https://www.space-track.org/auth/createAccount (free for .edu).
Fails silently if credentials are not set.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone

SPACETRACK_BASE = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE}/ajaxauth/login"
CDM_QUERY_URL = f"{SPACETRACK_BASE}/basicspacedata/query/class/cdm_public"

# Local persistent store — download once, never re-download
_DEFAULT_CDM_STORE = Path(__file__).parent.parent.parent / "data" / "prediction_logs" / "cdm_store.jsonl"
_FETCH_LOG = Path(__file__).parent.parent.parent / "data" / "prediction_logs" / "cdm_fetch_log.json"

# Rate limit: max 3 CDM queries per day (8-hour minimum interval)
_MIN_FETCH_INTERVAL_HOURS = 8


def _get_session() -> tuple[requests.Session, bool]:
    """Authenticate with Space-Track. Returns (session, success)."""
    user = os.environ.get("SPACETRACK_USER", "")
    passwd = os.environ.get("SPACETRACK_PASS", "")
    if not user or not passwd:
        return requests.Session(), False

    session = requests.Session()
    try:
        resp = session.post(LOGIN_URL, data={
            "identity": user,
            "password": passwd,
        }, timeout=30)
        resp.raise_for_status()
        if resp.text and "Login Failed" in resp.text:
            print(f"  Space-Track login failed: {resp.text[:200]}")
            return session, False
        return session, True
    except Exception as e:
        print(f"  Space-Track login failed: {e}")
        return session, False


def _can_fetch(fetch_log_path: Path = None) -> bool:
    """Check if we're allowed to fetch (respecting 8-hour interval)."""
    log_path = fetch_log_path or _FETCH_LOG
    if not log_path.exists():
        return True
    try:
        with open(log_path) as f:
            log = json.load(f)
        last_fetch = log.get("last_cdm_fetch", "")
        if not last_fetch:
            return True
        last_dt = datetime.fromisoformat(last_fetch)
        elapsed = datetime.now(timezone.utc) - last_dt
        return elapsed >= timedelta(hours=_MIN_FETCH_INTERVAL_HOURS)
    except Exception:
        return True


def _record_fetch(fetch_log_path: Path = None, max_cdm_id: str = ""):
    """Record that we just did a CDM fetch, with bookmark for delta downloads."""
    log_path = fetch_log_path or _FETCH_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {}
    if log_path.exists():
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            pass
    log["last_cdm_fetch"] = datetime.now(timezone.utc).isoformat()
    log["fetch_count"] = log.get("fetch_count", 0) + 1
    if max_cdm_id:
        log["last_cdm_id"] = max_cdm_id
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def _logout(session: requests.Session):
    """Logout from Space-Track to close the session (per API docs)."""
    try:
        session.get(f"{SPACETRACK_BASE}/ajaxauth/logout", timeout=10)
    except Exception:
        pass


def _parse_cdm(cdm: dict) -> dict:
    """Extract key fields from a raw CDM_PUBLIC record."""
    def _float(val, default=0.0):
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    def _int(val, default=0):
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default

    pc = _float(cdm.get("COLLISION_PROBABILITY"))
    miss_m = _float(cdm.get("MISS_DISTANCE"))

    return {
        "cdm_id": cdm.get("CDM_ID", ""),
        "tca": cdm.get("TCA", ""),
        "creation_date": cdm.get("CREATION_DATE", ""),
        "pc": pc,
        "miss_distance_m": miss_m,
        "miss_distance_km": miss_m / 1000.0,
        "relative_speed_m_s": _float(cdm.get("RELATIVE_SPEED")),
        "miss_r_m": _float(cdm.get("RELATIVE_POSITION_R")),
        "miss_t_m": _float(cdm.get("RELATIVE_POSITION_T")),
        "miss_n_m": _float(cdm.get("RELATIVE_POSITION_N")),
        "sat1_norad": _int(cdm.get("SAT1_OBJECT_DESIGNATOR")),
        "sat1_name": cdm.get("SAT1_OBJECT_NAME", ""),
        "sat1_type": cdm.get("SAT1_OBJECT_TYPE", ""),
        "sat2_norad": _int(cdm.get("SAT2_OBJECT_DESIGNATOR")),
        "sat2_name": cdm.get("SAT2_OBJECT_NAME", ""),
        "sat2_type": cdm.get("SAT2_OBJECT_TYPE", ""),
        "collision_probability_method": cdm.get("COLLISION_PROBABILITY_METHOD", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_local_cdms(store_path: Path = None) -> list[dict]:
    """Load all CDMs from local persistent store."""
    path = store_path or _DEFAULT_CDM_STORE
    if not path.exists():
        return []
    cdms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cdms.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cdms


def _append_cdms_to_store(cdms: list[dict], store_path: Path = None):
    """Append new CDMs to the local persistent store (deduplicating)."""
    path = store_path or _DEFAULT_CDM_STORE
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CDM IDs for dedup
    existing_ids = set()
    for existing in _load_local_cdms(path):
        cid = existing.get("cdm_id", "")
        if cid:
            existing_ids.add(cid)

    n_new = 0
    with open(path, "a") as f:
        for cdm in cdms:
            cid = cdm.get("cdm_id", "")
            if cid and cid not in existing_ids:
                f.write(json.dumps(cdm) + "\n")
                existing_ids.add(cid)
                n_new += 1

    return n_new


def _get_last_cdm_id(fetch_log_path: Path = None) -> str:
    """Get the highest CDM_ID from last fetch (bookmark for delta downloads)."""
    log_path = fetch_log_path or _FETCH_LOG
    if not log_path.exists():
        return ""
    try:
        with open(log_path) as f:
            log = json.load(f)
        return log.get("last_cdm_id", "")
    except Exception:
        return ""


def fetch_and_store_cdms(
    lookback_days: int = 3,
    min_pc: float = 1e-7,
    store_path: Path = None,
) -> list[dict]:
    """Fetch recent CDMs from Space-Track and store locally.

    This is the ONLY function that makes API calls. It:
      1. Checks the 8-hour rate limit
      2. Uses CDM_ID bookmark for delta downloads (only fetch new records)
      3. Falls back to date-range query on first run
      4. Appends new CDMs to local JSONL store (never re-downloads)
      5. Logs out session when done

    Uses 1 of 3 daily CDM query quota. Called once per daily pipeline run.
    """
    if not _can_fetch():
        print("  Space-Track CDM: rate-limited (last fetch < 8 hours ago)")
        return _load_local_cdms(store_path)

    session, authenticated = _get_session()
    if not authenticated:
        return _load_local_cdms(store_path)

    try:
        # Delta download: use CDM_ID bookmark if available
        last_cdm_id = _get_last_cdm_id()
        if last_cdm_id:
            # Only fetch CDMs with ID higher than our bookmark
            query_url = (
                f"{CDM_QUERY_URL}"
                f"/CDM_ID/>{last_cdm_id}"
                f"/COLLISION_PROBABILITY/>{min_pc}"
                f"/orderby/CDM_ID asc"
                f"/format/json"
            )
            print(f"  Space-Track CDM: delta fetch (CDM_ID > {last_cdm_id}) ...")
        else:
            # First run: fetch by date range
            lookback_str = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            query_url = (
                f"{CDM_QUERY_URL}"
                f"/TCA/>{lookback_str}"
                f"/COLLISION_PROBABILITY/>{min_pc}"
                f"/orderby/CDM_ID asc"
                f"/format/json"
            )
            print(f"  Space-Track CDM: initial fetch (TCA > {lookback_str}, Pc > {min_pc:.0e}) ...")

        resp = session.get(query_url, timeout=120)
        resp.raise_for_status()
        raw_cdms = resp.json() if resp.text else []
    except Exception as e:
        print(f"  Space-Track CDM query failed: {e}")
        _logout(session)
        return _load_local_cdms(store_path)

    # Logout to close session (per Space-Track docs)
    _logout(session)

    # Parse and store
    parsed = [_parse_cdm(c) for c in raw_cdms]
    parsed = [c for c in parsed if c["pc"] >= min_pc]
    n_new = _append_cdms_to_store(parsed, store_path)

    # Update bookmark: store highest CDM_ID for next delta download
    max_cdm_id = ""
    if parsed:
        max_cdm_id = max(p.get("cdm_id", "") for p in parsed)

    # Record the fetch with bookmark
    _record_fetch(max_cdm_id=max_cdm_id)

    print(f"  Space-Track CDM: {len(parsed)} CDMs fetched, {n_new} new (stored locally)")
    return parsed


def fetch_recent_cdms(
    lookback_days: int = 3,
    min_pc: float = 1e-5,
    cache_dir: Path = None,
) -> list[dict]:
    """Get recent CDMs, fetching from API if allowed, else from local store.

    Wrapper for backward compatibility. Prefers local store to avoid
    unnecessary API calls.
    """
    store_path = None
    if cache_dir:
        store_path = cache_dir / "cdm_store.jsonl"

    # Try API fetch (respects rate limit internally)
    fetched = fetch_and_store_cdms(
        lookback_days=lookback_days,
        min_pc=min_pc,
        store_path=store_path,
    )

    if fetched:
        return [c for c in fetched if c["pc"] >= min_pc]

    # Fall back to local store
    local = _load_local_cdms(store_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    return [c for c in local if c["pc"] >= min_pc and c.get("tca", "") >= cutoff]


def check_cdm_for_norad_ids(
    norad_ids: list[int],
    lookback_days: int = 7,
    min_pc: float = 1e-7,
    cache_dir: Path = None,
) -> dict[int, list[dict]]:
    """Cross-reference NORAD IDs against LOCAL CDM store.

    This function NEVER makes API calls. It reads from the local store
    populated by fetch_and_store_cdms(). This is safe to call for
    thousands of NORAD IDs with zero API quota cost.

    Args:
        norad_ids: NORAD catalog IDs to check.
        lookback_days: How far back to search in local store.
        min_pc: Minimum Pc to include.
        cache_dir: Directory containing cdm_store.jsonl.

    Returns:
        Map of norad_id -> list of CDM records.
    """
    store_path = None
    if cache_dir:
        store_path = cache_dir / "cdm_store.jsonl"

    all_cdms = _load_local_cdms(store_path)

    # Filter by date
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    # Build lookup
    id_set = set(norad_ids)
    results: dict[int, list[dict]] = {nid: [] for nid in norad_ids}

    for cdm in all_cdms:
        if cdm["pc"] < min_pc:
            continue
        if cdm.get("tca", "") < cutoff:
            continue

        sat1 = cdm.get("sat1_norad", 0)
        sat2 = cdm.get("sat2_norad", 0)

        if sat1 in id_set:
            results[sat1].append(cdm)
        if sat2 in id_set:
            results[sat2].append(cdm)

    n_with_cdms = sum(1 for v in results.values() if v)
    if n_with_cdms:
        total_cdms = sum(len(v) for v in results.values())
        print(f"  CDM local lookup: {n_with_cdms}/{len(norad_ids)} IDs have CDMs ({total_cdms} total)")

    return results


def cdm_pc_to_label(pc: float) -> tuple[float, float]:
    """Convert CDM Pc value to (risk_label, sample_weight) for training.

    Based on operational thresholds:
      - NASA CARA maneuver threshold: Pc > 1e-4
      - Starlink autonomous threshold: Pc > 1e-6
      - 18 SDS screening threshold: Pc > ~1e-7

    Unlike TLE-derived labels, CDM Pc values are computed with SP
    ephemerides and full covariance -- they are ground truth.
    """
    if pc >= 1e-4:
        return 1.0, 1.0     # NASA CARA maneuver threshold
    elif pc >= 1e-5:
        return 0.9, 0.95    # Kelvins risk threshold
    elif pc >= 1e-6:
        return 0.7, 0.8     # Starlink autonomous threshold
    elif pc >= 1e-7:
        return 0.3, 0.5     # Screening threshold
    else:
        return 0.0, 1.0
