"""Space weather indices for CDM prediction feature augmentation.

Fetches F10.7 (solar flux), Kp (geomagnetic activity), and Ap (geomagnetic
amplitude) from NOAA SWPC JSON endpoints. These indices may correlate with
atmospheric drag variations that affect conjunction geometry.

Note: LeoLabs' 2023 RFI response (Section 14) states operational experience
shows space weather sensitivity "has not improved the operator's situational
awareness during an event." We include these features as a controlled
experiment and report the result honestly -- a null result is still valuable.

No authentication or rate limits required for NOAA SWPC public data.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "space_weather"
CACHE_FILE = CACHE_DIR / "indices_cache.json"

# NOAA SWPC public JSON endpoints
F107_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

# 30-day climatological fallbacks (solar cycle 25 average conditions, 2024-2026)
CLIMATOLOGY = {"f107": 155.0, "kp": 2.3, "ap": 10.0}


class SpaceWeatherCache:
    """Fetch and cache space weather indices keyed by date.

    Usage:
        cache = SpaceWeatherCache()
        indices = cache.get_indices("2026-03-19")
        # {"f107": 162.3, "kp": 3.0, "ap": 15.0, "source": "swpc"}
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_file = self._cache_dir / "indices_cache.json"
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached indices from disk."""
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self):
        """Persist cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(json.dumps(self._cache, indent=2))

    def get_indices(self, date_str: str) -> dict:
        """Get space weather indices for a given date (YYYY-MM-DD or ISO).

        Returns dict with f107, kp, ap, and source ("swpc", "cache", or "climatology").
        Falls back gracefully: cache -> SWPC fetch -> 30-day average -> climatology.
        """
        date_key = date_str[:10]  # normalize to YYYY-MM-DD

        # Check cache first
        if date_key in self._cache:
            return {**self._cache[date_key], "source": "cache"}

        # Try fetching from NOAA SWPC
        indices = self._fetch_from_swpc(date_key)
        if indices:
            self._cache[date_key] = indices
            self._save_cache()
            return {**indices, "source": "swpc"}

        # Try computing 30-day rolling average from cached data
        avg = self._rolling_average(date_key, window=30)
        if avg:
            self._cache[date_key] = avg
            self._save_cache()
            return {**avg, "source": "rolling_avg"}

        # Fall back to climatology
        return {**CLIMATOLOGY, "source": "climatology"}

    def _fetch_from_swpc(self, date_key: str) -> Optional[dict]:
        """Fetch F10.7 and Kp/Ap from NOAA SWPC JSON endpoints."""
        try:
            import requests
        except ImportError:
            return None

        f107 = self._fetch_f107(date_key)
        kp_ap = self._fetch_kp_ap(date_key)

        if f107 is None and kp_ap is None:
            return None

        return {
            "f107": f107 if f107 is not None else CLIMATOLOGY["f107"],
            "kp": kp_ap["kp"] if kp_ap else CLIMATOLOGY["kp"],
            "ap": kp_ap["ap"] if kp_ap else CLIMATOLOGY["ap"],
        }

    def _fetch_f107(self, date_key: str) -> Optional[float]:
        """Fetch F10.7 solar flux index from SWPC observed solar cycle indices."""
        try:
            import requests
            resp = requests.get(F107_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Data is array of {"time-tag": "YYYY-MM", "f10.7": value, ...}
            # Match by year-month since this is monthly data
            target_ym = date_key[:7]  # YYYY-MM
            for entry in reversed(data):  # most recent first
                if entry.get("time-tag", "")[:7] == target_ym:
                    val = entry.get("f10.7")
                    if val is not None:
                        return float(val)

            # If exact month not found, use most recent available
            if data:
                for entry in reversed(data):
                    val = entry.get("f10.7")
                    if val is not None:
                        return float(val)
        except Exception:
            pass
        return None

    def _fetch_kp_ap(self, date_key: str) -> Optional[dict]:
        """Fetch Kp and Ap from SWPC planetary K-index product."""
        try:
            import requests
            resp = requests.get(KP_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Data is array: header row + data rows
            # Format: ["time_tag", "Kp", "Kp_fraction", "a_running", "station_count"]
            if len(data) < 2:
                return None

            # Find entries matching our date
            kp_values = []
            ap_values = []
            for row in data[1:]:  # skip header
                if len(row) < 4:
                    continue
                time_tag = str(row[0])
                if time_tag[:10] == date_key:
                    try:
                        kp_values.append(float(row[1]))
                    except (ValueError, TypeError):
                        pass
                    try:
                        ap_values.append(float(row[3]))
                    except (ValueError, TypeError):
                        pass

            if kp_values:
                return {
                    "kp": round(sum(kp_values) / len(kp_values), 1),
                    "ap": round(sum(ap_values) / len(ap_values), 0) if ap_values else CLIMATOLOGY["ap"],
                }

            # If date not found, use most recent available
            recent_kp = []
            recent_ap = []
            for row in data[-8:]:  # last 8 entries (1 day of 3-hour Kp)
                if len(row) < 4:
                    continue
                try:
                    recent_kp.append(float(row[1]))
                except (ValueError, TypeError):
                    pass
                try:
                    recent_ap.append(float(row[3]))
                except (ValueError, TypeError):
                    pass

            if recent_kp:
                return {
                    "kp": round(sum(recent_kp) / len(recent_kp), 1),
                    "ap": round(sum(recent_ap) / len(recent_ap), 0) if recent_ap else CLIMATOLOGY["ap"],
                }
        except Exception:
            pass
        return None

    def _rolling_average(self, date_key: str, window: int = 30) -> Optional[dict]:
        """Compute rolling average from cached data within the last N days."""
        try:
            target = datetime.strptime(date_key, "%Y-%m-%d")
        except ValueError:
            return None

        f107s, kps, aps = [], [], []
        for d in range(1, window + 1):
            dk = (target - timedelta(days=d)).strftime("%Y-%m-%d")
            if dk in self._cache:
                entry = self._cache[dk]
                if "f107" in entry:
                    f107s.append(entry["f107"])
                if "kp" in entry:
                    kps.append(entry["kp"])
                if "ap" in entry:
                    aps.append(entry["ap"])

        if not f107s and not kps:
            return None

        return {
            "f107": round(sum(f107s) / len(f107s), 1) if f107s else CLIMATOLOGY["f107"],
            "kp": round(sum(kps) / len(kps), 1) if kps else CLIMATOLOGY["kp"],
            "ap": round(sum(aps) / len(aps), 0) if aps else CLIMATOLOGY["ap"],
        }

    def bulk_lookup(self, dates: list[str]) -> dict[str, dict]:
        """Look up indices for multiple dates, fetching as needed."""
        unique_dates = sorted(set(d[:10] for d in dates))
        if unique_dates:
            # Fetch most recent date first (populates SWPC cache broadly),
            # then fetch any remaining uncached dates individually.
            self.get_indices(unique_dates[-1])
            for d in unique_dates:
                if d not in self._cache:
                    self.get_indices(d)

        results = {d[:10]: self.get_indices(d) for d in unique_dates}

        # Log source distribution for diagnostics
        sources: dict[str, int] = {}
        for v in results.values():
            src = v.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        n_climatology = sources.get("climatology", 0)
        n_total = len(results)
        if n_total > 0 and n_climatology > 0:
            pct = n_climatology / n_total * 100
            print(f"  Space weather: {n_total} dates, sources={sources}")
            if pct > 50:
                print(f"  WARNING: {pct:.0f}% of space weather lookups fell back to climatology defaults")
        return results


def normalize_space_weather(f107: float, kp: float, ap: float) -> tuple[float, float, float]:
    """Normalize space weather indices to roughly [0, 1] range for ML features.

    Ranges (Solar Cycle 25):
      F10.7: ~65 (quiet) to ~300 (active) -> log-scaled
      Kp: 0 (quiet) to 9 (extreme storm) -> linear /9
      Ap: 0 (quiet) to ~400 (extreme) -> log-scaled
    """
    f107_norm = (math.log10(max(f107, 60)) - math.log10(60)) / (math.log10(300) - math.log10(60))
    kp_norm = min(kp, 9.0) / 9.0
    ap_norm = math.log1p(ap) / math.log1p(400)
    return (
        round(min(max(f107_norm, 0), 1), 4),
        round(min(max(kp_norm, 0), 1), 4),
        round(min(max(ap_norm, 0), 1), 4),
    )
