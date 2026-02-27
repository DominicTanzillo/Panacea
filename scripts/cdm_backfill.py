"""Bulk CDM backfill from Space-Track.

Standalone script for manual local execution. Run up to 3x/day (8-hour
interval enforced). Each run fetches new CDMs via delta download (CDM_ID
bookmark) or does a full lookback on first run / bookmark reset.

Space-Track's cdm_public class has a rolling ~30-day window with ~3,200
CDMs. After the first full pull, subsequent runs pick up only new CDMs.

Usage:
    python scripts/cdm_backfill.py                     # Delta fetch (default)
    python scripts/cdm_backfill.py --lookback-days 30  # First run: 30-day pull
    python scripts/cdm_backfill.py --reset-bookmark    # Re-fetch from scratch
    python scripts/cdm_backfill.py --force              # Bypass rate limit
    python scripts/cdm_backfill.py --stats              # Just show store stats

Requires SPACETRACK_USER and SPACETRACK_PASS environment variables.
Register at https://www.space-track.org/auth/createAccount (free for .edu).
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.spacetrack_crossref import (
    fetch_and_store_cdms,
    _load_local_cdms,
    _DEFAULT_CDM_STORE,
    _FETCH_LOG,
    _can_fetch,
)


def print_store_stats(store_path: Path = None):
    """Print summary statistics of the local CDM store."""
    path = store_path or _DEFAULT_CDM_STORE
    cdms = _load_local_cdms(path)

    if not cdms:
        print("\n  CDM store is empty.")
        return

    print(f"\n{'='*60}")
    print(f"  CDM Store Statistics")
    print(f"{'='*60}")
    print(f"  Total CDMs: {len(cdms)}")

    # Date range
    tcas = [c.get("tca", "") for c in cdms if c.get("tca")]
    if tcas:
        print(f"  TCA range: {min(tcas)[:10]} to {max(tcas)[:10]}")

    # Pc distribution
    pcs = [c["pc"] for c in cdms if c.get("pc", 0) > 0]
    if pcs:
        print(f"\n  Pc distribution:")
        print(f"    >= 1e-4 (NASA CARA):   {sum(1 for p in pcs if p >= 1e-4):>5}")
        print(f"    >= 1e-5 (Kelvins):     {sum(1 for p in pcs if p >= 1e-5):>5}")
        print(f"    >= 1e-6 (Starlink):    {sum(1 for p in pcs if p >= 1e-6):>5}")
        print(f"    >= 1e-7 (screening):   {sum(1 for p in pcs if p >= 1e-7):>5}")
        print(f"    Max Pc: {max(pcs):.6e}")
        print(f"    Min Pc: {min(pcs):.6e}")

    # Miss distance
    dists = [c["miss_distance_km"] for c in cdms if c.get("miss_distance_km") is not None]
    if dists:
        print(f"\n  Miss distance (km):")
        print(f"    Min: {min(dists):.1f} km")
        print(f"    Median: {sorted(dists)[len(dists)//2]:.1f} km")
        print(f"    Max: {max(dists):.1f} km")
        print(f"    < 100 km: {sum(1 for d in dists if d < 100)}")
        print(f"    < 50 km:  {sum(1 for d in dists if d < 50)}")
        print(f"    < 25 km:  {sum(1 for d in dists if d < 25)}")
        print(f"    < 10 km:  {sum(1 for d in dists if d < 10)}")

    # Object types
    type_counter = Counter()
    for c in cdms:
        t1 = c.get("sat1_type", "UNKNOWN")
        t2 = c.get("sat2_type", "UNKNOWN")
        type_counter[f"{t1} vs {t2}"] += 1
    print(f"\n  Object type pairs (top 10):")
    for pair_type, count in type_counter.most_common(10):
        print(f"    {pair_type:30s} {count:>5}")

    # Emergency reportable
    emergency = sum(1 for c in cdms if c.get("emergency_reportable") == "Y")
    print(f"\n  Emergency reportable: {emergency}/{len(cdms)}")

    # Most common satellites
    sat_counter = Counter()
    for c in cdms:
        s1 = c.get("sat1_name", "?")
        s2 = c.get("sat2_name", "?")
        sat_counter[s1] += 1
        sat_counter[s2] += 1
    print(f"\n  Most common satellites (top 15):")
    for name, count in sat_counter.most_common(15):
        print(f"    {name:30s} {count:>5} CDMs")

    # Top 10 highest Pc events
    by_pc = sorted(cdms, key=lambda c: c.get("pc", 0), reverse=True)
    print(f"\n  Top 10 highest Pc events:")
    print(f"    {'Pc':>12s}  {'Miss(km)':>10s}  {'TCA':>12s}  {'Sat1':>20s}  {'Sat2':>20s}  {'Emrg':>4s}")
    for c in by_pc[:10]:
        print(f"    {c['pc']:>12.6e}  {c['miss_distance_km']:>10.1f}  "
              f"{c.get('tca', '?')[:10]:>12s}  {c.get('sat1_name', '?'):>20s}  "
              f"{c.get('sat2_name', '?'):>20s}  {c.get('emergency_reportable', ''):>4s}")

    # Fetch log
    if _FETCH_LOG.exists():
        try:
            with open(_FETCH_LOG) as f:
                log = json.load(f)
            print(f"\n  Last fetch: {log.get('last_cdm_fetch', '?')}")
            print(f"  Fetch count: {log.get('fetch_count', '?')}")
            print(f"  CDM_ID bookmark: {log.get('last_cdm_id', 'none')}")
        except Exception:
            pass

    can = _can_fetch()
    print(f"\n  Rate limit status: {'READY to fetch' if can else 'RATE-LIMITED (wait 8h)'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk CDM backfill from Space-Track",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lookback-days", type=int, default=30,
        help="Days to look back on initial fetch (default: 30)",
    )
    parser.add_argument(
        "--reset-bookmark", action="store_true",
        help="Ignore CDM_ID bookmark and fetch by date range",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass 8-hour rate limit (use carefully!)",
    )
    parser.add_argument(
        "--min-pc", type=float, default=1e-7,
        help="Minimum Pc to store (default: 1e-7)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Just print store statistics, don't fetch",
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Panacea CDM Backfill")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    if args.stats:
        print_store_stats()
        return

    # Check rate limit
    if not args.force and not _can_fetch():
        print("\n  Rate-limited (last fetch < 8 hours ago).")
        print("  Use --force to override, or wait for the 8-hour window.")
        print_store_stats()
        return

    # Fetch
    print(f"\n  Fetching CDMs (lookback={args.lookback_days}d, "
          f"min_pc={args.min_pc:.0e}, "
          f"reset_bookmark={args.reset_bookmark}, "
          f"force={args.force}) ...")

    result = fetch_and_store_cdms(
        lookback_days=args.lookback_days,
        min_pc=args.min_pc,
        force=args.force,
        reset_bookmark=args.reset_bookmark,
    )

    print(f"\n  Result: {result['fetched']} fetched, {result['new']} new, "
          f"{result['total']} total in store")

    if result.get("error"):
        print(f"  Error: {result['error']}")
    if result.get("rate_limited"):
        print("  (Rate-limited — returned cached data)")

    # Show stats
    print_store_stats()


if __name__ == "__main__":
    main()
