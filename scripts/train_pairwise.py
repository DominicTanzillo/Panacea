"""Train the CDM-Supervised Pairwise Risk Model (CSPR).

Builds training data by joining CDM records with TLE snapshots, computes
pairwise features, generates negative samples, and trains XGBoost.

Data flow:
  1. Load CDM store (data/prediction_logs/cdm_store.jsonl)
  2. For each CDM pair, find closest TLE snapshot by date
  3. Look up both satellites' TLEs in that snapshot
  4. Compute pairwise features via compute_pairwise_features()
  5. Generate negative samples: random TLE pairs NOT in any CDM
  6. Train CDMSupervisedRiskModel with cdm_pc_to_label() weights
  7. Evaluate with held-out split, generate learning curve

Usage:
    python scripts/train_pairwise.py                       # Full run
    python scripts/train_pairwise.py --min-cdms 50         # Lower CDM threshold
    python scripts/train_pairwise.py --neg-ratio 3         # 3 negatives per positive
    python scripts/train_pairwise.py --dry-run             # Check data only
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pairwise_features import (
    compute_pairwise_features,
    compute_batch_features,
    PAIRWISE_FEATURE_COLS,
)
from src.data.maneuver_detector import (
    load_tle_snapshot,
    mean_motion_to_sma,
    EARTH_RADIUS_KM,
)
from src.data.spacetrack_crossref import _load_local_cdms, cdm_pc_to_label
from src.model.pairwise import CDMSupervisedRiskModel

CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
SNAPSHOT_DIR = ROOT / "data" / "tle_snapshots"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def load_tle_snapshots() -> dict[str, list[dict]]:
    """Load all available TLE snapshots, keyed by date string."""
    snapshots = {}
    for f in sorted(SNAPSHOT_DIR.glob("*.json")):
        date_str = f.stem  # e.g., "2026-02-27"
        try:
            tles = load_tle_snapshot(f)
            snapshots[date_str] = tles
            print(f"  Loaded TLE snapshot {date_str}: {len(tles)} satellites")
        except Exception as e:
            print(f"  Failed to load {f}: {e}")
    return snapshots


def find_closest_snapshot(cdm_date: str, snapshot_dates: list[str]) -> str:
    """Find the TLE snapshot date closest to a CDM date."""
    if not snapshot_dates:
        return ""
    # Parse CDM date (could be "2026-02-24T03:38:37" or "2026-02-24")
    cdm_day = cdm_date[:10]
    # Find closest by string comparison (dates are ISO format)
    best = min(snapshot_dates, key=lambda d: abs(
        (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(cdm_day, "%Y-%m-%d")).days
    ))
    return best


def build_tle_lookup(tles: list[dict]) -> dict[int, dict]:
    """Build NORAD ID -> TLE lookup from a snapshot."""
    lookup = {}
    for tle in tles:
        nid = int(tle.get("NORAD_CAT_ID", 0))
        if nid > 0:
            lookup[nid] = tle
    return lookup


def build_training_data(
    cdms: list[dict],
    snapshots: dict[str, list[dict]],
    neg_ratio: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build training dataset from CDMs + TLE snapshots.

    Returns:
        X: Feature DataFrame
        y_pc: Raw Pc values (positive from CDMs, synthetic 1e-20 for negatives)
        weights: Sample weights from cdm_pc_to_label()
    """
    snapshot_dates = sorted(snapshots.keys())
    if not snapshot_dates:
        raise ValueError("No TLE snapshots available")

    # Deduplicate CDMs by (sat1, sat2) pair — keep highest Pc
    pair_cdms: dict[tuple[int, int], dict] = {}
    for cdm in cdms:
        s1 = cdm.get("sat1_norad", 0)
        s2 = cdm.get("sat2_norad", 0)
        if s1 == 0 or s2 == 0:
            continue
        key = (min(s1, s2), max(s1, s2))
        if key not in pair_cdms or cdm["pc"] > pair_cdms[key]["pc"]:
            pair_cdms[key] = cdm

    print(f"  Unique CDM pairs: {len(pair_cdms)} (from {len(cdms)} raw CDMs)")

    # Build positive samples from CDM pairs
    # With supplemental TLEs fetched by daily_predictions.py, most CDM objects
    # (debris, old payloads, rocket bodies) should now have real TLE data in
    # the snapshots. Skip pairs where either satellite is still missing.
    positive_pairs = []
    positive_pcs = []
    positive_weights = []
    cdm_norad_pairs = set()  # for negative sampling exclusion
    n_skipped = 0

    # Build combined lookup from all snapshots
    all_tle_lookup: dict[int, dict] = {}
    for date_str in snapshot_dates:
        for tle in snapshots[date_str]:
            nid = int(tle.get("NORAD_CAT_ID", 0))
            if nid > 0:
                all_tle_lookup[nid] = tle

    for (s1, s2), cdm in pair_cdms.items():
        tle1 = all_tle_lookup.get(s1)
        tle2 = all_tle_lookup.get(s2)

        # Skip pairs where either satellite has no real TLE data
        if tle1 is None or tle2 is None:
            n_skipped += 1
            continue

        positive_pairs.append((tle1, tle2))
        positive_pcs.append(cdm["pc"])
        _, weight = cdm_pc_to_label(cdm["pc"])
        positive_weights.append(weight)
        cdm_norad_pairs.add((s1, s2))

    n_pos = len(positive_pairs)
    print(f"  Positive samples: {n_pos} ({n_skipped} pairs skipped — missing TLE data)")

    if n_pos == 0:
        raise ValueError("No CDM pairs could be matched or synthesized")

    # Generate negative samples from same altitude shells
    # Use the most recent snapshot for negative sampling
    latest_snap = snapshot_dates[-1]
    latest_tles = snapshots[latest_snap]
    latest_lookup = build_tle_lookup(latest_tles)

    # Precompute altitudes for sampling
    norad_ids = list(latest_lookup.keys())
    altitudes = {}
    for nid in norad_ids:
        tle = latest_lookup[nid]
        mm = float(tle.get("MEAN_MOTION", 0))
        if mm > 0:
            sma = mean_motion_to_sma(mm)
            altitudes[nid] = sma - EARTH_RADIUS_KM

    # Group by altitude bands (50 km) for shell-matched negative sampling
    alt_bands: dict[int, list[int]] = {}
    for nid, alt in altitudes.items():
        band = int(round(alt / 50.0) * 50)
        alt_bands.setdefault(band, []).append(nid)

    np.random.seed(42)
    n_neg_target = n_pos * neg_ratio
    negative_pairs = []
    negative_attempts = 0
    max_attempts = n_neg_target * 10

    while len(negative_pairs) < n_neg_target and negative_attempts < max_attempts:
        negative_attempts += 1
        # Pick a random altitude band (weighted by population)
        bands = list(alt_bands.keys())
        band_sizes = [len(alt_bands[b]) for b in bands]
        band_probs = np.array(band_sizes, dtype=float)
        band_probs /= band_probs.sum()
        band = np.random.choice(bands, p=band_probs)

        candidates = alt_bands[band]
        if len(candidates) < 2:
            continue

        idx = np.random.choice(len(candidates), size=2, replace=False)
        n1, n2 = candidates[idx[0]], candidates[idx[1]]
        pair_key = (min(n1, n2), max(n1, n2))

        # Exclude CDM pairs
        if pair_key in cdm_norad_pairs:
            continue

        tle1 = latest_lookup[n1]
        tle2 = latest_lookup[n2]
        negative_pairs.append((tle1, tle2))

    n_neg = len(negative_pairs)
    print(f"  Negative samples (shell-matched random pairs): {n_neg}")

    # Compute features for all pairs
    print("  Computing pairwise features ...")
    all_pairs = positive_pairs + negative_pairs
    all_pcs = np.array(positive_pcs + [1e-20] * n_neg)
    all_weights = np.array(positive_weights + [1.0] * n_neg)

    # Use latest snapshot for density features
    X = compute_batch_features(all_pairs, all_tles=latest_tles)
    print(f"  Feature matrix: {X.shape}")

    return X, all_pcs, all_weights


def train_and_evaluate(
    X: pd.DataFrame,
    y_pc: np.ndarray,
    weights: np.ndarray,
    test_fraction: float = 0.2,
) -> tuple[CDMSupervisedRiskModel, dict, dict]:
    """Train model with train/test split, return model and metrics."""
    from sklearn.model_selection import train_test_split

    # Split preserving positive/negative ratio
    y_binary = (y_pc > 1e-10).astype(int)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_pc, weights,
        test_size=test_fraction,
        random_state=42,
        stratify=y_binary,
    )

    print(f"\n  Train: {len(X_train)} ({(y_train > 1e-10).sum()} positive)")
    print(f"  Test:  {len(X_test)} ({(y_test > 1e-10).sum()} positive)")

    # Train
    model = CDMSupervisedRiskModel()
    train_metrics = model.train(X_train, y_train, weights=w_train)
    print(f"  Train RMSE (log10 Pc): {train_metrics['train_rmse_log10pc']:.3f}")

    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\n  Test metrics:")
    print(f"    RMSE (log10 Pc): {test_metrics['rmse_log10pc']:.3f}")
    print(f"    MAE  (log10 Pc): {test_metrics['mae_log10pc']:.3f}")
    print(f"    AUC-PR:          {test_metrics['auc_pr']:.3f}")
    print(f"    AUC-ROC:         {test_metrics['auc_roc']:.3f}")
    print(f"    F1 @ 0.5:        {test_metrics['f1_at_05']:.3f}")

    return model, train_metrics, test_metrics


def generate_learning_curve(
    X: pd.DataFrame,
    y_pc: np.ndarray,
    weights: np.ndarray,
    fractions: list[float] = None,
) -> list[dict]:
    """Generate learning curve: AUC-PR vs training set size."""
    from sklearn.model_selection import train_test_split

    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    y_binary = (y_pc > 1e-10).astype(int)

    # Fixed test set (20%)
    X_pool, X_test, y_pool, y_test, w_pool, w_test = train_test_split(
        X, y_pc, weights,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )

    curve = []
    for frac in fractions:
        n = max(int(len(X_pool) * frac), 10)
        idx = np.random.RandomState(42).choice(len(X_pool), size=n, replace=False)
        X_sub = X_pool.iloc[idx]
        y_sub = y_pool[idx]
        w_sub = w_pool[idx]

        model = CDMSupervisedRiskModel()
        model.train(X_sub, y_sub, weights=w_sub, n_estimators=200)
        metrics = model.evaluate(X_test, y_test)

        point = {
            "fraction": frac,
            "n_train": n,
            "n_positive_train": int((y_sub > 1e-10).sum()),
            **metrics,
        }
        curve.append(point)
        print(f"    {frac:.0%} ({n} samples): AUC-PR={metrics['auc_pr']:.3f}")

    return curve


def main():
    parser = argparse.ArgumentParser(description="Train CSPR pairwise risk model")
    parser.add_argument("--min-cdms", type=int, default=100,
                        help="Minimum CDMs required to train")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Negative samples per positive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check data availability only")
    parser.add_argument("--no-learning-curve", action="store_true",
                        help="Skip learning curve generation")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  CSPR Pairwise Risk Model Training")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # 1. Load CDMs
    print("Loading CDM store ...")
    cdms = _load_local_cdms(CDM_STORE)
    print(f"  Total CDMs: {len(cdms)}")

    if len(cdms) < args.min_cdms:
        print(f"\n  Only {len(cdms)} CDMs (need {args.min_cdms}). "
              f"Accumulate more data or use --min-cdms {len(cdms)}")
        return

    # 2. Load TLE snapshots
    print("\nLoading TLE snapshots ...")
    snapshots = load_tle_snapshots()

    if not snapshots:
        print("  No TLE snapshots found. Run daily_predictions.py first.")
        return

    if args.dry_run:
        print(f"\n  Dry run: {len(cdms)} CDMs, {len(snapshots)} snapshots.")
        print(f"  Would train with --neg-ratio {args.neg_ratio}")
        return

    # 3. Build training data
    print("\nBuilding training data ...")
    X, y_pc, weights = build_training_data(cdms, snapshots, neg_ratio=args.neg_ratio)

    # 4. Train and evaluate
    print("\nTraining model ...")
    model, train_metrics, test_metrics = train_and_evaluate(X, y_pc, weights)

    # 5. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "pairwise_risk.json"
    model.save(model_path)

    # 6. Save feature importance
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    importance = model.feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  Top-10 feature importance:")
    for feat, imp in sorted_imp[:10]:
        print(f"    {feat:35s}: {imp:.4f}")

    imp_path = RESULTS_DIR / "pairwise_feature_importance.json"
    with open(imp_path, "w") as f:
        json.dump({"features": sorted_imp, "metrics": test_metrics}, f, indent=2)
    print(f"  Saved feature importance to {imp_path}")

    # 7. Learning curve
    if not args.no_learning_curve:
        print("\nGenerating learning curve ...")
        curve = generate_learning_curve(X, y_pc, weights)
        curve_path = RESULTS_DIR / "pairwise_learning_curve.json"
        with open(curve_path, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"  Saved learning curve to {curve_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  CDMs used: {len(cdms)}")
    print(f"  Training samples: {X.shape[0]} ({(y_pc > 1e-10).sum()} positive)")
    print(f"  Test AUC-PR: {test_metrics['auc_pr']:.3f}")
    print(f"  Model saved: {model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
