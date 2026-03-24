"""Hyperparameter tuning, feature importance, and temporal split analysis.

Runs overnight and outputs results to webapp-react/public/tuning_results.json
for the Pipeline page to display.

Usage:
    python scripts/tune_and_analyze.py              # Full suite
    python scripts/tune_and_analyze.py --quick      # Reduced grid for testing
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
OUTPUT_PATH = ROOT / "webapp-react" / "public" / "tuning_results.json"

from src.model.cdm_forecast import (
    extract_sequence_features,
    PC_ACTION_THRESHOLD,
)

# Space weather support (optional)
try:
    from src.data.space_weather import SpaceWeatherCache, normalize_space_weather
    _HAS_SPACE_WEATHER = True
except ImportError:
    _HAS_SPACE_WEATHER = False


# ── Data Loading ──────────────────────────────────────────────
def load_pairs():
    """Load CDM pairs from the store, group by conjunction pair + TCA date."""
    if not CDM_STORE.exists():
        print(f"ERROR: CDM store not found at {CDM_STORE}")
        sys.exit(1)

    pairs = defaultdict(list)
    with open(CDM_STORE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            n1 = c.get("sat1_norad", 0)
            n2 = c.get("sat2_norad", 0)
            tca = c.get("tca", "")[:10]
            if n1 and n2:
                pairs[(min(n1, n2), max(n1, n2), tca)].append(c)
    return dict(pairs)


def build_dataset(pairs: dict) -> tuple:
    """Build feature matrix and labels from CDM pairs.

    Returns: (X, y, feature_names, pair_keys, tca_dates)
    """
    feature_names = [
        "latest_log10_pc", "latest_log10_miss", "latest_tth",
        "pc_trend", "miss_trend", "pc_acceleration",
        "max_log10_pc", "min_log10_pc", "log_n_updates",
        "sat1_rcs", "sat2_rcs", "has_debris",
        "pc_range", "miss_range", "pc_last_delta", "miss_last_delta",
        "max_pc_rate", "time_coverage", "first_log10_pc", "emergency",
        "f107_norm", "kp_norm", "ap_norm",
    ]

    # Prefetch space weather indices for all CDM dates
    sw_cache = None
    if _HAS_SPACE_WEATHER:
        try:
            sw_cache = SpaceWeatherCache()
            all_dates = set()
            for cdms in pairs.values():
                for c in cdms:
                    cd = c.get("creation_date", "")[:10]
                    if cd:
                        all_dates.add(cd)
            if all_dates:
                sw_cache.bulk_lookup(sorted(all_dates))
        except Exception:
            sw_cache = None

    X_rows, y_rows, keys, tca_dates = [], [], [], []

    for key, cdms in pairs.items():
        cdms_sorted = sorted(cdms, key=lambda c: c.get("creation_date", ""))
        feat = extract_sequence_features(cdms_sorted)
        if feat is None:
            continue

        static = feat["static"]
        seq = feat["sequence"]
        if seq is None or len(seq) == 0:
            continue

        # Build 20-feature vector (same as CDMForecast.train)
        latest = seq[-1]
        log10_pc = latest[0]
        log10_miss = latest[1]
        tth = latest[2]

        # Trends
        if len(seq) >= 2:
            pcs = [s[0] for s in seq]
            miss_vals = [s[1] for s in seq]
            tth_vals = [s[2] for s in seq]
            n = len(pcs)
            x_lin = np.arange(n, dtype=float)
            pc_trend = np.polyfit(x_lin, pcs, 1)[0] if n >= 2 else 0.0
            miss_trend = miss_vals[-1] - miss_vals[0]
            if n >= 3:
                d1 = np.diff(pcs)
                pc_accel = np.mean(np.diff(d1)) if len(d1) >= 2 else 0.0
            else:
                pc_accel = 0.0
            pc_last_delta = pcs[-1] - pcs[-2]
            miss_last_delta = miss_vals[-1] - miss_vals[-2]
            # Max Pc rate
            rates = []
            for i in range(1, n):
                dt = abs(tth_vals[i-1] - tth_vals[i])
                if dt > 0:
                    rates.append(abs(pcs[i] - pcs[i-1]) / dt)
            max_pc_rate = max(rates) if rates else 0.0
            # Time coverage
            if tth_vals[0] > 0:
                time_coverage = (tth_vals[0] - tth_vals[-1]) / tth_vals[0]
            else:
                time_coverage = 0.0
        else:
            pc_trend = miss_trend = pc_accel = 0.0
            pc_last_delta = miss_last_delta = 0.0
            max_pc_rate = time_coverage = 0.0

        max_pc = max(s[0] for s in seq)
        min_pc = min(s[0] for s in seq)
        pc_range = max_pc - min_pc
        max_miss = max(s[1] for s in seq)
        min_miss = min(s[1] for s in seq)
        miss_range = max_miss - min_miss

        # Space weather features
        sw_f107, sw_kp, sw_ap = 0.5, 0.25, 0.15  # neutral defaults
        if sw_cache:
            latest_date = cdms_sorted[-1].get("creation_date", "")[:10]
            if latest_date:
                try:
                    sw = sw_cache.get_indices(latest_date)
                    sw_f107, sw_kp, sw_ap = normalize_space_weather(
                        sw.get("f107", 155), sw.get("kp", 2.3), sw.get("ap", 10))
                except Exception:
                    pass

        row = [
            log10_pc, log10_miss, tth,
            pc_trend, miss_trend, pc_accel,
            max_pc, min_pc, math.log1p(len(seq)),
            static.get("sat1_rcs", 1), static.get("sat2_rcs", 1),
            static.get("has_debris", 0),
            pc_range, miss_range, pc_last_delta, miss_last_delta,
            max_pc_rate, time_coverage,
            seq[0][0] if len(seq) > 0 else log10_pc,
            static.get("emergency", 0),
            sw_f107, sw_kp, sw_ap,
        ]

        # Label: max Pc in sequence exceeds threshold
        max_pc_val = max(c.get("pc", 0) or 0 for c in cdms_sorted)
        label = 1 if max_pc_val >= PC_ACTION_THRESHOLD else 0

        # TCA date for temporal splitting
        tca_str = cdms_sorted[-1].get("tca", "")[:10]

        X_rows.append(row)
        y_rows.append(label)
        keys.append(key)
        tca_dates.append(tca_str)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, feature_names, keys, tca_dates


# ── Logistic Regression (pure numpy, matching CDMForecast) ────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_lr(X_train, y_train, lr=0.05, reg=0.01, epochs=300, patience=30, pos_weight=None):
    """Train logistic regression with gradient descent."""
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0

    if pos_weight is None:
        pos_rate = y_train.mean()
        pos_weight = math.sqrt((1 - pos_rate) / max(pos_rate, 1e-6))

    # Z-score normalization
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    X_norm = (X_train - mu) / sd

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        logits = X_norm @ w + b
        preds = sigmoid(logits)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)

        weights = np.where(y_train == 1, pos_weight, 1.0)
        loss = -np.mean(weights * (y_train * np.log(preds) + (1 - y_train) * np.log(1 - preds)))
        loss += reg * np.sum(w ** 2)

        if loss < best_loss - 1e-6:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        residuals = weights * (preds - y_train)
        grad_w = (X_norm.T @ residuals) / n + 2 * reg * w
        grad_b = np.mean(residuals)
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mu, sd


def predict_lr(X, w, b, mu, sd):
    X_norm = (X - mu) / sd
    return sigmoid(X_norm @ w + b)


def eval_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    # AUC-PR (trapezoidal approximation)
    thresholds = np.linspace(0, 1, 200)
    precs, recs = [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp_t = np.sum((yp == 1) & (y_true == 1))
        fp_t = np.sum((yp == 1) & (y_true == 0))
        fn_t = np.sum((yp == 0) & (y_true == 1))
        p = tp_t / max(tp_t + fp_t, 1)
        r = tp_t / max(tp_t + fn_t, 1)
        precs.append(p)
        recs.append(r)
    # Sort by recall ascending
    pairs = sorted(zip(recs, precs))
    auc_pr = 0.0
    for i in range(1, len(pairs)):
        dr = pairs[i][0] - pairs[i - 1][0]
        auc_pr += dr * (pairs[i][1] + pairs[i - 1][1]) / 2

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "auc_pr": auc_pr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ══════════════════════════════════════════════════════════════
#  1. Hyperparameter Grid Search
# ══════════════════════════════════════════════════════════════
def run_grid_search(X, y, feature_names, quick=False):
    print("\n" + "=" * 60)
    print("PHASE 1: Hyperparameter Grid Search (Logistic Regression)")
    print("=" * 60)

    if quick:
        lr_values = [0.01, 0.05, 0.1]
        reg_values = [0.001, 0.01, 0.1]
        epoch_values = [300]
    else:
        lr_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        reg_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        epoch_values = [200, 300, 500]

    grid = list(product(lr_values, reg_values, epoch_values))
    print(f"Grid size: {len(grid)} configurations")
    print(f"5-fold CV on each -> {len(grid) * 5} total model trainings")

    # 5-fold stratified split
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_folds = 5

    folds = []
    for f in range(n_folds):
        p_start = len(pos_idx) * f // n_folds
        p_end = len(pos_idx) * (f + 1) // n_folds
        n_start = len(neg_idx) * f // n_folds
        n_end = len(neg_idx) * (f + 1) // n_folds
        test_idx = np.concatenate([pos_idx[p_start:p_end], neg_idx[n_start:n_end]])
        train_idx = np.array([i for i in range(n) if i not in set(test_idx)])
        folds.append((train_idx, test_idx))

    results = []
    best_f1 = 0
    best_recall = 0
    best_config = None
    best_recall_config = None

    for i, (lr, reg, epochs) in enumerate(grid):
        fold_metrics = []
        for train_idx, test_idx in folds:
            w, b, mu, sd = train_lr(X[train_idx], y[train_idx], lr=lr, reg=reg, epochs=epochs)
            probs = predict_lr(X[test_idx], w, b, mu, sd)
            m = eval_metrics(y[test_idx], probs)
            fold_metrics.append(m)

        avg_f1 = np.mean([m["f1"] for m in fold_metrics])
        avg_recall = np.mean([m["recall"] for m in fold_metrics])
        avg_precision = np.mean([m["precision"] for m in fold_metrics])
        avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
        avg_auc = np.mean([m["auc_pr"] for m in fold_metrics])

        entry = {
            "lr": lr, "reg": reg, "epochs": epochs,
            "f1": round(avg_f1, 4), "recall": round(avg_recall, 4),
            "precision": round(avg_precision, 4), "accuracy": round(avg_accuracy, 4),
            "auc_pr": round(avg_auc, 4),
        }
        results.append(entry)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_config = entry
        if avg_recall > best_recall:
            best_recall = avg_recall
            best_recall_config = entry

        if (i + 1) % 10 == 0 or i == len(grid) - 1:
            print(f"  [{i+1}/{len(grid)}] lr={lr} reg={reg} ep={epochs} -> F1={avg_f1:.3f} Recall={avg_recall:.3f}")

    # Sort by recall (primary), then F1 (secondary)
    results.sort(key=lambda r: (r["recall"], r["f1"]), reverse=True)

    print(f"\n  Best F1:     {best_config}")
    print(f"  Best Recall: {best_recall_config}")

    return {
        "grid_size": len(grid),
        "n_folds": n_folds,
        "best_by_f1": best_config,
        "best_by_recall": best_recall_config,
        "top_10": results[:10],
        "all_results": results,
        "current_config": {"lr": 0.05, "reg": 0.01, "epochs": 300},
    }


# ══════════════════════════════════════════════════════════════
#  2. Feature Importance (Permutation)
# ══════════════════════════════════════════════════════════════
def run_feature_importance(X, y, feature_names, n_repeats=10):
    print("\n" + "=" * 60)
    print("PHASE 2: Permutation Feature Importance")
    print("=" * 60)

    # Train on full data, evaluate on full data (for importance, not generalization)
    # Actually, use proper held-out evaluation
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    w, b, mu, sd = train_lr(X[train_idx], y[train_idx])
    base_probs = predict_lr(X[test_idx], w, b, mu, sd)
    base_metrics = eval_metrics(y[test_idx], base_probs)
    base_f1 = base_metrics["f1"]
    base_recall = base_metrics["recall"]

    print(f"  Baseline: F1={base_f1:.3f}, Recall={base_recall:.3f}")

    importances = []
    for j, fname in enumerate(feature_names):
        f1_drops = []
        recall_drops = []
        for rep in range(n_repeats):
            X_test_perm = X[test_idx].copy()
            rng2 = np.random.RandomState(42 + rep)
            X_test_perm[:, j] = rng2.permutation(X_test_perm[:, j])
            perm_probs = predict_lr(X_test_perm, w, b, mu, sd)
            perm_m = eval_metrics(y[test_idx], perm_probs)
            f1_drops.append(base_f1 - perm_m["f1"])
            recall_drops.append(base_recall - perm_m["recall"])

        importances.append({
            "feature": fname,
            "f1_drop_mean": round(float(np.mean(f1_drops)), 4),
            "f1_drop_std": round(float(np.std(f1_drops)), 4),
            "recall_drop_mean": round(float(np.mean(recall_drops)), 4),
            "recall_drop_std": round(float(np.std(recall_drops)), 4),
        })
        print(f"  [{j+1}/{len(feature_names)}] {fname:25s} -> F1 drop: {np.mean(f1_drops):+.4f} ± {np.std(f1_drops):.4f}")

    # Sort by F1 importance descending
    importances.sort(key=lambda x: x["f1_drop_mean"], reverse=True)

    return {
        "baseline_f1": round(base_f1, 4),
        "baseline_recall": round(base_recall, 4),
        "n_repeats": n_repeats,
        "features": importances,
    }


# ══════════════════════════════════════════════════════════════
#  3. Temporal Train/Test Split
# ══════════════════════════════════════════════════════════════
def run_temporal_split(X, y, feature_names, tca_dates):
    print("\n" + "=" * 60)
    print("PHASE 3: Temporal Train/Test Split Analysis")
    print("=" * 60)

    # Sort by TCA date
    sorted_idx = np.argsort(tca_dates)
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]
    dates_sorted = [tca_dates[i] for i in sorted_idx]

    n = len(y_sorted)
    results = []

    # Try multiple temporal split points (60/40 through 90/10)
    for train_frac in [0.6, 0.7, 0.8, 0.9]:
        split = int(n * train_frac)
        X_train, X_test = X_sorted[:split], X_sorted[split:]
        y_train, y_test = y_sorted[:split], y_sorted[split:]

        if len(y_test) < 10 or y_test.sum() < 2:
            continue

        train_end_date = dates_sorted[split - 1]
        test_start_date = dates_sorted[split]

        w, b, mu, sd = train_lr(X_train, y_train)
        probs = predict_lr(X_test, w, b, mu, sd)
        m = eval_metrics(y_test, probs)

        entry = {
            "train_fraction": train_frac,
            "train_size": int(split),
            "test_size": int(n - split),
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "train_positive_rate": round(float(y_train.mean()), 3),
            "test_positive_rate": round(float(y_test.mean()), 3),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        }
        results.append(entry)
        print(f"  {train_frac:.0%} train ({split} pairs, to {train_end_date}) -> "
              f"F1={m['f1']:.3f} Recall={m['recall']:.3f} Prec={m['precision']:.3f}")

    # Compare with random split baseline
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.7)
    w, b, mu, sd = train_lr(X_sorted[idx[:split]], y_sorted[idx[:split]])
    probs = predict_lr(X_sorted[idx[split:]], w, b, mu, sd)
    random_metrics = eval_metrics(y_sorted[idx[split:]], probs)

    return {
        "n_pairs": n,
        "date_range": {"earliest": dates_sorted[0], "latest": dates_sorted[-1]},
        "temporal_results": results,
        "random_split_baseline": {
            "train_fraction": 0.7,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in random_metrics.items()},
        },
    }


# ══════════════════════════════════════════════════════════════
#  4. Feature Correlation Analysis
# ══════════════════════════════════════════════════════════════
def run_correlation_analysis(X, y, feature_names):
    print("\n" + "=" * 60)
    print("PHASE 4: Feature Correlation Analysis")
    print("=" * 60)

    # Point-biserial correlation with label
    correlations = []
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        if col.std() < 1e-8:
            correlations.append({"feature": fname, "correlation": 0.0})
            continue
        corr = np.corrcoef(col, y.astype(float))[0, 1]
        correlations.append({
            "feature": fname,
            "correlation": round(float(corr), 4),
            "mean_pos": round(float(col[y == 1].mean()), 3),
            "mean_neg": round(float(col[y == 0].mean()), 3),
        })

    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Feature-feature correlation matrix (top pairs)
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            c = abs(corr_matrix[i, j])
            if c > 0.7:
                high_corr_pairs.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "correlation": round(float(corr_matrix[i, j]), 3),
                })

    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    print(f"  Top features correlated with label:")
    for c in correlations[:5]:
        print(f"    {c['feature']:25s} r={c['correlation']:+.3f}")

    print(f"  High inter-feature correlations (|r|>0.7): {len(high_corr_pairs)} pairs")

    return {
        "label_correlations": correlations,
        "high_correlation_pairs": high_corr_pairs,
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Reduced grid for testing")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("PANACEA Model Tuning & Analysis Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    print("\nLoading CDM pairs...")
    pairs = load_pairs()
    print(f"  {len(pairs)} conjunction pairs loaded")

    X, y, feature_names, pair_keys, tca_dates = build_dataset(pairs)
    print(f"  Feature matrix: {X.shape} ({y.sum()} positive, {len(y) - y.sum()} negative)")
    print(f"  Positive rate: {y.mean():.1%}")

    # Phase 1: Grid search
    grid_results = run_grid_search(X, y, feature_names, quick=args.quick)

    # Phase 2: Feature importance
    importance_results = run_feature_importance(X, y, feature_names)

    # Phase 3: Temporal split
    temporal_results = run_temporal_split(X, y, feature_names, tca_dates)

    # Phase 4: Correlations
    correlation_results = run_correlation_analysis(X, y, feature_names)

    # Save results
    elapsed = time.time() - start_time
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "dataset": {
            "n_pairs": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "positive_rate": round(float(y.mean()), 3),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        },
        "grid_search": grid_results,
        "feature_importance": importance_results,
        "temporal_split": temporal_results,
        "correlations": correlation_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
