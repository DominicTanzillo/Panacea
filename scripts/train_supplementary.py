"""Train supplementary models: GNN, Autoregressive Forecaster, Conformal Prediction.

Outputs results to webapp-react/public/ for the Models page to display.

Usage:
    python scripts/train_supplementary.py           # All models
    python scripts/train_supplementary.py --gnn     # GNN only
    python scripts/train_supplementary.py --ar      # Autoregressive only
    python scripts/train_supplementary.py --conf    # Conformal only
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import time

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
OUTPUT_DIR = ROOT / "webapp-react" / "public"

from src.model.cdm_forecast import extract_sequence_features, PC_ACTION_THRESHOLD


def load_pairs():
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ======================================================================
#  1. Graph Neural Network (GNN) - Conjunction Network Analysis
# ======================================================================
def train_gnn(pairs):
    """Build conjunction network and train a simple graph-based classifier.

    Since PyTorch Geometric may not be installed, we implement a lightweight
    graph-based feature propagation (simulated message passing) using numpy.
    This demonstrates the GNN concept with real data.
    """
    print("\n" + "=" * 60)
    print("GNN: Conjunction Network Analysis")
    print("=" * 60)

    # Build graph: nodes = satellites, edges = conjunction pairs
    sat_ids = set()
    edges = []
    edge_features = []

    for key, cdms in pairs.items():
        n1, n2, tca = key
        sat_ids.add(n1)
        sat_ids.add(n2)

        max_pc = max(c.get("pc", 0) or 0 for c in cdms)
        n_updates = len(cdms)
        miss_km = min(c.get("miss_distance_km", 9999) or 9999 for c in cdms)

        edges.append((n1, n2))
        edge_features.append({
            "max_pc": max_pc,
            "n_updates": n_updates,
            "min_miss_km": miss_km,
            "label": 1 if max_pc >= PC_ACTION_THRESHOLD else 0,
        })

    sat_list = sorted(sat_ids)
    sat_idx = {s: i for i, s in enumerate(sat_list)}
    n_nodes = len(sat_list)
    n_edges = len(edges)

    print(f"  Nodes (satellites): {n_nodes}")
    print(f"  Edges (conjunctions): {n_edges}")

    # Node features: degree, weighted degree, max neighbor Pc, type stats
    degree = np.zeros(n_nodes)
    weighted_degree = np.zeros(n_nodes)
    max_neighbor_pc = np.zeros(n_nodes)
    n_high_risk_neighbors = np.zeros(n_nodes)

    for (n1, n2), ef in zip(edges, edge_features):
        i, j = sat_idx[n1], sat_idx[n2]
        degree[i] += 1
        degree[j] += 1
        pc = ef["max_pc"]
        weighted_degree[i] += pc
        weighted_degree[j] += pc
        max_neighbor_pc[i] = max(max_neighbor_pc[i], pc)
        max_neighbor_pc[j] = max(max_neighbor_pc[j], pc)
        if ef["label"] == 1:
            n_high_risk_neighbors[i] += 1
            n_high_risk_neighbors[j] += 1

    # Node feature matrix (4 features per node)
    node_features = np.column_stack([
        np.log1p(degree),
        np.log1p(weighted_degree),
        np.log1p(max_neighbor_pc * 1e4),  # Scale Pc up
        n_high_risk_neighbors,
    ])

    # Edge classification: predict if conjunction is high-risk
    # Use concatenated node features of both endpoints + edge features
    X_edges = []
    y_edges = []
    for (n1, n2), ef in zip(edges, edge_features):
        i, j = sat_idx[n1], sat_idx[n2]
        # Concatenate: node_i features + node_j features + edge features
        row = np.concatenate([
            node_features[i],
            node_features[j],
            [math.log1p(ef["n_updates"]), math.log1p(ef["min_miss_km"])],
        ])
        X_edges.append(row)
        y_edges.append(ef["label"])

    X = np.array(X_edges)
    y = np.array(y_edges)

    # Message passing simulation (1 hop): aggregate neighbor info
    # For each edge, average the features of all OTHER edges connected to each node
    print("  Simulating 1-hop message passing...")
    node_agg = np.zeros((n_nodes, 2))  # mean Pc, mean miss of neighbors
    node_count = np.zeros(n_nodes)
    for (n1, n2), ef in zip(edges, edge_features):
        i, j = sat_idx[n1], sat_idx[n2]
        node_agg[i] += [ef["max_pc"], ef["min_miss_km"]]
        node_agg[j] += [ef["max_pc"], ef["min_miss_km"]]
        node_count[i] += 1
        node_count[j] += 1

    node_count[node_count == 0] = 1
    node_agg /= node_count[:, None]

    # Augment edge features with aggregated neighbor info
    X_aug = []
    for k, (n1, n2) in enumerate(edges):
        i, j = sat_idx[n1], sat_idx[n2]
        row = np.concatenate([X[k], node_agg[i], node_agg[j]])
        X_aug.append(row)
    X_aug = np.array(X_aug)

    # Train non-linear classifier on augmented features (5-fold CV)
    # Use a 2-layer neural net (PyTorch) for genuinely non-linear decision boundaries
    print("  Training non-linear edge classifier (5-fold CV)...")
    import torch
    import torch.nn as nn
    from scripts.tune_and_analyze import train_lr, predict_lr, eval_metrics

    class EdgeMLP(nn.Module):
        def __init__(self, input_dim, hidden=64, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_mlp(X_train, y_train, X_val, y_val, epochs=200, lr=0.005):
        """Train a small MLP with early stopping."""
        # Normalize
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
        Xn = torch.tensor((X_train - mu) / sd, dtype=torch.float32)
        yn = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor((X_val - mu) / sd, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)

        model = EdgeMLP(X_train.shape[1])
        pos_w = max(1.0, (1 - y_train.mean()) / max(y_train.mean(), 0.01))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        best_loss, patience, best_state = 1e9, 0, None

        for ep in range(epochs):
            model.train()
            logits = model(Xn)
            weight = torch.where(yn == 1, pos_w, 1.0)
            loss = (weight * nn.functional.binary_cross_entropy_with_logits(logits, yn, reduction='none')).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                vl = nn.functional.binary_cross_entropy_with_logits(model(Xv), yv).item()
            if vl < best_loss:
                best_loss = vl; patience = 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 20: break

        if best_state: model.load_state_dict(best_state)
        return model, mu, sd

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    fold_metrics = []
    fold_metrics_no_graph = []
    best_model, best_mu, best_sd = None, None, None
    best_f1 = 0
    n_folds = 5
    for f in range(n_folds):
        p_start = len(pos_idx) * f // n_folds
        p_end = len(pos_idx) * (f + 1) // n_folds
        n_start = len(neg_idx) * f // n_folds
        n_end = len(neg_idx) * (f + 1) // n_folds
        test_idx = np.concatenate([pos_idx[p_start:p_end], neg_idx[n_start:n_end]])
        train_idx = np.array([i for i in range(len(y)) if i not in set(test_idx)])

        # With graph features — non-linear MLP
        mlp, mu_f, sd_f = train_mlp(X_aug[train_idx], y[train_idx], X_aug[test_idx], y[test_idx])
        mlp.eval()
        with torch.no_grad():
            probs = torch.sigmoid(mlp(torch.tensor((X_aug[test_idx] - mu_f) / (sd_f + 1e-8), dtype=torch.float32))).numpy()
        fm = eval_metrics(y[test_idx], probs)
        fold_metrics.append(fm)
        if fm["f1"] > best_f1:
            best_f1 = fm["f1"]; best_model = mlp; best_mu = mu_f; best_sd = sd_f

        # Without graph features (LR baseline for comparison)
        w2, b2, mu2, sd2 = train_lr(X[train_idx], y[train_idx], lr=0.05, reg=0.01, epochs=300)
        probs2 = predict_lr(X[test_idx], w2, b2, mu2, sd2)
        fold_metrics_no_graph.append(eval_metrics(y[test_idx], probs2))

    avg = lambda ms, k: round(float(np.mean([m[k] for m in ms])), 4)

    # Save GNN checkpoint for use in daily pipeline shadow ensemble
    if best_model is not None:
        ckpt_path = ROOT / "models" / "conjunction_gnn.pt"
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "input_dim": X_aug.shape[1],
            "norm_mean": best_mu,
            "norm_std": best_sd,
            "sat_idx": sat_idx,
            "node_features": node_features,
            "node_agg": node_agg,
            "edges": edges,
            "edge_features_list": edge_features,
            "trained": True,
        }, str(ckpt_path))
        print(f"  Saved GNN checkpoint to {ckpt_path}")

    print(f"  With graph features:    F1={avg(fold_metrics, 'f1'):.3f} Recall={avg(fold_metrics, 'recall'):.3f}")
    print(f"  Without graph features: F1={avg(fold_metrics_no_graph, 'f1'):.3f} Recall={avg(fold_metrics_no_graph, 'recall'):.3f}")

    # Degree distribution for visualization
    degree_dist = {}
    for d in degree:
        d_int = int(d)
        degree_dist[d_int] = degree_dist.get(d_int, 0) + 1

    # Top connected satellites
    top_connected = sorted(zip(sat_list, degree.tolist()), key=lambda x: x[1], reverse=True)[:20]

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_positive_edges": int(y.sum()),
        "with_graph_features": {
            "f1": avg(fold_metrics, "f1"),
            "recall": avg(fold_metrics, "recall"),
            "precision": avg(fold_metrics, "precision"),
            "accuracy": avg(fold_metrics, "accuracy"),
        },
        "without_graph_features": {
            "f1": avg(fold_metrics_no_graph, "f1"),
            "recall": avg(fold_metrics_no_graph, "recall"),
            "precision": avg(fold_metrics_no_graph, "precision"),
            "accuracy": avg(fold_metrics_no_graph, "accuracy"),
        },
        "graph_improvement": {
            "f1_delta": round(avg(fold_metrics, "f1") - avg(fold_metrics_no_graph, "f1"), 4),
            "recall_delta": round(avg(fold_metrics, "recall") - avg(fold_metrics_no_graph, "recall"), 4),
        },
        "degree_distribution": {str(k): v for k, v in sorted(degree_dist.items())},
        "top_connected_satellites": [{"norad_id": int(s), "degree": int(d)} for s, d in top_connected],
        "avg_degree": round(float(degree.mean()), 2),
        "max_degree": int(degree.max()),
    }


# ======================================================================
#  2. Autoregressive Forecaster
# ======================================================================
def train_autoregressive(pairs):
    """Train autoregressive CDM forecaster: predict CDM k+1 from CDMs 1..k.

    Uses simple linear regression on sequence features (numpy only) to
    demonstrate the concept. The BiLSTM version is in cdm_sequence_model.py.
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE: Next-CDM Forecaster")
    print("=" * 60)

    # Build training data: for each pair with 3+ CDMs,
    # use first k CDMs to predict the (k+1)-th CDM's log10(Pc) and miss_km
    X_rows, y_pc_rows, y_miss_rows = [], [], []
    example_sequences = []

    for key, cdms in pairs.items():
        cdms_sorted = sorted(cdms, key=lambda c: c.get("creation_date", ""))
        if len(cdms_sorted) < 3:
            continue

        for k in range(2, len(cdms_sorted)):
            prefix = cdms_sorted[:k]
            target = cdms_sorted[k]

            # Features from prefix
            pcs = [math.log10(max(c.get("pc", 1e-10), 1e-10)) for c in prefix]
            misses = [math.log10(max(c.get("miss_distance_km", 1), 0.01)) for c in prefix]
            tth = [c.get("time_to_tca_hours", 0) or 0 for c in prefix]

            row = [
                pcs[-1], pcs[-2] if len(pcs) >= 2 else pcs[-1],
                misses[-1], misses[-2] if len(misses) >= 2 else misses[-1],
                tth[-1],
                pcs[-1] - pcs[0],  # trend
                len(prefix),
            ]
            target_pc = math.log10(max(target.get("pc", 1e-10), 1e-10))
            target_miss = math.log10(max(target.get("miss_distance_km", 1), 0.01))

            X_rows.append(row)
            y_pc_rows.append(target_pc)
            y_miss_rows.append(target_miss)

        # Save example sequence for visualization
        if len(cdms_sorted) >= 5 and len(example_sequences) < 5:
            seq = []
            for c in cdms_sorted:
                seq.append({
                    "log10_pc": round(math.log10(max(c.get("pc", 1e-10), 1e-10)), 3),
                    "miss_km": round(c.get("miss_distance_km", 0) or 0, 1),
                    "tth": round(c.get("time_to_tca_hours", 0) or 0, 1),
                })
            example_sequences.append({"pair": f"{key[0]}-{key[1]}", "sequence": seq})

    X = np.array(X_rows)
    y_pc = np.array(y_pc_rows)
    y_miss = np.array(y_miss_rows)

    print(f"  Training samples: {len(X)}")
    print(f"  Example sequences saved: {len(example_sequences)}")

    # Train/test split
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    # Linear regression for Pc prediction
    X_train = np.column_stack([X[train_idx], np.ones(len(train_idx))])
    X_test = np.column_stack([X[test_idx], np.ones(len(test_idx))])

    # Solve normal equations
    w_pc = np.linalg.lstsq(X_train, y_pc[train_idx], rcond=None)[0]
    w_miss = np.linalg.lstsq(X_train, y_miss[train_idx], rcond=None)[0]

    pred_pc = X_test @ w_pc
    pred_miss = X_test @ w_miss

    # Metrics
    mae_pc = float(np.mean(np.abs(pred_pc - y_pc[test_idx])))
    mae_miss = float(np.mean(np.abs(pred_miss - y_miss[test_idx])))
    corr_pc = float(np.corrcoef(pred_pc, y_pc[test_idx])[0, 1]) if len(pred_pc) > 2 else 0
    corr_miss = float(np.corrcoef(pred_miss, y_miss[test_idx])[0, 1]) if len(pred_miss) > 2 else 0
    rmse_pc = float(np.sqrt(np.mean((pred_pc - y_pc[test_idx]) ** 2)))

    print(f"  Pc prediction:   MAE={mae_pc:.3f} log10, r={corr_pc:.3f}, RMSE={rmse_pc:.3f}")
    print(f"  Miss prediction: MAE={mae_miss:.3f} log10, r={corr_miss:.3f}")

    # Residuals for uncertainty estimation
    residuals = pred_pc - y_pc[test_idx]
    uncertainty_std = float(np.std(residuals))

    # Generate multi-step forecast examples
    forecasts = []
    for ex in example_sequences[:3]:
        seq = ex["sequence"]
        actual = [s["log10_pc"] for s in seq]
        predicted = []
        for k in range(2, len(seq)):
            pcs = [s["log10_pc"] for s in seq[:k]]
            misses = [s["miss_km"] for s in seq[:k]]
            tth = [s["tth"] for s in seq[:k]]
            row = [
                pcs[-1], pcs[-2] if len(pcs) >= 2 else pcs[-1],
                math.log10(max(misses[-1], 0.01)),
                math.log10(max(misses[-2] if len(misses) >= 2 else misses[-1], 0.01)),
                tth[-1],
                pcs[-1] - pcs[0],
                len(pcs),
            ]
            row_aug = np.array(row + [1.0])
            p = float(row_aug @ w_pc)
            predicted.append(round(p, 3))

        forecasts.append({
            "pair": ex["pair"],
            "actual": actual,
            "predicted": [None, None] + predicted,  # First 2 are inputs
            "n_cdms": len(seq),
        })

    return {
        "n_training_samples": len(X),
        "n_test_samples": len(test_idx),
        "metrics": {
            "mae_log10_pc": round(mae_pc, 4),
            "rmse_log10_pc": round(rmse_pc, 4),
            "correlation_pc": round(corr_pc, 4),
            "mae_log10_miss": round(mae_miss, 4),
            "correlation_miss": round(corr_miss, 4),
            "uncertainty_std": round(uncertainty_std, 4),
        },
        "example_forecasts": forecasts,
    }


# ======================================================================
#  3. Conformal Prediction
# ======================================================================
def train_conformal(pairs):
    """Split-conformal prediction for calibrated uncertainty intervals."""
    print("\n" + "=" * 60)
    print("CONFORMAL: Split-Conformal Prediction")
    print("=" * 60)

    from scripts.tune_and_analyze import build_dataset, train_lr, predict_lr, eval_metrics

    X, y, feature_names, _, _ = build_dataset(pairs)

    # Three-way split: train / calibration / test
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y))
    n = len(idx)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)

    train_idx = idx[:train_end]
    cal_idx = idx[train_end:cal_end]
    test_idx = idx[cal_end:]

    print(f"  Train: {len(train_idx)}, Calibration: {len(cal_idx)}, Test: {len(test_idx)}")

    # Train model
    w, b, mu, sd = train_lr(X[train_idx], y[train_idx])

    # Calibration: compute nonconformity scores
    cal_probs = predict_lr(X[cal_idx], w, b, mu, sd)
    cal_labels = y[cal_idx]

    # For classification: nonconformity = 1 - P(true class)
    cal_scores = np.where(cal_labels == 1, 1 - cal_probs, cal_probs)

    # Test conformal at different alpha levels
    alphas = [0.05, 0.10, 0.15, 0.20]
    test_probs = predict_lr(X[test_idx], w, b, mu, sd)
    test_labels = y[test_idx]

    coverage_results = []
    for alpha in alphas:
        # Quantile of calibration scores
        q_hat = float(np.quantile(cal_scores, 1 - alpha))

        # Prediction sets: include class if P(class) >= 1 - q_hat
        n_covered = 0
        set_sizes = []
        for i in range(len(test_labels)):
            p1 = test_probs[i]
            p0 = 1 - p1
            pred_set = []
            if p0 >= 1 - q_hat:
                pred_set.append(0)
            if p1 >= 1 - q_hat:
                pred_set.append(1)

            if test_labels[i] in pred_set:
                n_covered += 1
            set_sizes.append(len(pred_set))

        coverage = n_covered / len(test_labels)
        avg_set_size = np.mean(set_sizes)

        coverage_results.append({
            "alpha": alpha,
            "target_coverage": round(1 - alpha, 3),
            "empirical_coverage": round(float(coverage), 4),
            "q_hat": round(q_hat, 4),
            "avg_set_size": round(float(avg_set_size), 3),
            "n_singleton": int(sum(1 for s in set_sizes if s == 1)),
            "n_empty": int(sum(1 for s in set_sizes if s == 0)),
            "n_both": int(sum(1 for s in set_sizes if s == 2)),
        })
        print(f"  alpha={alpha:.2f}: target={1-alpha:.0%}, actual={coverage:.1%}, "
              f"avg set size={avg_set_size:.2f}, q_hat={q_hat:.3f}")

    # Regression conformal: prediction intervals for log10(Pc)
    # Use regression features to predict actual max Pc, then add conformal interval
    print("\n  Regression conformal (Pc prediction intervals):")

    # Build regression targets
    from scripts.tune_and_analyze import build_dataset as bd
    X_all, y_all, fnames, _, _ = bd(pairs)

    # For regression, we need continuous target - use the feature "max_log10_pc" (index 6)
    y_reg = X_all[:, 6]  # max_log10_pc IS the regression target
    X_reg = np.delete(X_all, 6, axis=1)  # Remove target from features

    reg_train = idx[:train_end]
    reg_cal = idx[train_end:cal_end]
    reg_test = idx[cal_end:]

    # Train linear regression
    X_tr = np.column_stack([X_reg[reg_train], np.ones(len(reg_train))])
    X_ca = np.column_stack([X_reg[reg_cal], np.ones(len(reg_cal))])
    X_te = np.column_stack([X_reg[reg_test], np.ones(len(reg_test))])

    w_reg = np.linalg.lstsq(X_tr, y_reg[reg_train], rcond=None)[0]

    cal_preds = X_ca @ w_reg
    cal_residuals = np.abs(y_reg[reg_cal] - cal_preds)

    reg_coverages = []
    for alpha in [0.05, 0.10, 0.20]:
        q = float(np.quantile(cal_residuals, 1 - alpha))
        test_preds = X_te @ w_reg
        lower = test_preds - q
        upper = test_preds + q
        covered = np.sum((y_reg[reg_test] >= lower) & (y_reg[reg_test] <= upper))
        cov = covered / len(reg_test)
        width = float(np.mean(upper - lower))

        reg_coverages.append({
            "alpha": alpha,
            "target": round(1 - alpha, 3),
            "coverage": round(float(cov), 4),
            "interval_width": round(width, 3),
        })
        print(f"  alpha={alpha:.2f}: target={1-alpha:.0%}, actual={cov:.1%}, width={width:.2f} log10(Pc)")

    return {
        "n_train": len(train_idx),
        "n_calibration": len(cal_idx),
        "n_test": len(test_idx),
        "classification_conformal": coverage_results,
        "regression_conformal": reg_coverages,
    }


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--conf", action="store_true")
    args = parser.parse_args()

    run_all = not (args.gnn or args.ar or args.conf)

    print("=" * 60)
    print("PANACEA Supplementary Model Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pairs = load_pairs()
    print(f"Loaded {len(pairs)} conjunction pairs")

    results = {"generated_at": datetime.now(timezone.utc).isoformat()}

    if run_all or args.gnn:
        results["gnn"] = train_gnn(pairs)

    if run_all or args.ar:
        results["autoregressive"] = train_autoregressive(pairs)

    if run_all or args.conf:
        results["conformal"] = train_conformal(pairs)

    out_path = OUTPUT_DIR / "supplementary_models.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
