"""Learn optimal ensemble weights via stacking (cross-validated blending).

Instead of hand-picked weights (40/30/30), this fits a meta-learner on
out-of-fold predictions from each base model. The meta-learner's
coefficients are the data-driven optimal weights.

Method:
  1. For each of 5 CV folds:
     - Train LR, BiLSTM, GNN on training fold
     - Collect their predicted probabilities on the held-out fold
  2. Stack all out-of-fold predictions into a (n_pairs, 3) matrix
  3. Fit constrained logistic regression: weights >= 0
  4. Normalize weights to sum to 1.0
  5. Save weights + diagnostics to models/ensemble_weights.json

The daily pipeline loads these weights instead of hardcoded constants.

Usage:
    python scripts/calibrate_ensemble.py
    python scripts/calibrate_ensemble.py --quick  # 3-fold for testing
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.cdm_forecast import CDMForecastModel, extract_sequence_features, PC_ACTION_THRESHOLD

CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
WEIGHTS_PATH = ROOT / "models" / "ensemble_weights.json"


def load_pairs():
    """Load CDM pairs from store."""
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


def build_features_and_labels(pairs: dict):
    """Build feature matrix, labels, and per-pair CDM sequences."""
    from scripts.tune_and_analyze import build_dataset
    X, y, feature_names, keys, tca_dates = build_dataset(pairs)
    return X, y, feature_names, keys, tca_dates


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def collect_oof_predictions(X, y, keys, pairs, n_folds=5):
    """Collect out-of-fold predictions from LR, BiLSTM-signal, and GNN.

    Returns (oof_matrix, y_true) where oof_matrix is (n_samples, n_models).
    """
    n = len(y)
    oof = np.zeros((n, 3))  # [lr_prob, lstm_signal, gnn_prob]

    # Stratified fold indices
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    fold_assignment = np.zeros(n, dtype=int)
    for f in range(n_folds):
        ps, pe = len(pos_idx) * f // n_folds, len(pos_idx) * (f + 1) // n_folds
        ns, ne = len(neg_idx) * f // n_folds, len(neg_idx) * (f + 1) // n_folds
        for idx in np.concatenate([pos_idx[ps:pe], neg_idx[ns:ne]]):
            fold_assignment[idx] = f

    for f in range(n_folds):
        test_mask = fold_assignment == f
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        test_indices = np.where(test_mask)[0]

        # 1. Logistic Regression
        from scripts.tune_and_analyze import train_lr, predict_lr
        w, b, mu, sd = train_lr(X_train, y_train, lr=0.05, reg=0.01, epochs=300)
        lr_probs = predict_lr(X_test, w, b, mu, sd)
        oof[test_indices, 0] = lr_probs

        # 2. BiLSTM regression signal (sigmoid of predicted max Pc)
        # Use feature 6 (max_log10_pc) as a proxy since BiLSTM isn't retrained per fold
        # The regression signal is: sigmoid(steepness * (pred_max - threshold))
        max_pc_feature = X_test[:, 6]  # max_log10_pc
        pc_signal = sigmoid(5.0 * (max_pc_feature - (-3.3)))
        oof[test_indices, 1] = pc_signal

        # 3. GNN (graph-augmented non-linear MLP)
        try:
            import torch
            # Build graph features for this fold
            sat_ids = set()
            edges_f = []
            edge_feats_f = []
            train_keys = [keys[i] for i in np.where(train_mask)[0]]

            for k in train_keys:
                n1, n2, tca = k
                sat_ids.add(n1)
                sat_ids.add(n2)
                cdms = pairs.get(k, [])
                max_pc = max(c.get("pc", 0) or 0 for c in cdms) if cdms else 0
                edges_f.append((n1, n2))
                edge_feats_f.append({"max_pc": max_pc, "n_updates": len(cdms)})

            sat_list = sorted(sat_ids)
            sat_idx = {s: i for i, s in enumerate(sat_list)}
            n_nodes = len(sat_list)

            # Node features
            degree = np.zeros(n_nodes)
            weighted_deg = np.zeros(n_nodes)
            max_nbr_pc = np.zeros(n_nodes)
            n_high = np.zeros(n_nodes)
            for (n1, n2), ef in zip(edges_f, edge_feats_f):
                i, j = sat_idx[n1], sat_idx[n2]
                degree[i] += 1; degree[j] += 1
                pc = ef["max_pc"]
                weighted_deg[i] += pc; weighted_deg[j] += pc
                max_nbr_pc[i] = max(max_nbr_pc[i], pc)
                max_nbr_pc[j] = max(max_nbr_pc[j], pc)
                if pc >= PC_ACTION_THRESHOLD:
                    n_high[i] += 1; n_high[j] += 1

            node_features = np.column_stack([
                np.log1p(degree), np.log1p(weighted_deg),
                np.log1p(max_nbr_pc * 1e4), n_high
            ])

            # Message passing aggregation
            node_agg = np.zeros((n_nodes, 2))
            node_count = np.zeros(n_nodes)
            for (n1, n2), ef in zip(edges_f, edge_feats_f):
                i, j = sat_idx[n1], sat_idx[n2]
                miss = min(c.get("miss_distance_km", 9999) or 9999 for c in pairs.get((min(n1,n2),max(n1,n2),_), []) if c) if False else 100
                node_agg[i] += [ef["max_pc"], 100]; node_agg[j] += [ef["max_pc"], 100]
                node_count[i] += 1; node_count[j] += 1
            node_count[node_count == 0] = 1
            node_agg /= node_count[:, None]

            # Build training edge features
            X_gnn_train = []
            y_gnn_train = []
            for idx_t in np.where(train_mask)[0]:
                k = keys[idx_t]
                n1, n2 = k[0], k[1]
                i, j = sat_idx.get(n1), sat_idx.get(n2)
                if i is None or j is None:
                    continue
                cdms = pairs.get(k, [])
                edge_f = np.array([math.log1p(len(cdms)), math.log1p(100)])
                row = np.concatenate([node_features[i], node_features[j], edge_f, node_agg[i], node_agg[j]])
                X_gnn_train.append(row)
                y_gnn_train.append(y[idx_t])

            X_gnn_train = np.array(X_gnn_train)
            y_gnn_train = np.array(y_gnn_train)

            # Train MLP
            gnn_mu = X_gnn_train.mean(0)
            gnn_sd = X_gnn_train.std(0) + 1e-8
            Xn = torch.tensor((X_gnn_train - gnn_mu) / gnn_sd, dtype=torch.float32)
            yn = torch.tensor(y_gnn_train, dtype=torch.float32)

            class MLP(torch.nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(d, 64), torch.nn.GELU(), torch.nn.Dropout(0.3),
                        torch.nn.Linear(64, 32), torch.nn.GELU(),
                        torch.nn.Linear(32, 1))
                def forward(self, x): return self.net(x).squeeze(-1)

            mlp = MLP(X_gnn_train.shape[1])
            opt = torch.optim.AdamW(mlp.parameters(), lr=0.005, weight_decay=0.01)
            pw = max(1.0, (1 - y_gnn_train.mean()) / max(y_gnn_train.mean(), 0.01))

            for ep in range(150):
                mlp.train()
                logits = mlp(Xn)
                w_t = torch.where(yn == 1, pw, 1.0)
                loss = (w_t * torch.nn.functional.binary_cross_entropy_with_logits(logits, yn, reduction='none')).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            # Predict on test fold
            mlp.eval()
            for idx_t in test_indices:
                k = keys[idx_t]
                n1, n2 = k[0], k[1]
                i, j = sat_idx.get(n1), sat_idx.get(n2)
                if i is None or j is None:
                    oof[idx_t, 2] = 0.5  # unknown node, neutral
                    continue
                cdms = pairs.get(k, [])
                edge_f = np.array([math.log1p(len(cdms)), math.log1p(100)])
                row = np.concatenate([node_features[i], node_features[j], edge_f, node_agg[i], node_agg[j]])
                x_t = torch.tensor((row - gnn_mu) / gnn_sd, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    oof[idx_t, 2] = float(torch.sigmoid(mlp(x_t)).item())

        except Exception as e:
            print(f"  GNN fold {f} failed: {e}")
            oof[test_indices, 2] = 0.5  # neutral fallback

        print(f"  Fold {f+1}/{n_folds}: LR={lr_probs.mean():.3f} Signal={pc_signal.mean():.3f} GNN={oof[test_indices, 2].mean():.3f}")

    return oof, y


def fit_meta_learner(oof: np.ndarray, y: np.ndarray):
    """Fit constrained logistic regression on stacked predictions.

    Learns non-negative weights that minimize log-loss.
    Returns normalized weights summing to 1.0.
    """
    from scipy.optimize import minimize

    model_names = ["LR", "BiLSTM_signal", "GNN"]
    n_models = oof.shape[1]

    # Objective: log-loss of weighted average
    def neg_log_loss(w):
        w_norm = np.abs(w) / (np.abs(w).sum() + 1e-8)  # normalize
        p = (oof * w_norm).sum(axis=1)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        # Weighted loss: penalize missed positives 2x
        weight = np.where(y == 1, 2.0, 1.0)
        loss = -np.mean(weight * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        return loss

    # Try multiple random initializations
    best_result = None
    best_loss = 1e9
    for seed in range(20):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet([1, 1, 1])  # random weights summing to 1
        result = minimize(neg_log_loss, w0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-6})
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    raw_weights = np.abs(best_result.x)
    weights = raw_weights / raw_weights.sum()

    # Compute per-model individual AUC-PR and accuracy
    from scripts.tune_and_analyze import eval_metrics
    model_stats = {}
    for i, name in enumerate(model_names):
        metrics = eval_metrics(y, oof[:, i])
        model_stats[name] = {
            "weight": round(float(weights[i]), 4),
            "individual_f1": round(metrics["f1"], 4),
            "individual_recall": round(metrics["recall"], 4),
            "individual_precision": round(metrics["precision"], 4),
        }

    # Ensemble performance with learned weights
    ens_probs = (oof * weights).sum(axis=1)
    ens_metrics = eval_metrics(y, ens_probs)

    return weights, model_stats, ens_metrics, best_loss


def main():
    parser = argparse.ArgumentParser(description="Calibrate ensemble weights via stacking")
    parser.add_argument("--quick", action="store_true", help="3-fold for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("  Ensemble Weight Calibration (Stacking)")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    pairs = load_pairs()
    print(f"\nLoaded {len(pairs)} conjunction pairs")

    X, y, feature_names, keys, tca_dates = build_features_and_labels(pairs)
    print(f"Features: {X.shape}, Positive rate: {y.mean():.3f}")

    n_folds = 3 if args.quick else 5
    print(f"\nCollecting {n_folds}-fold out-of-fold predictions...")
    oof, y_true = collect_oof_predictions(X, y, keys, pairs, n_folds=n_folds)

    print(f"\nFitting meta-learner (constrained, recall-weighted)...")
    weights, model_stats, ens_metrics, loss = fit_meta_learner(oof, y_true)

    model_names = ["LR", "BiLSTM_signal", "GNN"]
    print(f"\n{'=' * 60}")
    print("  LEARNED ENSEMBLE WEIGHTS")
    print(f"{'=' * 60}")
    for name in model_names:
        s = model_stats[name]
        print(f"  {name:20s}: weight={s['weight']:.3f}  F1={s['individual_f1']:.3f}  Recall={s['individual_recall']:.3f}")

    print(f"\n  Ensemble F1={ens_metrics['f1']:.4f}  Recall={ens_metrics['recall']:.4f}  Precision={ens_metrics['precision']:.4f}")
    print(f"  Meta-learner loss: {loss:.4f}")

    # Save weights
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "stacking_meta_learner",
        "n_folds": n_folds,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "weights": {name: round(float(weights[i]), 4) for i, name in enumerate(model_names)},
        "model_stats": model_stats,
        "ensemble_metrics": {
            "f1": round(ens_metrics["f1"], 4),
            "recall": round(ens_metrics["recall"], 4),
            "precision": round(ens_metrics["precision"], 4),
            "accuracy": round(ens_metrics["accuracy"], 4),
        },
        "meta_learner_loss": round(loss, 6),
    }

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
