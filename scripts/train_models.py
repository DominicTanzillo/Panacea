"""Local GPU training + cross-validation for CDM forecasting models.

Trains the CDMSequenceModel (BiLSTM with transfer learning from Kelvins),
calibrates probabilities, and produces a results table comparing LR, LSTM,
and Ensemble.

Usage:
    python scripts/train_models.py                    # Full training
    python scripts/train_models.py --skip-pretrain    # Skip Kelvins (use existing ckpt)
    python scripts/train_models.py --cv-only          # Cross-validation only
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
KELVINS_CSV = ROOT / "data" / "cdm" / "Collision Avoidance Challenge - Dataset" / "kelvins_competition_data" / "train_data.csv"
CKPT_PATH = ROOT / "models" / "cdm_sequence_model.pt"


def load_spacetrack_pairs(cdm_store_path: str) -> dict:
    """Load and group Space-Track CDMs by conjunction pair + TCA date."""
    pairs = defaultdict(list)
    with open(cdm_store_path) as f:
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


def train_full(args):
    """Full training pipeline: pretrain on Kelvins, fine-tune on Space-Track."""
    import torch
    from src.model.cdm_sequence_model import CDMSequenceModel

    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}")

    if not CDM_STORE.exists():
        print(f"ERROR: CDM store not found at {CDM_STORE}")
        return

    model = CDMSequenceModel()

    # Phase 1: Pre-train on Kelvins
    if not args.skip_pretrain and KELVINS_CSV.exists():
        print("\n" + "=" * 60)
        pretrain_metrics = model.pretrain_kelvins(str(KELVINS_CSV))
        print("=" * 60)
    elif CKPT_PATH.exists() and args.skip_pretrain:
        print("Loading existing checkpoint (skip-pretrain mode)...")
        model = CDMSequenceModel.load(str(CKPT_PATH))
        if not model.trained:
            print("Checkpoint incompatible or untrained — will train from scratch")
            model = CDMSequenceModel()
            if KELVINS_CSV.exists():
                model.pretrain_kelvins(str(KELVINS_CSV))
    else:
        print("No Kelvins CSV — training from scratch on Space-Track only")

    # Phase 2: Fine-tune on Space-Track
    print("\n" + "=" * 60)
    ft_metrics = model.finetune_spacetrack(str(CDM_STORE))
    print("=" * 60)

    # Save checkpoint
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(CKPT_PATH))
    print(f"\nSaved checkpoint to {CKPT_PATH}")

    return model


def cross_validate(n_folds: int = 5):
    """5-fold stratified cross-validation comparing LR, LSTM, and Ensemble."""
    import torch
    from src.model.cdm_forecast import CDMForecastModel
    from src.model.cdm_sequence_model import CDMSequenceModel

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}\n")

    if not CDM_STORE.exists():
        print(f"ERROR: CDM store not found at {CDM_STORE}")
        return

    # Load all pairs and prepare data
    pair_map = load_spacetrack_pairs(str(CDM_STORE))
    THRESHOLD_LOG10 = math.log10(5e-4)  # -3.3

    pair_keys = sorted(pair_map.keys())
    labels = []
    for pk in pair_keys:
        cdms = pair_map[pk]
        pcs = [float(c.get("pc", 0) or 0) for c in cdms]
        max_pc = max(pcs) if pcs else 0
        labels.append(1 if max_pc > 0 and math.log10(max(max_pc, 1e-20)) > THRESHOLD_LOG10 else 0)

    labels = np.array(labels)
    n_pos = labels.sum()
    n_total = len(labels)
    print(f"Dataset: {n_total} pairs, {n_pos} positive ({100*n_pos/n_total:.1f}%)")

    # Stratified splits
    rng = np.random.RandomState(42)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, n_folds)
    neg_folds = np.array_split(neg_idx, n_folds)

    results = {"lr": [], "lstm": [], "ensemble": []}

    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        test_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_idx = np.concatenate([
            np.concatenate([pos_folds[i] for i in range(n_folds) if i != fold]),
            np.concatenate([neg_folds[i] for i in range(n_folds) if i != fold]),
        ])

        # Prepare train/test CDM stores (temporary)
        train_pairs = {pair_keys[i]: pair_map[pair_keys[i]] for i in train_idx}
        test_pairs = {pair_keys[i]: pair_map[pair_keys[i]] for i in test_idx}

        # Write temporary CDM stores
        tmp_train = ROOT / "models" / f"_cv_train_{fold}.jsonl"
        tmp_test = ROOT / "models" / f"_cv_test_{fold}.jsonl"
        for tmp_path, pairs in [(tmp_train, train_pairs), (tmp_test, test_pairs)]:
            with open(tmp_path, "w") as f:
                for cdms in pairs.values():
                    for c in cdms:
                        f.write(json.dumps(c) + "\n")

        # 1. Logistic Regression (CDMForecastModel)
        try:
            lr_model = CDMForecastModel()
            lr_metrics = lr_model.train(str(tmp_train))
            lr_test = lr_metrics.get("test", {})
            results["lr"].append({
                "auc_pr": lr_test.get("auc_pr", 0),
                "f1": lr_test.get("f1", 0),
                "recall": lr_test.get("recall", 0),
                "precision": lr_test.get("precision", 0),
                "accuracy": lr_test.get("accuracy", 0),
            })
            print(f"  LR: AUC-PR={lr_test.get('auc_pr', 0):.3f}, F1={lr_test.get('f1', 0):.3f}")
        except Exception as e:
            print(f"  LR failed: {e}")
            results["lr"].append({"auc_pr": 0, "f1": 0, "recall": 0, "precision": 0, "accuracy": 0})

        # 2. BiLSTM (CDMSequenceModel)
        try:
            lstm_model = CDMSequenceModel()
            if KELVINS_CSV.exists():
                lstm_model.pretrain_kelvins(str(KELVINS_CSV))
            lstm_ft = lstm_model.finetune_spacetrack(str(tmp_train))
            lstm_test = lstm_ft.get("test", {})
            results["lstm"].append({
                "auc_pr": 0,  # LSTM reports MAE, not AUC-PR directly
                "f1": lstm_test.get("esc_f1", 0),
                "recall": lstm_test.get("esc_recall", 0),
                "precision": lstm_test.get("esc_precision", 0),
                "accuracy": lstm_test.get("esc_accuracy", 0),
                "mae_log10_pc": lstm_test.get("mae_log10_pc", 0),
                "correlation": lstm_test.get("correlation_pc", 0),
                "ece": lstm_test.get("ece", 0),
                "temperature": lstm_test.get("temperature", 1.0),
            })
            print(f"  LSTM: F1={lstm_test.get('esc_f1', 0):.3f}, "
                  f"MAE={lstm_test.get('mae_log10_pc', 0):.3f}, "
                  f"ECE={lstm_test.get('ece', 0):.4f}")
        except Exception as e:
            print(f"  LSTM failed: {e}")
            results["lstm"].append({"auc_pr": 0, "f1": 0, "recall": 0, "precision": 0,
                                   "accuracy": 0, "mae_log10_pc": 0, "correlation": 0})

        # 3. Ensemble (LR + LSTM)
        # Compute ensemble on test pairs
        try:
            if lr_model and lstm_model and lstm_model.trained:
                n_correct, n_total_ens, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
                for pk_i in test_idx:
                    pk = pair_keys[pk_i]
                    cdms = pair_map[pk]
                    actual_label = labels[pk_i]

                    # LR prediction
                    lr_preds = lr_model.predict_all_pairs(str(tmp_test))
                    lr_prob = 0.5  # default
                    for lp in lr_preds:
                        n1, n2, tca = pk
                        if lp["sat1_norad"] == n1 and lp["sat2_norad"] == n2:
                            lr_prob = lp.get("exceedance_probability", 0.5)
                            break

                    # LSTM prediction
                    lstm_pred = lstm_model.predict(cdms)
                    lstm_prob = lstm_pred.get("escalation_probability", 0.5)
                    pred_max_pc = lstm_pred.get("predicted_max_log10_pc", -4.0)

                    # Ensemble: w1*LR + w2*LSTM + w3*sigmoid((pred_max + 3.3) * 5)
                    import torch
                    pc_signal = float(torch.sigmoid(torch.tensor((pred_max_pc + 3.3) * 5.0)).item())
                    p_ens = 0.4 * lr_prob + 0.3 * lstm_prob + 0.3 * pc_signal
                    pred_label = 1 if p_ens >= 0.5 else 0

                    n_total_ens += 1
                    if pred_label == actual_label:
                        n_correct += 1
                    if pred_label == 1 and actual_label == 1:
                        tp += 1
                    elif pred_label == 1 and actual_label == 0:
                        fp += 1
                    elif pred_label == 0 and actual_label == 1:
                        fn += 1
                    else:
                        tn += 1

                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-10)
                acc = n_correct / max(n_total_ens, 1)
                results["ensemble"].append({
                    "f1": f1, "recall": rec, "precision": prec, "accuracy": acc,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                })
                print(f"  Ensemble: F1={f1:.3f}, Acc={acc:.1%}")
        except Exception as e:
            print(f"  Ensemble failed: {e}")
            results["ensemble"].append({"f1": 0, "recall": 0, "precision": 0, "accuracy": 0})

        # Cleanup temp files
        tmp_train.unlink(missing_ok=True)
        tmp_test.unlink(missing_ok=True)

    # Print results table
    print(f"\n{'='*60}")
    print(f"  Cross-Validation Results ({n_folds}-fold)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 47)

    for model_name, key in [("LogReg", "lr"), ("BiLSTM", "lstm"), ("Ensemble", "ensemble")]:
        fold_results = results[key]
        if fold_results:
            f1s = [r["f1"] for r in fold_results]
            precs = [r["precision"] for r in fold_results]
            recs = [r["recall"] for r in fold_results]
            accs = [r["accuracy"] for r in fold_results]
            print(f"{model_name:<15} {np.mean(f1s):>7.3f}± {np.mean(precs):>6.3f}± "
                  f"{np.mean(recs):>6.3f}± {np.mean(accs):>6.1%}")

    # Save results
    results_path = ROOT / "models" / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "n_folds": n_folds,
            "n_pairs": n_total,
            "n_positive": int(n_pos),
            "positive_rate": float(n_pos / n_total),
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CDM models on local GPU")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip Kelvins pre-training (use existing checkpoint)")
    parser.add_argument("--cv-only", action="store_true",
                        help="Run cross-validation only (no full training)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    if args.cv_only:
        cross_validate(args.cv_folds)
    else:
        model = train_full(args)
        if model and model.trained:
            print("\nRunning cross-validation...")
            cross_validate(args.cv_folds)


if __name__ == "__main__":
    main()
