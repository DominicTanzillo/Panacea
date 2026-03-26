"""Panacea CLI — predict and train CDM escalation models.

Usage:
    panacea predict --cdm-store data/prediction_logs/cdm_store.jsonl [-o output.json]
    panacea train   --cdm-store data/prediction_logs/cdm_store.jsonl [-o models/cdm_forecast.json]
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_predict(args):
    """Run predictions on all pairs in a CDM store."""
    from src.model.cdm_forecast import CDMForecastModel

    cdm_store = Path(args.cdm_store)
    if not cdm_store.exists():
        print(f"Error: CDM store not found: {cdm_store}", file=sys.stderr)
        sys.exit(1)

    # Load model if specified, otherwise train on the fly
    if args.model and Path(args.model).exists():
        model = CDMForecastModel.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        model = CDMForecastModel()
        metrics = model.train(str(cdm_store))
        mode = metrics.get("mode", "heuristic")
        f1 = metrics.get("test", {}).get("f1", 0)
        print(f"Trained model ({mode}, test F1={f1:.3f})")

    predictions = model.predict_all_pairs(str(cdm_store))

    output = {
        "n_pairs": len(predictions),
        "n_actionable": sum(1 for p in predictions if p.get("action_recommended")),
        "n_escalating": sum(1 for p in predictions if p.get("risk_direction") == "escalating"),
        "predictions": predictions,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"Wrote {len(predictions)} predictions to {out_path}")
    else:
        json.dump(output, sys.stdout, indent=2, default=str)
        print()


def cmd_train(args):
    """Train a CDM forecast model and save it."""
    from src.model.cdm_forecast import CDMForecastModel

    cdm_store = Path(args.cdm_store)
    if not cdm_store.exists():
        print(f"Error: CDM store not found: {cdm_store}", file=sys.stderr)
        sys.exit(1)

    model = CDMForecastModel()
    metrics = model.train(str(cdm_store))

    mode = metrics.get("mode", "heuristic")
    n_pairs = metrics.get("n_labeled", 0)
    test = metrics.get("test", {})
    print(f"Trained {mode} model on {n_pairs} pairs")
    if test:
        print(f"  Test: F1={test.get('f1', 0):.3f}, Recall={test.get('recall', 0):.3f}, "
              f"Precision={test.get('precision', 0):.3f}, AUC-PR={test.get('auc_pr', 0):.3f}")

    out_path = Path(args.output) if args.output else Path("models/cdm_forecast.json")
    model.save(out_path)
    print(f"Model saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="panacea",
        description="Panacea SSA — ML-based satellite conjunction assessment",
    )
    parser.add_argument("--version", action="version", version="panacea-ssa 0.1.0")
    sub = parser.add_subparsers(dest="command")

    # predict
    p_pred = sub.add_parser("predict", help="Predict Pc escalation for CDM pairs")
    p_pred.add_argument("--cdm-store", required=True, help="Path to CDM store (.jsonl)")
    p_pred.add_argument("--model", "-m", help="Path to trained model (.json)")
    p_pred.add_argument("--output", "-o", help="Output file (default: stdout)")

    # train
    p_train = sub.add_parser("train", help="Train CDM forecast model")
    p_train.add_argument("--cdm-store", required=True, help="Path to CDM store (.jsonl)")
    p_train.add_argument("--output", "-o", help="Output model path (default: models/cdm_forecast.json)")

    args = parser.parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
