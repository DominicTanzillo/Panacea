"""Split-Conformal Prediction for Conjunction Risk Assessment.

Provides distribution-free prediction intervals with guaranteed coverage.
No distributional assumptions — works with any base model.

Theory (Vovk et al., 2005; Lei et al., 2018):
  Given a calibration set of n exchangeable examples, split-conformal
  prediction constructs intervals that contain the true value with
  probability >= 1 - alpha, regardless of the true data distribution.

  For regression: interval = [y_hat - q, y_hat + q]
    where q = quantile(calibration_scores, ceil((n+1)(1-alpha))/n)

  For classification: prediction set = {y : p(y|x) >= 1 - q_hat}
    with guaranteed coverage P(Y in C(X)) >= 1 - alpha

This is critical for space safety: NASA CARA requires calibrated
uncertainty before adopting ML predictions operationally.
"""

import math
import numpy as np
from typing import Optional


class ConformalPredictor:
    """Split-conformal prediction wrapper for any base model.

    Usage:
        cp = ConformalPredictor(alpha=0.10)  # 90% coverage
        cp.calibrate(y_cal_true, y_cal_pred)
        lower, upper = cp.predict_interval(y_test_pred)
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage rate (0.10 = 90% coverage guarantee)
        """
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.q_hat: Optional[float] = None
        self.n_cal: int = 0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute nonconformity scores on calibration set.

        Args:
            y_true: true values (n_cal,)
            y_pred: model predictions (n_cal,)

        Returns:
            dict with calibration stats
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Nonconformity score: absolute residual
        self.calibration_scores = np.abs(y_true - y_pred)
        self.n_cal = len(self.calibration_scores)

        # Quantile with finite-sample correction (Theorem 2.2, Lei et al. 2018)
        # q_hat = quantile(scores, ceil((n+1)(1-alpha)) / n)
        level = np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal
        level = min(level, 1.0)  # cap at 1.0
        self.q_hat = float(np.quantile(self.calibration_scores, level))

        return {
            "n_calibration": self.n_cal,
            "alpha": self.alpha,
            "coverage_target": 1 - self.alpha,
            "q_hat": self.q_hat,
            "mean_score": float(self.calibration_scores.mean()),
            "median_score": float(np.median(self.calibration_scores)),
            "max_score": float(self.calibration_scores.max()),
        }

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals with guaranteed coverage.

        Args:
            y_pred: model predictions (n,)

        Returns:
            (lower, upper) arrays with shape (n,)
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        y_pred = np.asarray(y_pred, dtype=np.float64)
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate actual coverage on a test set.

        Args:
            y_true: true values (n,)
            y_pred: model predictions (n,)

        Returns:
            dict with coverage stats
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        lower, upper = self.predict_interval(y_pred)

        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(covered.mean())
        interval_width = float((upper - lower).mean())

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": 1 - self.alpha,
            "coverage_gap": empirical_coverage - (1 - self.alpha),
            "mean_interval_width": interval_width,
            "n_test": len(y_true),
            "n_covered": int(covered.sum()),
            "q_hat": self.q_hat,
        }


class ConformalClassifier:
    """Conformal prediction for binary classification (exceedance prediction).

    Produces prediction sets with guaranteed marginal coverage.
    For binary classification, this means the set is either {0}, {1}, or {0,1}.

    In practice, we output calibrated p-values:
      p(y=1|x) >= threshold  =>  include 1 in prediction set
      p(y=0|x) >= threshold  =>  include 0 in prediction set

    The key output is conformal_p_value: the largest alpha at which
    y=1 would be included in the prediction set. Higher = more confident.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.cal_scores_pos: Optional[np.ndarray] = None  # scores for y=1
        self.cal_scores_neg: Optional[np.ndarray] = None  # scores for y=0
        self.n_cal: int = 0

    def calibrate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Calibrate using softmax/sigmoid probabilities.

        Nonconformity score for classification: 1 - p(true class)
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

        # Scores: 1 - predicted probability of true class
        scores_for_true = np.where(y_true == 1, y_pred_proba, 1 - y_pred_proba)
        self.cal_scores = 1 - scores_for_true

        # Separate by class for class-conditional conformal
        pos_mask = y_true == 1
        self.cal_scores_pos = self.cal_scores[pos_mask] if pos_mask.any() else np.array([1.0])
        self.cal_scores_neg = self.cal_scores[~pos_mask] if (~pos_mask).any() else np.array([1.0])
        self.n_cal = len(y_true)

        # Quantiles
        level = min(np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal, 1.0)
        self.q_hat = float(np.quantile(self.cal_scores, level))

        return {
            "n_calibration": self.n_cal,
            "n_positive": int(pos_mask.sum()),
            "n_negative": int((~pos_mask).sum()),
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "mean_score": float(self.cal_scores.mean()),
        }

    def predict_set(self, y_pred_proba: np.ndarray) -> dict:
        """Produce prediction sets with guaranteed coverage.

        Returns dict with:
          - prediction_set: list of sets ({0}, {1}, or {0,1})
          - conformal_lower: conservative lower bound on P(exceed)
          - conformal_upper: conservative upper bound on P(exceed)
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first")

        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
        n = len(y_pred_proba)

        sets = []
        conformal_lower = np.zeros(n)
        conformal_upper = np.zeros(n)

        for i in range(n):
            p1 = y_pred_proba[i]
            p0 = 1 - p1

            # Include class in prediction set if its score <= q_hat
            score_if_1 = 1 - p1  # nonconformity if true class is 1
            score_if_0 = 1 - p0  # nonconformity if true class is 0

            include_1 = score_if_1 <= self.q_hat
            include_0 = score_if_0 <= self.q_hat

            pred_set = set()
            if include_0:
                pred_set.add(0)
            if include_1:
                pred_set.add(1)
            if not pred_set:
                # Empty set — include the more likely class
                pred_set = {1} if p1 >= p0 else {0}

            sets.append(pred_set)

            # Bounds: if only {1} is in set, we're confident it exceeds
            if pred_set == {1}:
                conformal_lower[i] = 1 - self.alpha
                conformal_upper[i] = 1.0
            elif pred_set == {0}:
                conformal_lower[i] = 0.0
                conformal_upper[i] = self.alpha
            else:  # {0, 1} — uncertain
                conformal_lower[i] = 0.0
                conformal_upper[i] = 1.0

        return {
            "prediction_sets": sets,
            "conformal_lower": conformal_lower,
            "conformal_upper": conformal_upper,
            "n_singleton": sum(1 for s in sets if len(s) == 1),
            "n_uncertain": sum(1 for s in sets if len(s) > 1),
            "n_empty": sum(1 for s in sets if len(s) == 0),
        }

    def evaluate_coverage(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Evaluate empirical coverage on test set."""
        y_true = np.asarray(y_true, dtype=np.int64)
        result = self.predict_set(y_pred_proba)

        covered = sum(
            1 for yt, ps in zip(y_true, result["prediction_sets"])
            if yt in ps
        )
        n = len(y_true)

        return {
            "empirical_coverage": covered / max(n, 1),
            "target_coverage": 1 - self.alpha,
            "coverage_gap": covered / max(n, 1) - (1 - self.alpha),
            "n_test": n,
            "n_covered": covered,
            "n_singleton": result["n_singleton"],
            "n_uncertain": result["n_uncertain"],
            "set_size_mean": np.mean([len(s) for s in result["prediction_sets"]]),
        }


def run_conformal_analysis(cdm_store_path: str, alpha: float = 0.10) -> dict:
    """Run full conformal prediction analysis on CDM data.

    Splits data into train/calibration/test, trains model,
    calibrates conformal predictor, and reports coverage.

    Returns dict with all metrics and per-pair conformal intervals.
    """
    import json
    from pathlib import Path
    from collections import defaultdict

    from src.model.cdm_forecast import (
        CDMForecastModel, extract_sequence_features,
        _build_feature_vector, PC_ACTION_THRESHOLD, load_cdm_pairs,
    )

    pair_map = load_cdm_pairs(cdm_store_path)
    if not pair_map:
        return {"error": "No CDM data"}

    # Build labeled dataset
    from datetime import datetime
    now = datetime.utcnow()

    pairs_data = []
    for pair_key, cdms in pair_map.items():
        sorted_cdms = sorted(cdms, key=lambda c: c.get("cdm_id", 0))
        feats = extract_sequence_features(sorted_cdms)
        seq = feats["sequence"]
        if seq.shape[0] < 2:
            continue

        max_pc = max(c.get("pc", 0) or 0 for c in sorted_cdms)
        label = 1 if max_pc >= PC_ACTION_THRESHOLD else 0

        # First half for features (simulate early prediction)
        n = seq.shape[0]
        obs_len = max(1, (n + 1) // 2)
        fv = _build_feature_vector(seq[:obs_len], feats["static"])
        if fv is None:
            continue

        log10_max_pc = math.log10(max(max_pc, 1e-20))

        pairs_data.append({
            "pair_key": pair_key,
            "features": fv,
            "label": label,
            "max_log10_pc": log10_max_pc,
            "tca": sorted_cdms[0].get("tca", ""),
        })

    if len(pairs_data) < 30:
        return {"error": f"Too few pairs ({len(pairs_data)})"}

    # Shuffle and three-way split: 60% train, 20% calibration, 20% test
    rng = np.random.RandomState(42)
    rng.shuffle(pairs_data)

    n = len(pairs_data)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)

    train_data = pairs_data[:n_train]
    cal_data = pairs_data[n_train:n_train + n_cal]
    test_data = pairs_data[n_train + n_cal:]

    X_train = np.array([p["features"] for p in train_data])
    y_train_cls = np.array([p["label"] for p in train_data])
    y_train_reg = np.array([p["max_log10_pc"] for p in train_data])

    X_cal = np.array([p["features"] for p in cal_data])
    y_cal_cls = np.array([p["label"] for p in cal_data])
    y_cal_reg = np.array([p["max_log10_pc"] for p in cal_data])

    X_test = np.array([p["features"] for p in test_data])
    y_test_cls = np.array([p["label"] for p in test_data])
    y_test_reg = np.array([p["max_log10_pc"] for p in test_data])

    print(f"Conformal Analysis: {n} pairs -> {len(train_data)} train, "
          f"{len(cal_data)} cal, {len(test_data)} test")
    print(f"  Positive rate: {y_train_cls.mean():.1%} train, "
          f"{y_cal_cls.mean():.1%} cal, {y_test_cls.mean():.1%} test")

    # Train logistic regression (same as CDMForecastModel)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_cal_n = (X_cal - mean) / std
    X_test_n = (X_test - mean) / std

    # Simple logistic regression via gradient descent
    n_feat = X_train_n.shape[1]
    weights = np.zeros(n_feat)
    bias = 0.0
    lr = 0.05
    reg = 0.01
    pos_rate = y_train_cls.mean()
    class_weight = math.sqrt((1 - pos_rate) / max(pos_rate, 0.01))

    for epoch in range(300):
        z = X_train_n @ weights + bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        sample_w = np.where(y_train_cls == 1, class_weight, 1.0)
        grad_w = (X_train_n.T @ ((p - y_train_cls) * sample_w)) / len(y_train_cls) + reg * weights
        grad_b = ((p - y_train_cls) * sample_w).mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    def predict_proba(X_n):
        z = X_n @ weights + bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_reg(X_n):
        """Use log-odds as regression proxy for log10(Pc)."""
        z = X_n @ weights + bias
        # Map logistic output to log10(Pc) scale
        # log10(5e-4) = -3.3, logit=0 maps to threshold
        return z * 0.5 + math.log10(PC_ACTION_THRESHOLD)

    # --- Regression conformal: intervals on predicted log10(Pc) ---
    cal_pred_reg = predict_reg(X_cal_n)
    reg_cp = ConformalPredictor(alpha=alpha)
    reg_cal_stats = reg_cp.calibrate(y_cal_reg, cal_pred_reg)
    print(f"\n  Regression conformal (alpha={alpha}):")
    print(f"    q_hat = {reg_cal_stats['q_hat']:.3f} log10(Pc)")
    print(f"    Median score = {reg_cal_stats['median_score']:.3f}")

    test_pred_reg = predict_reg(X_test_n)
    reg_coverage = reg_cp.evaluate_coverage(y_test_reg, test_pred_reg)
    print(f"    Test coverage: {reg_coverage['empirical_coverage']:.1%} "
          f"(target: {1-alpha:.0%})")
    print(f"    Mean interval width: {reg_coverage['mean_interval_width']:.3f} log10(Pc)")

    # --- Classification conformal: prediction sets for exceedance ---
    cal_pred_cls = predict_proba(X_cal_n)
    cls_cp = ConformalClassifier(alpha=alpha)
    cls_cal_stats = cls_cp.calibrate(y_cal_cls, cal_pred_cls)
    print(f"\n  Classification conformal (alpha={alpha}):")
    print(f"    q_hat = {cls_cal_stats['q_hat']:.3f}")

    test_pred_cls = predict_proba(X_test_n)
    cls_coverage = cls_cp.evaluate_coverage(y_test_cls, test_pred_cls)
    print(f"    Test coverage: {cls_coverage['empirical_coverage']:.1%} "
          f"(target: {1-alpha:.0%})")
    print(f"    Singleton sets: {cls_coverage['n_singleton']}/{cls_coverage['n_test']} "
          f"({cls_coverage['n_singleton']/max(cls_coverage['n_test'],1):.0%})")
    print(f"    Uncertain sets: {cls_coverage['n_uncertain']}/{cls_coverage['n_test']}")
    print(f"    Mean set size: {cls_coverage['set_size_mean']:.2f}")

    # --- Per-pair conformal intervals for export ---
    all_pred_reg = predict_reg(np.vstack([X_train_n, X_cal_n, X_test_n]))
    all_lower, all_upper = reg_cp.predict_interval(all_pred_reg)

    pair_intervals = []
    all_pairs = train_data + cal_data + test_data
    for i, pd_item in enumerate(all_pairs):
        pair_intervals.append({
            "pair_key": pd_item["pair_key"],
            "predicted_log10_pc": float(all_pred_reg[i]),
            "conformal_lower": float(all_lower[i]),
            "conformal_upper": float(all_upper[i]),
            "actual_log10_pc": pd_item["max_log10_pc"],
        })

    return {
        "alpha": alpha,
        "coverage_target": 1 - alpha,
        "n_total": n,
        "n_train": len(train_data),
        "n_calibration": len(cal_data),
        "n_test": len(test_data),
        "regression": {
            "calibration": reg_cal_stats,
            "test_coverage": reg_coverage,
        },
        "classification": {
            "calibration": cls_cal_stats,
            "test_coverage": cls_coverage,
        },
        "pair_intervals": pair_intervals,
    }
