"""CDM-Supervised Pairwise Risk Model (CSPR).

XGBoost-based model trained on Space-Track CDM Pc labels, predicting
collision risk from TLE-derived pairwise features. At inference time,
only free public TLE data is needed — no CDM subscription required.

Novel contribution:
  1. First TLE-only model supervised by CDM ground truth
  2. Democratizes conjunction screening (no ITAR, no CDM subscription)
  3. Online learning — improves daily as CDM data accumulates
  4. Addresses TLE precision limitation (paper Section 6.4)

Model file: models/pairwise_risk.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from src.data.pairwise_features import PAIRWISE_FEATURE_COLS


class CDMSupervisedRiskModel:
    """XGBoost model predicting log10(Pc) from pairwise TLE features.

    Trained on CDM Pc labels (ground truth), predicts from TLE features only.
    Supports both regression (log10 Pc) and binary risk classification.
    """

    def __init__(self):
        self.model = None            # xgb.XGBRegressor
        self.feature_cols: list[str] = list(PAIRWISE_FEATURE_COLS)
        self.normalization: dict = {}  # Feature stats for reporting
        self.training_meta: dict = {}  # CDM count, date range, version

    def train(
        self,
        X: pd.DataFrame,
        y_pc: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> dict:
        """Train on pairwise features with CDM Pc labels.

        Args:
            X: Feature DataFrame with PAIRWISE_FEATURE_COLS columns.
            y_pc: Raw Pc values (will be converted to log10).
            weights: Sample weights from cdm_pc_to_label().
            n_estimators: Number of boosting rounds.
            max_depth: Max tree depth.
            learning_rate: XGBoost learning rate.

        Returns:
            Dict with training metrics.
        """
        import xgboost as xgb

        # Target: log10(Pc), clipped to [-20, 0]
        y_log10 = np.clip(np.log10(np.maximum(y_pc, 1e-20)), -20.0, 0.0)

        # Ensure feature ordering
        X_train = X[self.feature_cols].copy()
        X_train = X_train.fillna(0.0)

        # Store normalization stats for reporting
        self.normalization = {
            "feature_means": X_train.mean().to_dict(),
            "feature_stds": X_train.std().to_dict(),
            "y_mean": float(np.mean(y_log10)),
            "y_std": float(np.std(y_log10)),
        }

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            eval_metric="rmse",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        fit_params = {}
        if weights is not None:
            fit_params["sample_weight"] = weights

        self.model.fit(X_train.values, y_log10, **fit_params)

        # Training predictions for metrics
        y_pred = self.model.predict(X_train.values)
        rmse = float(np.sqrt(np.mean((y_pred - y_log10) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_log10)))

        self.training_meta = {
            "n_samples": len(y_pc),
            "n_features": len(self.feature_cols),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "train_rmse_log10pc": rmse,
            "train_mae_log10pc": mae,
            "y_log10_range": [float(np.min(y_log10)), float(np.max(y_log10))],
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        return self.training_meta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log10(Pc) from pairwise TLE features.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of log10(Pc) predictions, clipped to [-20, 0].
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call train() or load() first")

        X_pred = X[self.feature_cols].fillna(0.0)
        y_log10 = self.model.predict(X_pred.values)
        return np.clip(y_log10, -20.0, 0.0)

    def predict_risk(self, X: pd.DataFrame, threshold_log10: float = -5.0) -> np.ndarray:
        """Predict binary risk probability (sigmoid-scaled log10 Pc).

        Maps log10(Pc) to [0, 1] using a sigmoid centered at threshold.
        Default threshold -5.0 matches the Kelvins risk definition (Pc > 1e-5).

        Args:
            X: Feature DataFrame.
            threshold_log10: Center of sigmoid mapping.

        Returns:
            Array of risk probabilities in [0, 1].
        """
        y_log10 = self.predict(X)
        # Sigmoid: higher Pc (less negative log10) -> higher risk
        # Scale factor 2.0 gives moderate steepness
        risk = 1.0 / (1.0 + np.exp(-2.0 * (y_log10 - threshold_log10)))
        return risk

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores (gain-based)."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, [float(v) for v in importances]))

    def save(self, path: Path):
        """Save model to JSON (XGBoost native format + metadata)."""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model in native JSON format
        xgb_path = path.with_suffix(".xgb.json")
        self.model.save_model(str(xgb_path))

        # Save metadata
        meta = {
            "model_type": "CDMSupervisedRiskModel",
            "version": 1,
            "feature_cols": self.feature_cols,
            "normalization": self.normalization,
            "training_meta": self.training_meta,
            "xgb_model_path": xgb_path.name,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  CSPR model saved to {path} + {xgb_path}")

    @classmethod
    def load(cls, path: Path) -> "CDMSupervisedRiskModel":
        """Load model from JSON."""
        import xgboost as xgb

        path = Path(path)
        with open(path) as f:
            meta = json.load(f)

        model = cls()
        model.feature_cols = meta.get("feature_cols", list(PAIRWISE_FEATURE_COLS))
        model.normalization = meta.get("normalization", {})
        model.training_meta = meta.get("training_meta", {})

        xgb_path = path.parent / meta["xgb_model_path"]
        model.model = xgb.XGBRegressor()
        model.model.load_model(str(xgb_path))

        return model

    def evaluate(
        self,
        X: pd.DataFrame,
        y_pc: np.ndarray,
        threshold_log10: float = -5.0,
    ) -> dict:
        """Evaluate model on test data.

        Returns regression metrics (RMSE, MAE on log10 Pc) and classification
        metrics (AUC-PR, F1 for binary risk at threshold).
        """
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            roc_auc_score,
        )

        y_log10_true = np.clip(np.log10(np.maximum(y_pc, 1e-20)), -20.0, 0.0)
        y_log10_pred = self.predict(X)
        y_risk_pred = self.predict_risk(X, threshold_log10)

        # Binary labels at threshold
        y_binary = (y_log10_true >= threshold_log10).astype(float)

        metrics = {
            "rmse_log10pc": float(np.sqrt(np.mean((y_log10_pred - y_log10_true) ** 2))),
            "mae_log10pc": float(np.mean(np.abs(y_log10_pred - y_log10_true))),
            "n_samples": len(y_pc),
            "n_positive": int(y_binary.sum()),
        }

        if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
            metrics["auc_pr"] = float(average_precision_score(y_binary, y_risk_pred))
            metrics["auc_roc"] = float(roc_auc_score(y_binary, y_risk_pred))
            # F1 at 0.5 threshold
            y_pred_binary = (y_risk_pred >= 0.5).astype(float)
            metrics["f1_at_05"] = float(f1_score(y_binary, y_pred_binary))
        else:
            metrics["auc_pr"] = 0.0
            metrics["auc_roc"] = 0.0
            metrics["f1_at_05"] = 0.0

        return metrics
