"""Autoregressive CDM Sequence Forecaster.

Given CDM updates 1...k, predicts CDM k+1's Pc, miss distance, and time-to-TCA.
Multi-step rollouts: feed predictions back as input to forecast 2, 3, ... steps ahead.

This is a harder, more impressive task than binary classification:
  - Predicts continuous trajectories, not just yes/no
  - Naturally produces uncertainty via prediction variance across rollout steps
  - Directly useful for operators: "what will the next CDM look like?"

Architecture: Reuses the bidirectional LSTM encoder from CDMSequenceModel,
but replaces the classification/regression heads with a next-step predictor.

Output per step:
  - predicted_log10_pc: next CDM's collision probability
  - predicted_log10_miss: next CDM's miss distance
  - predicted_time_to_tca: next CDM's time-to-TCA (hours)
  - uncertainty: prediction std from Monte Carlo dropout
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Same feature space as CDMSequenceModel
FEATURES = [
    "log10_pc", "log10_miss_km", "time_to_tca_hours",
    "delta_log10_pc", "delta_log10_miss", "delta_time_hours",
    "pc_rate", "miss_rate",
]
N_FEATURES = len(FEATURES)
MAX_SEQ_LEN = 30


class _AutoregressiveNet(nn.Module):
    """BiLSTM that predicts the next CDM update from a sequence of past updates."""

    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Predict next step's 3 core features: log10_pc, log10_miss, time_to_tca
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # [log10_pc, log10_miss, time_to_tca]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, seq_len, n_features) padded sequences
            lengths: (batch,) actual lengths

        Returns:
            predictions: (batch, 3) next-step [log10_pc, log10_miss, ttca]
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x.size(1)
        )

        # Use the last valid hidden state for each sequence
        batch_size = x.size(0)
        last_hidden = torch.zeros(batch_size, lstm_out.size(2), device=x.device)
        for i in range(batch_size):
            last_hidden[i] = lstm_out[i, lengths[i] - 1]

        return self.predictor(last_hidden)


class CDMAutoregressive:
    """Autoregressive CDM sequence forecaster with multi-step rollouts."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[_AutoregressiveNet] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None
        self.trained = False
        self.metrics: dict = {}

    def _init_net(self):
        self.net = _AutoregressiveNet(input_size=N_FEATURES).to(self.device)

    def train(self, cdm_store_path: str, epochs: int = 50, lr: float = 1e-3) -> dict:
        """Train on CDM sequences: predict next update from previous ones."""
        print("=== Training Autoregressive CDM Forecaster ===")

        pairs = self._load_pairs(cdm_store_path)
        sequences, targets = [], []

        for pair_key, cdms in pairs.items():
            seq = self._cdm_list_to_sequence(cdms)
            if seq is None or len(seq) < 3:
                continue

            # Create training examples: for each position k (2 <= k < n),
            # input = seq[0:k], target = seq[k] (core 3 features)
            for k in range(2, len(seq)):
                input_seq = seq[:k]
                target = seq[k, :3]  # log10_pc, log10_miss, time_to_tca
                sequences.append(input_seq)
                targets.append(target)

        if len(sequences) < 20:
            print(f"  Too few training examples ({len(sequences)})")
            return {"error": "insufficient data"}

        print(f"  Training examples: {len(sequences)} (from {len(pairs)} pairs)")

        # Split
        rng = np.random.RandomState(42)
        idx = np.arange(len(sequences))
        rng.shuffle(idx)
        n_train = int(0.8 * len(idx))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        train_seqs = [sequences[i] for i in train_idx]
        train_targets = np.array([targets[i] for i in train_idx], dtype=np.float32)
        test_seqs = [sequences[i] for i in test_idx]
        test_targets = np.array([targets[i] for i in test_idx], dtype=np.float32)

        # Normalize sequences
        all_feats = np.concatenate(train_seqs, axis=0)
        self.norm_mean = all_feats.mean(axis=0).astype(np.float32)
        self.norm_std = (all_feats.std(axis=0) + 1e-8).astype(np.float32)

        train_seqs = [((s - self.norm_mean) / self.norm_std).astype(np.float32) for s in train_seqs]
        test_seqs = [((s - self.norm_mean) / self.norm_std).astype(np.float32) for s in test_seqs]

        # Normalize targets
        self.target_mean = train_targets.mean(axis=0)
        self.target_std = train_targets.std(axis=0) + 1e-8

        train_targets_n = (train_targets - self.target_mean) / self.target_std
        test_targets_n = (test_targets - self.target_mean) / self.target_std

        # Pad sequences
        def pad_batch(seqs, max_len=MAX_SEQ_LEN):
            n_feat = seqs[0].shape[1]
            padded = np.zeros((len(seqs), max_len, n_feat), dtype=np.float32)
            lengths = np.zeros(len(seqs), dtype=np.int64)
            for i, s in enumerate(seqs):
                L = min(len(s), max_len)
                padded[i, :L] = s[:L]
                lengths[i] = L
            return torch.tensor(padded), torch.tensor(lengths)

        train_x, train_l = pad_batch(train_seqs)
        test_x, test_l = pad_batch(test_seqs)
        train_y = torch.tensor(train_targets_n, dtype=torch.float32)
        test_y = torch.tensor(test_targets_n, dtype=torch.float32)

        # Init and train
        self._init_net()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        patience = 0

        for epoch in range(epochs):
            self.net.train()
            # Mini-batch training
            perm = torch.randperm(len(train_x))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(train_x), 64):
                batch_idx = perm[start:start + 64]
                bx = train_x[batch_idx].to(self.device)
                bl = train_l[batch_idx].to(self.device)
                by = train_y[batch_idx].to(self.device)

                pred = self.net(bx, bl)
                loss = F.huber_loss(pred, by, delta=1.0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(test_x.to(self.device), test_l.to(self.device))
                val_loss = F.huber_loss(val_pred, test_y.to(self.device)).item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Denormalize for interpretable metrics
                pred_denorm = val_pred.cpu().numpy() * self.target_std + self.target_mean
                mae_pc = float(np.mean(np.abs(pred_denorm[:, 0] - test_targets[:, 0])))
                mae_miss = float(np.mean(np.abs(pred_denorm[:, 1] - test_targets[:, 1])))
                print(f"  Epoch {epoch+1:3d} | train_loss={total_loss/n_batches:.4f} "
                      f"val_loss={val_loss:.4f} | MAE_pc={mae_pc:.3f} MAE_miss={mae_miss:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        # Final metrics
        self.net.eval()
        with torch.no_grad():
            val_pred = self.net(test_x.to(self.device), test_l.to(self.device))
        pred_denorm = val_pred.cpu().numpy() * self.target_std + self.target_mean

        mae_pc = float(np.mean(np.abs(pred_denorm[:, 0] - test_targets[:, 0])))
        mae_miss = float(np.mean(np.abs(pred_denorm[:, 1] - test_targets[:, 1])))
        mae_ttca = float(np.mean(np.abs(pred_denorm[:, 2] - test_targets[:, 2])))

        corr_pc = float(np.corrcoef(pred_denorm[:, 0], test_targets[:, 0])[0, 1])

        print(f"\n  Autoregressive Results:")
        print(f"    MAE log10(Pc): {mae_pc:.3f}")
        print(f"    MAE log10(miss): {mae_miss:.3f}")
        print(f"    MAE time_to_tca: {mae_ttca:.1f} hours")
        print(f"    Correlation(Pc): {corr_pc:.3f}")

        self.trained = True
        self.metrics = {
            "mae_log10_pc": round(mae_pc, 4),
            "mae_log10_miss": round(mae_miss, 4),
            "mae_time_to_tca": round(mae_ttca, 2),
            "correlation_pc": round(corr_pc, 4),
            "n_train": len(train_seqs),
            "n_test": len(test_seqs),
            "n_pairs": len(pairs),
        }
        return self.metrics

    def forecast(self, cdm_sequence: list[dict], n_steps: int = 3,
                 mc_samples: int = 20) -> list[dict]:
        """Multi-step forecast with Monte Carlo dropout uncertainty.

        Args:
            cdm_sequence: list of CDM dicts
            n_steps: number of future steps to predict
            mc_samples: number of MC dropout forward passes for uncertainty

        Returns:
            list of dicts with predicted Pc, miss distance, and uncertainty
        """
        if not self.trained or self.net is None:
            return []

        seq = self._cdm_list_to_sequence(cdm_sequence)
        if seq is None or len(seq) < 2:
            return []

        # Normalize
        seq_norm = ((seq - self.norm_mean) / self.norm_std).astype(np.float32)
        forecasts = []

        current_seq = seq_norm.copy()
        current_raw = seq.copy()

        for step in range(n_steps):
            # MC Dropout: run multiple passes with dropout enabled
            self.net.train()  # enable dropout
            mc_preds = []

            L = min(len(current_seq), MAX_SEQ_LEN)
            padded = np.zeros((MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
            padded[:L] = current_seq[-L:]

            x = torch.tensor(padded).unsqueeze(0).to(self.device)
            lengths = torch.tensor([L]).to(self.device)

            for _ in range(mc_samples):
                with torch.no_grad():
                    pred = self.net(x, lengths)
                mc_preds.append(pred.cpu().numpy()[0])

            mc_preds = np.array(mc_preds)

            # Denormalize
            mc_denorm = mc_preds * self.target_std + self.target_mean
            mean_pred = mc_denorm.mean(axis=0)
            std_pred = mc_denorm.std(axis=0)

            predicted_log10_pc = float(mean_pred[0])
            predicted_log10_miss = float(mean_pred[1])
            predicted_ttca = float(max(mean_pred[2], 0))

            # Estimate time delta from last known step
            if len(current_raw) >= 2:
                last_ttca = current_raw[-1, 2]
                prev_ttca = current_raw[-2, 2]
                avg_delta = abs(prev_ttca - last_ttca)
            else:
                avg_delta = 6.0  # default 6 hours between CDMs

            forecasts.append({
                "step": step + 1,
                "predicted_log10_pc": round(predicted_log10_pc, 3),
                "predicted_pc": round(10 ** predicted_log10_pc, 8),
                "predicted_log10_miss_km": round(predicted_log10_miss, 3),
                "predicted_miss_km": round(10 ** predicted_log10_miss, 2),
                "predicted_time_to_tca_hours": round(predicted_ttca, 1),
                "uncertainty_log10_pc": round(float(std_pred[0]), 4),
                "uncertainty_log10_miss": round(float(std_pred[1]), 4),
                "hours_ahead": round(avg_delta * (step + 1), 1),
            })

            # Append prediction to sequence for next step (autoregressive)
            new_step_norm = (mean_pred - self.target_mean[:3]) / self.target_std[:3]
            # Build full 8-feature vector (derivatives approximated)
            prev_raw = current_raw[-1]
            delta_pc = predicted_log10_pc - prev_raw[0]
            delta_miss = predicted_log10_miss - prev_raw[1]
            delta_t = avg_delta
            pc_rate = delta_pc / max(delta_t, 0.01)
            miss_rate = delta_miss / max(delta_t, 0.01)

            new_raw = np.array([
                predicted_log10_pc, predicted_log10_miss, predicted_ttca,
                delta_pc, delta_miss, delta_t, pc_rate, miss_rate,
            ], dtype=np.float32)
            new_norm = ((new_raw - self.norm_mean) / self.norm_std).astype(np.float32)

            current_seq = np.vstack([current_seq, new_norm.reshape(1, -1)])
            current_raw = np.vstack([current_raw, new_raw.reshape(1, -1)])

        self.net.eval()  # restore eval mode
        return forecasts

    def _cdm_list_to_sequence(self, cdms: list[dict]) -> Optional[np.ndarray]:
        """Convert CDM list to feature array (same as CDMSequenceModel)."""
        sorted_cdms = sorted(cdms, key=lambda c: c.get("cdm_id", c.get("CDM_ID", 0)))

        seen = set()
        deduped = []
        for c in sorted_cdms:
            created = c.get("creation_date", c.get("CREATED", ""))
            if created and created in seen:
                continue
            if created:
                seen.add(created)
            deduped.append(c)

        tca_str = deduped[0].get("tca", deduped[0].get("TCA", "")) if deduped else ""
        try:
            tca_dt = datetime.fromisoformat(tca_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            tca_dt = None

        raw = []
        for c in deduped:
            pc = c.get("pc", c.get("PC", 0))
            if not pc or (isinstance(pc, str) and not pc.strip()):
                continue
            pc = float(pc)
            if pc <= 0:
                continue
            miss_km = c.get("miss_distance_km", c.get("MIN_RNG", 0))
            miss_km = float(miss_km) if miss_km else 0

            created = c.get("creation_date", c.get("CREATED", ""))
            ttca = 72.0
            if tca_dt and created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    ttca = max(0, (tca_dt - created_dt).total_seconds() / 3600)
                except (ValueError, AttributeError):
                    pass

            raw.append((math.log10(max(pc, 1e-20)), math.log10(max(miss_km, 0.001)), ttca))

        if not raw:
            return None

        features = []
        for i, (log_pc, log_miss, ttca) in enumerate(raw):
            if i == 0:
                delta_pc = delta_miss = delta_t = pc_rate = miss_rate = 0.0
            else:
                prev_pc, prev_miss, prev_ttca = raw[i - 1]
                delta_pc = log_pc - prev_pc
                delta_miss = log_miss - prev_miss
                delta_t = prev_ttca - ttca
                pc_rate = delta_pc / max(delta_t, 0.01) if delta_t > 0.01 else 0.0
                miss_rate = delta_miss / max(delta_t, 0.01) if delta_t > 0.01 else 0.0
            features.append([log_pc, log_miss, ttca, delta_pc, delta_miss, delta_t, pc_rate, miss_rate])

        return np.array(features, dtype=np.float32)

    def _load_pairs(self, path: str) -> dict:
        pairs = defaultdict(list)
        with open(path) as f:
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

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict() if self.net else None,
            "norm_mean": self.norm_mean, "norm_std": self.norm_std,
            "target_mean": self.target_mean, "target_std": self.target_std,
            "metrics": self.metrics, "trained": self.trained,
        }, str(path))

    @classmethod
    def load(cls, path: str) -> "CDMAutoregressive":
        model = cls()
        ckpt = torch.load(str(path), map_location=model.device, weights_only=False)
        model._init_net()
        if ckpt.get("model_state_dict"):
            model.net.load_state_dict(ckpt["model_state_dict"])
        model.norm_mean = ckpt["norm_mean"]
        model.norm_std = ckpt["norm_std"]
        model.target_mean = ckpt.get("target_mean")
        model.target_std = ckpt.get("target_std")
        model.metrics = ckpt.get("metrics", {})
        model.trained = ckpt.get("trained", False)
        return model
