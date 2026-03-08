"""CDM LSTM Sequence Model with Kelvins → Space-Track Transfer Learning.

Architecture:
  Input (seq_len, 3) → LSTM(64, 2 layers, dropout=0.2) → last hidden
  → FC(64→32) → ReLU → Dropout(0.3) → FC(32→1) → Sigmoid

Shared features per timestep:
  - log10(Pc)
  - log10(miss_distance_km)
  - time_to_tca_hours

Phase 1: Pre-train on Kelvins (~160k CDM rows, ~4k events)
Phase 2: Fine-tune on Space-Track CDMs (freeze LSTM → unfreeze all)
"""

import json
import math
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.cdm_forecast import (
    PC_ACTION_THRESHOLD,
    load_cdm_pairs,
    compute_trend_features,
    extract_sequence_features,
)

MAX_SEQ_LEN = 30
LABEL_THRESHOLD_LOG10 = math.log10(PC_ACTION_THRESHOLD)  # log10(5e-4) ≈ -3.3
# Kelvins data has much lower Pc values (median max_risk ~ -9).
# Using -5.0 (~5% positive) for pre-training gives reasonable class balance
# and teaches the LSTM the general pattern of Pc escalation sequences.
KELVINS_PRETRAIN_THRESHOLD_LOG10 = -5.0


class _LSTMNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x, lengths):
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden) — take last layer
        out = self.head(h_n[-1])
        return torch.sigmoid(out).squeeze(-1)


class _CDMSeqDataset(Dataset):
    def __init__(self, sequences, labels, max_len=MAX_SEQ_LEN):
        self.labels = np.array(labels, dtype=np.float32)
        self.lengths = []
        self.padded = []
        for seq in sequences:
            L = min(len(seq), max_len)
            self.lengths.append(L)
            padded = np.zeros((max_len, 3), dtype=np.float32)
            padded[:L] = seq[:L]
            self.padded.append(padded)
        self.padded = np.array(self.padded)
        self.lengths = np.array(self.lengths, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.padded[idx]),
            torch.tensor(self.lengths[idx]),
            torch.tensor(self.labels[idx]),
        )


def _compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(float)
    y = labels.astype(float)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y),
    }


class CDMLSTMModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _LSTMNet().to(self.device)
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.trained = False
        self.metrics: dict = {}

    def _normalize_sequences(self, sequences: list[np.ndarray], fit=False):
        """Z-score normalize features across all timesteps."""
        all_feats = np.concatenate(sequences, axis=0)  # (total_steps, 3)
        if fit:
            self.norm_mean = all_feats.mean(axis=0).astype(np.float32)
            self.norm_std = (all_feats.std(axis=0) + 1e-8).astype(np.float32)
        return [((s - self.norm_mean) / self.norm_std).astype(np.float32) for s in sequences]

    def pretrain_kelvins(self, kelvins_csv_path: str) -> dict:
        """Pre-train on Kelvins dataset. Returns metrics dict."""
        print(f"Loading Kelvins data from {kelvins_csv_path}...")
        events = defaultdict(list)
        with open(kelvins_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row["event_id"]
                events[eid].append(row)

        # Build sequences
        sequences, labels = [], []
        for eid, rows in events.items():
            rows.sort(key=lambda r: float(r["time_to_tca"]), reverse=True)
            full_seq = []
            for r in rows:
                risk = float(r["risk"])  # already log10(Pc)
                miss_m = float(r["miss_distance"])
                miss_km = miss_m / 1000.0
                log_miss_km = math.log10(max(miss_km, 1e-6))
                ttca = float(r["time_to_tca"])  # already hours
                full_seq.append([risk, log_miss_km, ttca])

            full_seq = np.array(full_seq, dtype=np.float32)
            max_risk = full_seq[:, 0].max()
            label = 1 if max_risk > KELVINS_PRETRAIN_THRESHOLD_LOG10 else 0

            # Use first ceil(n/2) updates
            n = len(full_seq)
            obs_len = max(1, math.ceil(n / 2))
            sequences.append(full_seq[:obs_len])
            labels.append(label)

        labels = np.array(labels)
        print(f"Kelvins: {len(sequences)} events, {labels.sum():.0f} positive ({labels.mean()*100:.1f}%)")

        # Stratified 80/20 split
        rng = np.random.RandomState(42)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_pos_train = int(0.8 * len(pos_idx))
        n_neg_train = int(0.8 * len(neg_idx))
        train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
        test_idx = np.concatenate([pos_idx[n_pos_train:], neg_idx[n_neg_train:]])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = labels[train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        test_labels = labels[test_idx]

        # Normalize (fit on train)
        train_seqs = self._normalize_sequences(train_seqs, fit=True)
        test_seqs = self._normalize_sequences(test_seqs)

        # Class weight
        pos_rate = max(train_labels.mean(), 0.01)
        pos_weight = math.sqrt((1 - pos_rate) / pos_rate)
        print(f"Class weight: pos_weight={pos_weight:.3f}")

        train_ds = _CDMSeqDataset(train_seqs, train_labels)
        test_ds = _CDMSeqDataset(test_seqs, test_labels)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=256)

        # Train
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        weight_tensor = torch.tensor([1.0, pos_weight], device=self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(50):
            self.net.train()
            total_loss = 0
            n_batches = 0
            for x, lengths, y in train_dl:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                probs = self.net(x, lengths)
                w = torch.where(y == 1, weight_tensor[1], weight_tensor[0])
                loss = nn.functional.binary_cross_entropy(probs, y, weight=w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            # Validation loss
            self.net.eval()
            val_loss = 0
            val_n = 0
            with torch.no_grad():
                for x, lengths, y in test_dl:
                    x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                    probs = self.net(x, lengths)
                    w = torch.where(y == 1, weight_tensor[1], weight_tensor[0])
                    loss = nn.functional.binary_cross_entropy(probs, y, weight=w)
                    val_loss += loss.item() * len(y)
                    val_n += len(y)
            val_loss /= max(val_n, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d} | train_loss={total_loss/n_batches:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 5:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        self.net.load_state_dict(best_state)

        # Evaluate
        train_metrics = self._eval_dataloader(train_dl)
        test_metrics = self._eval_dataloader(test_dl)

        print(f"Kelvins train: acc={train_metrics['accuracy']:.3f} F1={train_metrics['f1']:.3f}")
        print(f"Kelvins test:  acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}")

        self.metrics["pretrain"] = {
            "train": train_metrics, "test": test_metrics,
            "n_events": len(sequences), "pos_rate": round(float(labels.mean()), 4),
        }
        return self.metrics["pretrain"]

    def _eval_dataloader(self, dl) -> dict:
        self.net.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, lengths, y in dl:
                x, lengths = x.to(self.device), lengths.to(self.device)
                probs = self.net(x, lengths)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        return _compute_metrics(probs, labels)

    def finetune_spacetrack(self, cdm_store_path: str, emergency_csv_path: str = None) -> dict:
        """Fine-tune on Space-Track CDMs. Returns metrics dict."""
        # Load CDMs from both sources
        all_cdms = {}  # cdm_id -> dict with normalized fields

        # Source 1: cdm_store.jsonl
        print(f"Loading Space-Track CDMs from {cdm_store_path}...")
        with open(cdm_store_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = str(c.get("cdm_id", ""))
                if not cid:
                    continue
                all_cdms[cid] = {
                    "cdm_id": cid,
                    "pc": c.get("pc", 0) or 0,
                    "miss_km": c.get("miss_distance_km", 0) or 0,
                    "tca": c.get("tca", ""),
                    "created": c.get("creation_date", ""),
                    "n1": c.get("sat1_norad", 0),
                    "n2": c.get("sat2_norad", 0),
                    "s1_name": c.get("sat1_name", ""),
                    "s2_name": c.get("sat2_name", ""),
                }

        # Source 2: emergency CSV
        if emergency_csv_path and Path(emergency_csv_path).exists():
            print(f"Loading emergency CDMs from {emergency_csv_path}...")
            with open(emergency_csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = str(row.get("CDM_ID", ""))
                    if not cid or cid in all_cdms:
                        continue
                    pc_str = row.get("PC", "")
                    pc = float(pc_str) if pc_str else 0
                    min_rng = float(row.get("MIN_RNG", 0) or 0)
                    all_cdms[cid] = {
                        "cdm_id": cid,
                        "pc": pc,
                        "miss_km": min_rng,
                        "tca": row.get("TCA", ""),
                        "created": row.get("CREATED", ""),
                        "n1": int(row.get("SAT_1_ID", 0) or 0),
                        "n2": int(row.get("SAT_2_ID", 0) or 0),
                        "s1_name": row.get("SAT_1_NAME", ""),
                        "s2_name": row.get("SAT_2_NAME", ""),
                    }

        print(f"Total unique CDMs: {len(all_cdms)}")

        # Group by pair (min/max NORAD), sort by CDM_ID
        pairs = defaultdict(list)
        for c in all_cdms.values():
            n1, n2 = c["n1"], c["n2"]
            key = (min(n1, n2), max(n1, n2))
            pairs[key].append(c)

        for key in pairs:
            pairs[key].sort(key=lambda c: c["cdm_id"])

        # Build sequences
        sequences, labels = [], []
        for key, cdms in pairs.items():
            full_seq = []
            for c in cdms:
                pc = c["pc"]
                if pc <= 0:
                    continue
                log_pc = math.log10(max(pc, 1e-20))
                log_miss = math.log10(max(c["miss_km"], 1e-6))

                # Compute time_to_tca_hours
                ttca = 72.0
                tca_str = c["tca"]
                created_str = c["created"]
                if tca_str and created_str:
                    try:
                        tca_dt = datetime.fromisoformat(tca_str.replace("Z", "+00:00").replace("+00:00", ""))
                        created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00").replace("+00:00", ""))
                        ttca = max(0, (tca_dt - created_dt).total_seconds() / 3600)
                    except (ValueError, AttributeError):
                        pass

                full_seq.append([log_pc, log_miss, ttca])

            if len(full_seq) < 1:
                continue

            full_seq = np.array(full_seq, dtype=np.float32)
            max_pc = 10 ** full_seq[:, 0].max()
            label = 1 if max_pc >= PC_ACTION_THRESHOLD else 0

            n = len(full_seq)
            obs_len = max(1, math.ceil(n / 2))
            sequences.append(full_seq[:obs_len])
            labels.append(label)

        labels = np.array(labels)
        print(f"Space-Track: {len(sequences)} pairs, {labels.sum():.0f} positive ({labels.mean()*100:.1f}%)")

        # 70/30 split, seed=42
        rng = np.random.RandomState(42)
        idx = np.arange(len(sequences))
        rng.shuffle(idx)
        n_train = int(0.7 * len(idx))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = labels[train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        test_labels = labels[test_idx]

        # Re-fit normalization on Space-Track data (different Pc distribution)
        train_seqs = self._normalize_sequences(train_seqs, fit=True)
        test_seqs = self._normalize_sequences(test_seqs)

        pos_rate = max(train_labels.mean(), 0.01)
        pos_weight = math.sqrt((1 - pos_rate) / pos_rate)

        train_ds = _CDMSeqDataset(train_seqs, train_labels)
        test_ds = _CDMSeqDataset(test_seqs, test_labels)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=256)

        weight_tensor = torch.tensor([1.0, pos_weight], device=self.device)

        # Phase 1: Freeze LSTM, fine-tune head only (lr=1e-4, 10 epochs)
        print("Fine-tune Phase 1: FC head only (LSTM frozen)...")
        for p in self.net.lstm.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=1e-4
        )

        for epoch in range(10):
            self.net.train()
            total_loss, n_b = 0, 0
            for x, lengths, y in train_dl:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                probs = self.net(x, lengths)
                w = torch.where(y == 1, weight_tensor[1], weight_tensor[0])
                loss = nn.functional.binary_cross_entropy(probs, y, weight=w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d} | loss={total_loss/n_b:.4f}")

        # Phase 2: Unfreeze all, lr=1e-5, 20 epochs
        print("Fine-tune Phase 2: All layers unfrozen...")
        for p in self.net.lstm.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-5)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(20):
            self.net.train()
            total_loss, n_b = 0, 0
            for x, lengths, y in train_dl:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                probs = self.net(x, lengths)
                w = torch.where(y == 1, weight_tensor[1], weight_tensor[0])
                loss = nn.functional.binary_cross_entropy(probs, y, weight=w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1

            # Val loss
            self.net.eval()
            val_loss, val_n = 0, 0
            with torch.no_grad():
                for x, lengths, y in test_dl:
                    x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                    probs = self.net(x, lengths)
                    w = torch.where(y == 1, weight_tensor[1], weight_tensor[0])
                    loss = nn.functional.binary_cross_entropy(probs, y, weight=w)
                    val_loss += loss.item() * len(y)
                    val_n += len(y)
            val_loss /= max(val_n, 1)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d} | loss={total_loss/n_b:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 5:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        train_metrics = self._eval_dataloader(train_dl)
        test_metrics = self._eval_dataloader(test_dl)

        print(f"Space-Track train: acc={train_metrics['accuracy']:.3f} F1={train_metrics['f1']:.3f}")
        print(f"Space-Track test:  acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}")

        self.trained = True
        self.metrics["finetune"] = {
            "train": train_metrics, "test": test_metrics,
            "n_pairs": len(sequences), "pos_rate": round(float(labels.mean()), 4),
        }
        return self.metrics["finetune"]

    def _predict_single(self, seq_features: np.ndarray) -> float:
        """Predict probability for a single normalized sequence."""
        self.net.eval()
        seq_norm = ((seq_features - self.norm_mean) / self.norm_std).astype(np.float32)
        L = min(len(seq_norm), MAX_SEQ_LEN)
        padded = np.zeros((MAX_SEQ_LEN, 3), dtype=np.float32)
        padded[:L] = seq_norm[:L]
        x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([L]).to(self.device)
        with torch.no_grad():
            prob = self.net(x, lengths).item()
        return prob

    def predict(self, cdm_sequence: list[dict]) -> dict:
        """Predict for a single pair. Returns CDMForecastModel-compatible dict."""
        feats = extract_sequence_features(cdm_sequence)
        seq = feats["sequence"]
        meta = feats["meta"]

        if seq.shape[0] == 0:
            return {
                "will_exceed_threshold": False,
                "exceedance_probability": 0.0,
                "forecast_pc": 0,
                "risk_direction": "unknown",
                "confidence": 0,
                "action_recommended": False,
                "time_series": [],
                "sat1_name": "", "sat2_name": "",
                "sat1_norad": 0, "sat2_norad": 0,
                "tca": "", "current_pc": 0, "current_miss_km": 0, "n_updates": 0,
            }

        trends = compute_trend_features(seq)

        # Build 3-feature sequence for LSTM
        lstm_seq = seq[:, :3].copy()  # log10_pc, log10_miss, time_to_tca

        exc_prob = self._predict_single(lstm_seq)

        current_pc = meta["latest_pc"]
        already_exceeded = current_pc >= PC_ACTION_THRESHOLD
        will_exceed = exc_prob >= 0.5 or already_exceeded

        action = False
        if already_exceeded and trends["risk_direction"] != "de-escalating":
            action = True
        elif will_exceed and trends["risk_direction"] == "escalating":
            action = True
        elif exc_prob >= 0.7:
            action = True

        n = seq.shape[0]
        length_conf = min(n / 10.0, 1.0)
        model_conf = abs(exc_prob - 0.5) * 2
        confidence = 0.4 * length_conf + 0.6 * model_conf

        time_series = []
        for i in range(seq.shape[0]):
            time_series.append({
                "update_idx": i + 1,
                "log10_pc": round(float(seq[i, 0]), 2),
                "pc": float(seq[i, 3]),
                "miss_distance_km": round(float(seq[i, 4]), 1),
                "time_to_tca_hours": round(float(seq[i, 2]), 1),
            })

        return {
            "will_exceed_threshold": will_exceed,
            "exceedance_probability": round(exc_prob, 4),
            "forecast_pc": trends["forecast_pc"],
            "risk_direction": trends["risk_direction"],
            "confidence": round(confidence, 2),
            "action_recommended": action,
            "time_series": time_series,
            "sat1_name": meta["sat1_name"],
            "sat2_name": meta["sat2_name"],
            "sat1_norad": meta["sat1_norad"],
            "sat2_norad": meta["sat2_norad"],
            "tca": meta["tca"],
            "current_pc": current_pc,
            "current_miss_km": meta["latest_miss_km"],
            "n_updates": n,
        }

    def predict_all_pairs(self, cdm_store_path: str) -> list[dict]:
        """Predict all pairs in CDM store."""
        pair_map = load_cdm_pairs(cdm_store_path)
        predictions = []
        for pair_key, pair_cdms in pair_map.items():
            pred = self.predict(pair_cdms)
            predictions.append(pred)
        predictions.sort(key=lambda p: p.get("exceedance_probability", 0), reverse=True)
        return predictions

    def save(self, path: str):
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "metrics": self.metrics,
            "trained": self.trained,
        }, str(p))

    @classmethod
    def load(cls, path: str) -> "CDMLSTMModel":
        """Load model checkpoint."""
        model = cls()
        checkpoint = torch.load(str(path), map_location=model.device, weights_only=False)
        model.net.load_state_dict(checkpoint["model_state_dict"])
        model.norm_mean = checkpoint["norm_mean"]
        model.norm_std = checkpoint["norm_std"]
        model.metrics = checkpoint.get("metrics", {})
        model.trained = checkpoint.get("trained", False)
        return model
