"""CDM Sequence-to-Risk Regression Model.

A genuinely useful deep learning model for conjunction risk forecasting.

WHAT'S DIFFERENT FROM THE OLD MODEL:
The old LSTM did binary classification ("will Pc exceed 5e-4?").
This model does THREE harder, more useful things:

1. REGRESSION: Predicts the continuous max log10(Pc) the pair will reach.
   Operators learn HOW BAD it'll get, not just yes/no.

2. MISS DISTANCE FORECASTING: Predicts future minimum miss distance.
   This is the most operationally relevant metric.

3. MULTI-HORIZON: Predicts at multiple future time windows.
   "What happens in 6h, 12h, 24h?"

Architecture:
  Input (seq_len, N_features) -> LSTM(128, 2 layers, dropout=0.3)
  -> Attention pooling over hidden states
  -> Multi-head output:
    Head 1: max log10(Pc) regression
    Head 2: min miss distance (km) regression
    Head 3: binary escalation classification (auxiliary)

Pre-trained on Kelvins (13K events, 103->7 features), fine-tuned on Space-Track.
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

# Engineered per-timestep features for Space-Track CDMs.
# We only get 16 raw fields (no covariance/velocity), so we engineer
# temporal dynamics that give the LSTM meaningful sequential patterns.
SPACETRACK_FEATURES = [
    "log10_pc",           # 0: current log10(Pc)
    "log10_miss_km",      # 1: current log10(miss distance in km)
    "time_to_tca_hours",  # 2: hours until TCA at this CDM update
    "delta_log10_pc",     # 3: change in log10(Pc) from previous update
    "delta_log10_miss",   # 4: change in log10(miss) from previous update
    "delta_time_hours",   # 5: hours elapsed since previous CDM update
    "pc_rate",            # 6: d(log10_pc)/d(hours) — Pc velocity
    "miss_rate",          # 7: d(log10_miss)/d(hours) — miss velocity
]
# Additional Kelvins features for pre-training (richer data)
KELVINS_EXTRA = ["relative_speed", "log10_miss_m", "max_risk_estimate", "miss_distance"]

MAX_SEQ_LEN = 30
N_SHARED_FEATURES = len(SPACETRACK_FEATURES)  # 8
N_KELVINS_FEATURES = N_SHARED_FEATURES + len(KELVINS_EXTRA)  # 12


class _AttentionPooling(nn.Module):
    """Attention over LSTM hidden states — learns to focus on critical CDM updates."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out, lengths):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        # Mask padding
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


class _MultiTaskNet(nn.Module):
    """Multi-task LSTM with attention and three output heads."""
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _AttentionPooling(hidden_size)

        # Shared trunk after attention
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Head 1: Predict max log10(Pc) — regression
        self.head_max_pc = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Head 2: Predict min miss distance (log10 km) — regression
        self.head_min_miss = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Head 3: Binary escalation classification — auxiliary
        self.head_escalation = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths):
        # Run LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                        total_length=x.size(1))

        # Attention pooling
        context, attn_weights = self.attention(lstm_out, lengths)

        # Shared features
        trunk_out = self.trunk(context)

        # Multi-task outputs
        max_pc = self.head_max_pc(trunk_out).squeeze(-1)       # regression
        min_miss = self.head_min_miss(trunk_out).squeeze(-1)   # regression
        escalation = torch.sigmoid(self.head_escalation(trunk_out).squeeze(-1))  # classification

        return max_pc, min_miss, escalation, attn_weights


class _SeqRegressionDataset(Dataset):
    """Dataset for multi-task sequence regression."""
    def __init__(self, sequences, targets, max_len=MAX_SEQ_LEN):
        """
        targets: list of dicts with keys:
          - max_log10_pc: float (regression target)
          - min_log10_miss: float (regression target)
          - escalated: int (0/1, auxiliary classification)
        """
        self.targets = targets
        self.lengths = []
        self.padded = []
        n_feat = sequences[0].shape[1] if len(sequences) > 0 else 3

        for seq in sequences:
            L = min(len(seq), max_len)
            self.lengths.append(L)
            padded = np.zeros((max_len, n_feat), dtype=np.float32)
            padded[:L] = seq[:L]
            self.padded.append(padded)

        self.padded = np.array(self.padded)
        self.lengths = np.array(self.lengths, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        t = self.targets[idx]
        return (
            torch.from_numpy(self.padded[idx]),
            torch.tensor(self.lengths[idx]),
            torch.tensor(t["max_log10_pc"], dtype=torch.float32),
            torch.tensor(t["min_log10_miss"], dtype=torch.float32),
            torch.tensor(t["escalated"], dtype=torch.float32),
        )


class CDMSequenceModel:
    """Multi-task CDM sequence model with transfer learning."""

    # Threshold for "escalation" classification head
    ESCALATION_THRESHOLD_LOG10 = math.log10(5e-4)  # -3.3

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[_MultiTaskNet] = None
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None
        self.trained = False
        self.metrics: dict = {}
        self.n_input_features = N_SHARED_FEATURES

    def _init_net(self, input_size):
        self.n_input_features = input_size
        self.net = _MultiTaskNet(input_size=input_size).to(self.device)

    def _normalize_sequences(self, sequences: list[np.ndarray], fit=False):
        all_feats = np.concatenate(sequences, axis=0)
        if fit:
            self.norm_mean = all_feats.mean(axis=0).astype(np.float32)
            self.norm_std = (all_feats.std(axis=0) + 1e-8).astype(np.float32)
        return [((s - self.norm_mean) / self.norm_std).astype(np.float32) for s in sequences]

    def pretrain_kelvins(self, kelvins_csv_path: str) -> dict:
        """Pre-train on Kelvins with rich features and regression targets."""
        print(f"=== Pre-training on Kelvins (regression) ===")
        print(f"Loading {kelvins_csv_path}...")

        events = defaultdict(list)
        with open(kelvins_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                events[row["event_id"]].append(row)

        sequences, targets = [], []
        for eid, rows in events.items():
            rows.sort(key=lambda r: float(r["time_to_tca"]), reverse=True)
            # First pass: extract raw values
            raw_vals = []
            for r in rows:
                risk = float(r["risk"])  # log10(Pc)
                miss_m = float(r["miss_distance"])
                miss_km = miss_m / 1000.0
                ttca = float(r["time_to_tca"])
                rel_speed = float(r.get("relative_speed", 0) or 0)
                log_miss_m = math.log10(max(miss_m, 1e-3))
                max_risk_est = float(r.get("max_risk_estimate", risk) or risk)
                log_miss_km = math.log10(max(miss_km, 1e-6))
                raw_vals.append((risk, log_miss_km, ttca, rel_speed, log_miss_m, max_risk_est, miss_km))

            # Second pass: compute temporal derivatives + build full feature vector
            full_seq = []
            for i, (risk, log_miss_km, ttca, rel_speed, log_miss_m, max_risk_est, miss_km) in enumerate(raw_vals):
                if i == 0:
                    delta_pc = 0.0
                    delta_miss = 0.0
                    delta_t = 0.0
                    pc_rate = 0.0
                    miss_rate = 0.0
                else:
                    prev = raw_vals[i - 1]
                    delta_pc = risk - prev[0]
                    delta_miss = log_miss_km - prev[1]
                    delta_t = prev[2] - ttca  # positive = time elapsed
                    if delta_t > 0.01:
                        pc_rate = delta_pc / delta_t
                        miss_rate = delta_miss / delta_t
                    else:
                        pc_rate = 0.0
                        miss_rate = 0.0

                # 12 features: 8 shared (matching Space-Track) + 4 Kelvins extras
                full_seq.append([
                    risk,                          # 0: log10(Pc)
                    log_miss_km,                   # 1: log10(miss_km)
                    ttca,                          # 2: time_to_tca_hours
                    delta_pc,                      # 3: delta log10(Pc)
                    delta_miss,                    # 4: delta log10(miss)
                    delta_t,                       # 5: hours since prev update
                    pc_rate,                       # 6: d(log10_pc)/d(hours)
                    miss_rate,                     # 7: d(log10_miss)/d(hours)
                    rel_speed / 1000.0,            # 8: relative speed (km/s)
                    log_miss_m,                    # 9: log10(miss_m)
                    max_risk_est,                  # 10: max risk estimate
                    miss_km,                       # 11: miss_km (raw)
                ])

            full_seq = np.array(full_seq, dtype=np.float32)

            # REGRESSION TARGETS from the FULL sequence
            max_log10_pc = float(full_seq[:, 0].max())
            min_miss_km = float(10 ** full_seq[:, 1].min())
            min_log10_miss = float(full_seq[:, 1].min())
            escalated = 1 if max_log10_pc > self.ESCALATION_THRESHOLD_LOG10 else 0

            # Input: first ceil(n/2) CDM updates (simulate early prediction)
            n = len(full_seq)
            obs_len = max(2, math.ceil(n / 2))
            sequences.append(full_seq[:obs_len])
            targets.append({
                "max_log10_pc": max_log10_pc,
                "min_log10_miss": min_log10_miss,
                "escalated": escalated,
            })

        print(f"Kelvins: {len(sequences)} events")
        print(f"  Escalated (>{self.ESCALATION_THRESHOLD_LOG10:.1f}): "
              f"{sum(t['escalated'] for t in targets)} ({100*sum(t['escalated'] for t in targets)/len(targets):.1f}%)")

        max_pcs = np.array([t["max_log10_pc"] for t in targets])
        print(f"  max_log10_pc range: [{max_pcs.min():.1f}, {max_pcs.max():.1f}], "
              f"mean={max_pcs.mean():.1f}, std={max_pcs.std():.1f}")

        # Split
        rng = np.random.RandomState(42)
        idx = np.arange(len(sequences))
        rng.shuffle(idx)
        n_train = int(0.8 * len(idx))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        train_seqs = [sequences[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        test_targets = [targets[i] for i in test_idx]

        # Normalize
        train_seqs = self._normalize_sequences(train_seqs, fit=True)
        test_seqs = self._normalize_sequences(test_seqs)

        # Normalize regression targets for stable training
        train_max_pcs = np.array([t["max_log10_pc"] for t in train_targets])
        train_min_miss = np.array([t["min_log10_miss"] for t in train_targets])
        self.target_mean = np.array([train_max_pcs.mean(), train_min_miss.mean()], dtype=np.float32)
        self.target_std = np.array([train_max_pcs.std() + 1e-8, train_min_miss.std() + 1e-8], dtype=np.float32)

        # Init network with Kelvins feature count
        self._init_net(input_size=N_KELVINS_FEATURES)

        train_ds = _SeqRegressionDataset(train_seqs, train_targets)
        test_ds = _SeqRegressionDataset(test_seqs, test_targets)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=256)

        # Train
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_val_loss = float("inf")
        best_state = None
        patience = 0

        for epoch in range(50):
            self.net.train()
            losses = {"total": 0, "pc": 0, "miss": 0, "esc": 0, "n": 0}

            for x, lengths, y_pc, y_miss, y_esc in train_dl:
                x = x.to(self.device)
                lengths = lengths.to(self.device)
                y_pc, y_miss, y_esc = y_pc.to(self.device), y_miss.to(self.device), y_esc.to(self.device)

                pred_pc, pred_miss, pred_esc, _ = self.net(x, lengths)

                # Normalize targets
                y_pc_norm = (y_pc - self.target_mean[0]) / self.target_std[0]
                y_miss_norm = (y_miss - self.target_mean[1]) / self.target_std[1]

                # Multi-task loss
                loss_pc = nn.functional.mse_loss(pred_pc, y_pc_norm)
                loss_miss = nn.functional.mse_loss(pred_miss, y_miss_norm)
                loss_esc = nn.functional.binary_cross_entropy(pred_esc, y_esc)

                # Weighted combination: regression is primary, classification is auxiliary
                loss = 1.0 * loss_pc + 0.5 * loss_miss + 0.3 * loss_esc

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                losses["total"] += loss.item()
                losses["pc"] += loss_pc.item()
                losses["miss"] += loss_miss.item()
                losses["esc"] += loss_esc.item()
                losses["n"] += 1

            scheduler.step()

            # Validation
            val_metrics = self._evaluate(test_dl)
            val_loss = val_metrics["mae_log10_pc"]

            if (epoch + 1) % 5 == 0 or epoch == 0:
                n = losses["n"]
                print(f"  Epoch {epoch+1:2d} | loss={losses['total']/n:.4f} "
                      f"(pc={losses['pc']/n:.4f} miss={losses['miss']/n:.4f} esc={losses['esc']/n:.4f}) "
                      f"| val_MAE_pc={val_metrics['mae_log10_pc']:.3f} "
                      f"val_MAE_miss={val_metrics['mae_log10_miss']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 7:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        train_metrics = self._evaluate(train_dl)
        test_metrics = self._evaluate(test_dl)

        print(f"\nKelvins Results:")
        print(f"  Train: MAE(log10_pc)={train_metrics['mae_log10_pc']:.3f}, "
              f"MAE(log10_miss)={train_metrics['mae_log10_miss']:.3f}, "
              f"Escalation F1={train_metrics['esc_f1']:.3f}")
        print(f"  Test:  MAE(log10_pc)={test_metrics['mae_log10_pc']:.3f}, "
              f"MAE(log10_miss)={test_metrics['mae_log10_miss']:.3f}, "
              f"Escalation F1={test_metrics['esc_f1']:.3f}")

        self.metrics["pretrain"] = {"train": train_metrics, "test": test_metrics,
                                     "n_events": len(sequences)}
        return self.metrics["pretrain"]

    def finetune_spacetrack(self, cdm_store_path: str) -> dict:
        """Fine-tune on Space-Track CDMs using only shared features."""
        print(f"\n=== Fine-tuning on Space-Track (regression) ===")

        pair_map = self._load_spacetrack_pairs(cdm_store_path)

        sequences, targets = [], []
        for pair_key, cdms in pair_map.items():
            full_seq = self._cdm_list_to_sequence(cdms)
            if full_seq is None or len(full_seq) < 2:
                continue

            # Regression targets from FULL sequence
            max_log10_pc = float(full_seq[:, 0].max())
            min_log10_miss = float(full_seq[:, 1].min())
            escalated = 1 if max_log10_pc > self.ESCALATION_THRESHOLD_LOG10 else 0

            # Input: first half
            n = len(full_seq)
            obs_len = max(2, math.ceil(n / 2))
            sequences.append(full_seq[:obs_len])
            targets.append({
                "max_log10_pc": max_log10_pc,
                "min_log10_miss": min_log10_miss,
                "escalated": escalated,
            })

        print(f"Space-Track: {len(sequences)} pairs")
        esc_count = sum(t["escalated"] for t in targets)
        print(f"  Escalated: {esc_count} ({100*esc_count/max(len(targets),1):.1f}%)")

        if len(sequences) < 10:
            print("  Too few pairs for fine-tuning")
            return {}

        # Split
        rng = np.random.RandomState(42)
        idx = np.arange(len(sequences))
        rng.shuffle(idx)
        n_train = int(0.7 * len(idx))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        train_seqs = [sequences[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        test_targets = [targets[i] for i in test_idx]

        # Re-normalize on Space-Track data
        train_seqs = self._normalize_sequences(train_seqs, fit=True)
        test_seqs = self._normalize_sequences(test_seqs)

        # Re-normalize targets
        train_pcs = np.array([t["max_log10_pc"] for t in train_targets])
        train_miss = np.array([t["min_log10_miss"] for t in train_targets])
        self.target_mean = np.array([train_pcs.mean(), train_miss.mean()], dtype=np.float32)
        self.target_std = np.array([train_pcs.std() + 1e-8, train_miss.std() + 1e-8], dtype=np.float32)

        # Adapt network input size: Kelvins had 12 features, Space-Track has 8
        # First 8 features are shared (including temporal derivatives).
        # Create new network with 8 inputs, transfer LSTM weights where possible.
        if self.net is not None and self.n_input_features != N_SHARED_FEATURES:
            print(f"  Adapting network: {self.n_input_features} -> {N_SHARED_FEATURES} input features")
            old_state = self.net.state_dict()
            self._init_net(input_size=N_SHARED_FEATURES)
            new_state = self.net.state_dict()

            # Transfer all weights except the first LSTM input projection
            old_n = self.n_input_features  # before adaptation
            for key in new_state:
                if key in old_state and new_state[key].shape == old_state[key].shape:
                    new_state[key] = old_state[key]
                elif "lstm.weight_ih_l0" in key and key in old_state:
                    old_w = old_state[key]  # (4*hidden, old_n)
                    if old_w.shape[1] > N_SHARED_FEATURES:
                        # Shrinking (e.g. 12->8): slice columns
                        new_state[key] = old_w[:, :N_SHARED_FEATURES].clone()
                    else:
                        # Expanding (e.g. 3->8): copy old cols, init new randomly
                        new_state[key][:, :old_w.shape[1]] = old_w
                    print(f"    Adapted {key}: {old_w.shape} -> {new_state[key].shape}")

            self.net.load_state_dict(new_state)
        elif self.net is None:
            self._init_net(input_size=N_SHARED_FEATURES)

        train_ds = _SeqRegressionDataset(train_seqs, train_targets)
        test_ds = _SeqRegressionDataset(test_seqs, test_targets)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=256)

        # Phase 1: Freeze LSTM, train heads (10 epochs)
        print("  Phase 1: Freeze LSTM, train heads...")
        for p in self.net.lstm.parameters():
            p.requires_grad = False
        for p in self.net.attention.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=5e-4
        )

        for epoch in range(10):
            self._train_epoch(train_dl, optimizer)
            if (epoch + 1) % 5 == 0:
                m = self._evaluate(test_dl)
                print(f"    Epoch {epoch+1} | val_MAE_pc={m['mae_log10_pc']:.3f}")

        # Phase 2: Unfreeze all, fine-tune (30 epochs with early stopping)
        print("  Phase 2: Full fine-tune...")
        for p in self.net.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(30):
            self._train_epoch(train_dl, optimizer)
            m = self._evaluate(test_dl)

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1} | val_MAE_pc={m['mae_log10_pc']:.3f} "
                      f"MAE_miss={m['mae_log10_miss']:.3f} esc_F1={m['esc_f1']:.3f}")

            if m["mae_log10_pc"] < best_val:
                best_val = m["mae_log10_pc"]
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 7:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        train_metrics = self._evaluate(train_dl)
        test_metrics = self._evaluate(test_dl)

        print(f"\nSpace-Track Results:")
        print(f"  Train: MAE(log10_pc)={train_metrics['mae_log10_pc']:.3f}, "
              f"MAE(log10_miss)={train_metrics['mae_log10_miss']:.3f}, "
              f"Escalation F1={train_metrics['esc_f1']:.3f}")
        print(f"  Test:  MAE(log10_pc)={test_metrics['mae_log10_pc']:.3f}, "
              f"MAE(log10_miss)={test_metrics['mae_log10_miss']:.3f}, "
              f"Escalation F1={test_metrics['esc_f1']:.3f}")

        self.trained = True
        self.metrics["finetune"] = {"train": train_metrics, "test": test_metrics,
                                     "n_pairs": len(sequences)}
        return self.metrics["finetune"]

    def _train_epoch(self, dl, optimizer):
        self.net.train()
        for x, lengths, y_pc, y_miss, y_esc in dl:
            x = x.to(self.device)
            lengths = lengths.to(self.device)
            y_pc, y_miss, y_esc = y_pc.to(self.device), y_miss.to(self.device), y_esc.to(self.device)

            pred_pc, pred_miss, pred_esc, _ = self.net(x, lengths)

            y_pc_norm = (y_pc - self.target_mean[0]) / self.target_std[0]
            y_miss_norm = (y_miss - self.target_mean[1]) / self.target_std[1]

            loss_pc = nn.functional.huber_loss(pred_pc, y_pc_norm, delta=1.0)
            loss_miss = nn.functional.huber_loss(pred_miss, y_miss_norm, delta=1.0)
            loss_esc = nn.functional.binary_cross_entropy(pred_esc, y_esc)

            loss = 1.0 * loss_pc + 0.5 * loss_miss + 0.3 * loss_esc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()

    def _evaluate(self, dl) -> dict:
        self.net.eval()
        all_pred_pc, all_pred_miss, all_pred_esc = [], [], []
        all_y_pc, all_y_miss, all_y_esc = [], [], []

        with torch.no_grad():
            for x, lengths, y_pc, y_miss, y_esc in dl:
                x, lengths = x.to(self.device), lengths.to(self.device)
                pred_pc, pred_miss, pred_esc, _ = self.net(x, lengths)

                # Denormalize predictions
                pred_pc_denorm = pred_pc.cpu().numpy() * self.target_std[0] + self.target_mean[0]
                pred_miss_denorm = pred_miss.cpu().numpy() * self.target_std[1] + self.target_mean[1]

                all_pred_pc.append(pred_pc_denorm)
                all_pred_miss.append(pred_miss_denorm)
                all_pred_esc.append(pred_esc.cpu().numpy())
                all_y_pc.append(y_pc.numpy())
                all_y_miss.append(y_miss.numpy())
                all_y_esc.append(y_esc.numpy())

        pred_pc = np.concatenate(all_pred_pc)
        pred_miss = np.concatenate(all_pred_miss)
        pred_esc = np.concatenate(all_pred_esc)
        y_pc = np.concatenate(all_y_pc)
        y_miss = np.concatenate(all_y_miss)
        y_esc = np.concatenate(all_y_esc)

        # Regression metrics
        mae_pc = float(np.mean(np.abs(pred_pc - y_pc)))
        mae_miss = float(np.mean(np.abs(pred_miss - y_miss)))
        rmse_pc = float(np.sqrt(np.mean((pred_pc - y_pc) ** 2)))

        # Correlation
        if len(pred_pc) > 2:
            corr_pc = float(np.corrcoef(pred_pc, y_pc)[0, 1]) if np.std(y_pc) > 0 else 0.0
        else:
            corr_pc = 0.0

        # Escalation classification metrics
        esc_preds = (pred_esc >= 0.5).astype(float)
        tp = int(((esc_preds == 1) & (y_esc == 1)).sum())
        fp = int(((esc_preds == 1) & (y_esc == 0)).sum())
        fn = int(((esc_preds == 0) & (y_esc == 1)).sum())
        tn = int(((esc_preds == 0) & (y_esc == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)

        return {
            "mae_log10_pc": round(mae_pc, 4),
            "rmse_log10_pc": round(rmse_pc, 4),
            "mae_log10_miss": round(mae_miss, 4),
            "correlation_pc": round(corr_pc, 4),
            "esc_accuracy": round(acc, 4),
            "esc_precision": round(prec, 4),
            "esc_recall": round(rec, 4),
            "esc_f1": round(f1, 4),
            "n": len(y_pc),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def predict(self, cdm_sequence: list[dict]) -> dict:
        """Predict for a single conjunction pair.

        Returns:
          - predicted_max_log10_pc: continuous risk estimate
          - predicted_min_miss_km: forecasted minimum miss distance
          - escalation_probability: P(exceeds 5e-4)
          - attention_weights: which CDM updates mattered most
        """
        if not self.trained or self.net is None:
            return {"error": "Model not trained"}

        seq = self._cdm_list_to_sequence(cdm_sequence)
        if seq is None or len(seq) < 1:
            return {"error": "Empty sequence"}

        # Normalize
        seq_norm = ((seq - self.norm_mean) / self.norm_std).astype(np.float32)
        L = min(len(seq_norm), MAX_SEQ_LEN)
        padded = np.zeros((MAX_SEQ_LEN, N_SHARED_FEATURES), dtype=np.float32)
        padded[:L] = seq_norm[:L]

        self.net.eval()
        x = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([L]).to(self.device)

        with torch.no_grad():
            pred_pc, pred_miss, pred_esc, attn_weights = self.net(x, lengths)

        # Denormalize
        max_log10_pc = float(pred_pc.item() * self.target_std[0] + self.target_mean[0])
        min_log10_miss = float(pred_miss.item() * self.target_std[1] + self.target_mean[1])
        esc_prob = float(pred_esc.item())

        # Convert to physical units
        predicted_max_pc = 10 ** max_log10_pc
        predicted_min_miss_km = 10 ** min_log10_miss

        # Attention weights for interpretability
        weights = attn_weights[0, :L].cpu().numpy().tolist()

        return {
            "predicted_max_log10_pc": round(max_log10_pc, 3),
            "predicted_max_pc": predicted_max_pc,
            "predicted_min_miss_km": round(predicted_min_miss_km, 2),
            "escalation_probability": round(esc_prob, 4),
            "attention_weights": weights,
        }

    def _cdm_list_to_sequence(self, cdms: list[dict]) -> Optional[np.ndarray]:
        """Convert CDM list to (N, 8) engineered feature array.

        Features per timestep:
          0: log10(Pc)
          1: log10(miss_km)
          2: time_to_tca_hours
          3: delta_log10_pc      (change from previous CDM)
          4: delta_log10_miss    (change from previous CDM)
          5: delta_time_hours    (hours since previous CDM update)
          6: pc_rate             (d(log10_pc) / d(hours) — Pc velocity)
          7: miss_rate           (d(log10_miss) / d(hours) — miss velocity)

        These temporal derivatives let the LSTM learn acceleration patterns
        (e.g. "Pc is increasing AND accelerating" vs "Pc spiked but is leveling off").
        """
        sorted_cdms = sorted(cdms, key=lambda c: c.get("cdm_id", c.get("CDM_ID", 0)))

        # Deduplicate by creation_date
        seen = set()
        deduped = []
        for c in sorted_cdms:
            created = c.get("creation_date", c.get("CREATED", ""))
            if created and created in seen:
                continue
            if created:
                seen.add(created)
            deduped.append(c)

        # First pass: extract raw values
        raw = []
        tca_str = deduped[0].get("tca", deduped[0].get("TCA", "")) if deduped else ""
        try:
            tca_dt = datetime.fromisoformat(tca_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            tca_dt = None

        for c in deduped:
            pc = c.get("pc", c.get("PC", 0))
            if not pc or (isinstance(pc, str) and not pc.strip()):
                continue
            pc = float(pc)
            if pc <= 0:
                continue

            miss_km = c.get("miss_distance_km", c.get("MIN_RNG", 0))
            if isinstance(miss_km, str):
                miss_km = float(miss_km) if miss_km.strip() else 0
            miss_km = float(miss_km) if miss_km else 0

            created = c.get("creation_date", c.get("CREATED", ""))
            ttca = 72.0
            if tca_dt and created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    ttca = max(0, (tca_dt - created_dt).total_seconds() / 3600)
                except (ValueError, AttributeError):
                    pass

            log_pc = math.log10(max(pc, 1e-20))
            log_miss = math.log10(max(miss_km, 0.001))
            raw.append((log_pc, log_miss, ttca))

        if not raw:
            return None

        # Second pass: compute temporal derivatives
        features = []
        for i, (log_pc, log_miss, ttca) in enumerate(raw):
            if i == 0:
                delta_pc = 0.0
                delta_miss = 0.0
                delta_t = 0.0
                pc_rate = 0.0
                miss_rate = 0.0
            else:
                prev_pc, prev_miss, prev_ttca = raw[i - 1]
                delta_pc = log_pc - prev_pc
                delta_miss = log_miss - prev_miss
                delta_t = prev_ttca - ttca  # positive = time elapsed
                if delta_t > 0.01:  # avoid division by tiny intervals
                    pc_rate = delta_pc / delta_t
                    miss_rate = delta_miss / delta_t
                else:
                    pc_rate = 0.0
                    miss_rate = 0.0

            features.append([
                log_pc,       # 0
                log_miss,     # 1
                ttca,         # 2
                delta_pc,     # 3
                delta_miss,   # 4
                delta_t,      # 5
                pc_rate,      # 6
                miss_rate,    # 7
            ])

        return np.array(features, dtype=np.float32)

    def _load_spacetrack_pairs(self, cdm_store_path: str) -> dict:
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

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict() if self.net else None,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "metrics": self.metrics,
            "trained": self.trained,
            "n_input_features": self.n_input_features,
        }, str(p))

    @classmethod
    def load(cls, path: str) -> "CDMSequenceModel":
        model = cls()
        ckpt = torch.load(str(path), map_location=model.device, weights_only=False)
        model.n_input_features = ckpt.get("n_input_features", N_SHARED_FEATURES)
        model._init_net(input_size=model.n_input_features)
        if ckpt.get("model_state_dict"):
            model.net.load_state_dict(ckpt["model_state_dict"])
        model.norm_mean = ckpt["norm_mean"]
        model.norm_std = ckpt["norm_std"]
        model.target_mean = ckpt.get("target_mean")
        model.target_std = ckpt.get("target_std")
        model.metrics = ckpt.get("metrics", {})
        model.trained = ckpt.get("trained", False)
        return model
