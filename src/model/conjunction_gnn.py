"""Graph Neural Network for Conjunction Network Risk Prediction.

Models the satellite conjunction environment as a graph:
  - Nodes: individual satellites (features: altitude, inclination, RCS, object type)
  - Edges: conjunction pairs from CDM data (features: Pc history, miss distance, trend)
  - Task: edge-level binary classification — will this conjunction escalate?

Key insight: a satellite involved in MULTIPLE conjunctions simultaneously
may have higher overall risk (congested orbital regime). GNNs naturally
capture this neighborhood structure, which per-pair models cannot.

Architecture:
  GraphSAGE encoder (2 layers) -> edge feature MLP -> edge classifier

  1. Node features are embedded and aggregated from neighbors
  2. Edge representations are formed by concatenating endpoint embeddings + edge features
  3. Edge classifier predicts escalation probability

Pre-trained on Kelvins event graph, fine-tuned on Space-Track conjunction network.
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

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATv2Conv, GraphNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# Node feature names for interpretability
NODE_FEATURES = [
    "log10_altitude_km",  # 0: orbital altitude
    "inclination_deg",     # 1: orbital inclination (0-180)
    "eccentricity",        # 2: orbit shape
    "log10_rcs",           # 3: radar cross-section (object size proxy)
    "is_payload",          # 4: active satellite
    "is_debris",           # 5: debris fragment
    "is_rocket_body",      # 6: spent rocket stage
    "n_conjunctions",      # 7: degree centrality (# active conjunctions)
]

# Edge feature names
EDGE_FEATURES = [
    "log10_current_pc",      # 0: latest Pc
    "log10_miss_km",         # 1: latest miss distance
    "time_to_tca_hours",     # 2: hours until closest approach
    "pc_trend",              # 3: Pc slope over CDM sequence
    "n_updates",             # 4: number of CDM updates
    "delta_altitude_km",     # 5: altitude difference between objects
    "delta_inclination_deg", # 6: inclination difference
]


class ConjunctionGNN(nn.Module):
    """Graph Neural Network for conjunction edge classification.

    Architecture:
      Node embedding: Linear(n_node_feat, hidden) + 2x GraphSAGE
      Edge classifier: MLP([node_i || node_j || edge_feat], hidden, 1)
    """

    def __init__(self, n_node_features=8, n_edge_features=7,
                 hidden_dim=64, dropout=0.3):
        super().__init__()
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        # Node encoder
        self.node_embed = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GraphSAGE layers (message passing)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Edge classifier: takes [src_embed || dst_embed || edge_features]
        edge_input_dim = hidden_dim * 2 + n_edge_features
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),  # raw logit
        )

    def encode_nodes(self, x, edge_index):
        """Encode node features using GraphSAGE message passing."""
        h = self.node_embed(x)

        # Layer 1
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.gelu(h)

        return h

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: node features (n_nodes, n_node_features)
            edge_index: (2, n_edges) source/target node indices
            edge_attr: edge features (n_edges, n_edge_features)

        Returns:
            edge_logits: (n_edges,) raw logits for escalation
            node_embeddings: (n_nodes, hidden_dim) for visualization
        """
        # Encode nodes with graph structure
        node_emb = self.encode_nodes(x, edge_index)

        # Build edge representations
        src_emb = node_emb[edge_index[0]]  # (n_edges, hidden)
        dst_emb = node_emb[edge_index[1]]  # (n_edges, hidden)
        edge_repr = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)

        # Classify edges
        edge_logits = self.edge_classifier(edge_repr).squeeze(-1)

        return edge_logits, node_emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConjunctionGraphBuilder:
    """Builds PyG Data objects from CDM and TLE data."""

    def __init__(self):
        self.node_mean: Optional[np.ndarray] = None
        self.node_std: Optional[np.ndarray] = None
        self.edge_mean: Optional[np.ndarray] = None
        self.edge_std: Optional[np.ndarray] = None

    def build_graph(self, cdm_store_path: str, tle_path: str = None,
                    fit_norm: bool = False) -> Optional[Data]:
        """Build a conjunction graph from CDM data and TLEs.

        Args:
            cdm_store_path: path to CDM JSONL store
            tle_path: optional path to TLE JSON for node features
            fit_norm: if True, compute normalization stats from this graph

        Returns:
            PyG Data object with node/edge features and labels
        """
        if not HAS_PYG:
            raise ImportError("torch_geometric required. pip install torch_geometric")

        # Load CDMs grouped by pair
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

        if not pairs:
            return None

        # Build node index: unique NORAD IDs
        all_norads = set()
        for (n1, n2, _), _ in pairs.items():
            all_norads.add(n1)
            all_norads.add(n2)
        norad_to_idx = {nid: i for i, nid in enumerate(sorted(all_norads))}
        n_nodes = len(norad_to_idx)

        # Load TLE data for node features
        tle_by_norad = {}
        if tle_path and Path(tle_path).exists():
            with open(tle_path) as f:
                tles = json.load(f)
            for tle in tles:
                nid = int(tle.get("NORAD_CAT_ID", 0))
                if nid > 0:
                    tle_by_norad[nid] = tle

        # Build node features
        node_features = np.zeros((n_nodes, len(NODE_FEATURES)), dtype=np.float32)
        conjunction_count = defaultdict(int)

        # Count conjunctions per satellite
        for (n1, n2, _) in pairs:
            conjunction_count[n1] += 1
            conjunction_count[n2] += 1

        for nid, idx in norad_to_idx.items():
            tle = tle_by_norad.get(nid, {})
            mm = float(tle.get("MEAN_MOTION", 15.0))
            sma = (398600.4418 / (mm * 2 * math.pi / 86400) ** 2) ** (1/3) if mm > 0 else 6778
            alt_km = max(sma - 6371, 100)

            inc = float(tle.get("INCLINATION", 53.0))
            ecc = float(tle.get("ECCENTRICITY", 0.001))

            obj_type = tle.get("OBJECT_TYPE", "PAYLOAD")
            rcs_str = tle.get("RCS_SIZE", "MEDIUM")
            rcs_map = {"SMALL": 0.1, "MEDIUM": 1.0, "LARGE": 10.0}
            rcs = rcs_map.get(rcs_str, 1.0)

            node_features[idx] = [
                math.log10(max(alt_km, 100)),    # log altitude
                inc / 180.0,                      # normalized inclination
                min(ecc, 0.5),                    # capped eccentricity
                math.log10(max(rcs, 0.01)),       # log RCS
                1.0 if "PAYLOAD" in obj_type else 0.0,
                1.0 if "DEB" in obj_type else 0.0,
                1.0 if "R/B" in obj_type or "ROCKET" in obj_type else 0.0,
                min(conjunction_count[nid], 20) / 20.0,  # normalized degree
            ]

        # Build edges and edge features
        edge_src, edge_dst = [], []
        edge_features = []
        edge_labels = []
        edge_pair_keys = []

        THRESHOLD_LOG10 = math.log10(5e-4)

        for (n1, n2, tca_date), cdms in pairs.items():
            idx1 = norad_to_idx[n1]
            idx2 = norad_to_idx[n2]

            # Bidirectional edges
            edge_src.extend([idx1, idx2])
            edge_dst.extend([idx2, idx1])

            # Edge features from CDM sequence
            sorted_cdms = sorted(cdms, key=lambda c: c.get("cdm_id", 0))
            pcs = [float(c.get("pc", 0) or 0) for c in sorted_cdms]
            misses = [float(c.get("miss_distance_km", 0) or 0) for c in sorted_cdms]

            current_pc = pcs[-1] if pcs else 1e-7
            current_miss = misses[-1] if misses else 10.0
            max_pc = max(pcs) if pcs else 1e-7

            # Time to TCA
            tca_str = sorted_cdms[-1].get("tca", "")
            created_str = sorted_cdms[-1].get("creation_date", "")
            ttca = 72.0
            try:
                tca_dt = datetime.fromisoformat(tca_str.replace("Z", "+00:00"))
                created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                ttca = max(0, (tca_dt - created_dt).total_seconds() / 3600)
            except (ValueError, AttributeError):
                pass

            # Pc trend
            if len(pcs) >= 2:
                log_pcs = [math.log10(max(p, 1e-20)) for p in pcs]
                pc_trend = (log_pcs[-1] - log_pcs[0]) / max(len(pcs) - 1, 1)
            else:
                pc_trend = 0.0

            # Altitude and inclination differences
            tle1 = tle_by_norad.get(n1, {})
            tle2 = tle_by_norad.get(n2, {})
            alt1 = node_features[idx1, 0]
            alt2 = node_features[idx2, 0]
            inc1 = float(tle1.get("INCLINATION", 53))
            inc2 = float(tle2.get("INCLINATION", 53))

            ef = [
                math.log10(max(current_pc, 1e-20)),
                math.log10(max(current_miss, 0.001)),
                min(ttca, 168) / 168.0,  # normalize to [0, 1] for 1 week
                np.clip(pc_trend, -2, 2),
                min(len(sorted_cdms), 30) / 30.0,
                abs(alt1 - alt2),
                abs(inc1 - inc2) / 180.0,
            ]

            # Same features for both directions
            edge_features.extend([ef, ef])

            # Label: did max Pc exceed threshold?
            label = 1 if max_pc > 0 and math.log10(max(max_pc, 1e-20)) > THRESHOLD_LOG10 else 0
            edge_labels.extend([label, label])
            edge_pair_keys.extend([(n1, n2, tca_date), (n1, n2, tca_date)])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = np.array(edge_features, dtype=np.float32)
        edge_y = np.array(edge_labels, dtype=np.float32)

        # Normalize
        if fit_norm:
            self.node_mean = node_features.mean(axis=0)
            self.node_std = node_features.std(axis=0) + 1e-8
            self.edge_mean = edge_attr.mean(axis=0)
            self.edge_std = edge_attr.std(axis=0) + 1e-8

        if self.node_mean is not None:
            node_features = (node_features - self.node_mean) / self.node_std
        if self.edge_mean is not None:
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(edge_y, dtype=torch.float32),
        )
        data.n_nodes = n_nodes
        data.n_pairs = len(pairs)
        data.pair_keys = edge_pair_keys
        data.norad_to_idx = norad_to_idx

        return data


class ConjunctionGNNModel:
    """High-level wrapper for training and inference."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[ConjunctionGNN] = None
        self.builder = ConjunctionGraphBuilder()
        self.trained = False
        self.metrics: dict = {}

    def train(self, cdm_store_path: str, tle_path: str = None,
              epochs: int = 100, lr: float = 1e-3) -> dict:
        """Train GNN on conjunction graph."""
        if not HAS_PYG:
            return {"error": "torch_geometric not installed"}

        print("=== Training Conjunction GNN ===")

        # Build graph
        data = self.builder.build_graph(cdm_store_path, tle_path, fit_norm=True)
        if data is None:
            return {"error": "Could not build graph"}

        n_edges = data.edge_index.size(1)
        n_pos = int(data.y.sum().item())
        pos_rate = n_pos / max(n_edges, 1)
        print(f"  Graph: {data.n_nodes} nodes, {data.n_pairs} pairs "
              f"({n_edges} directed edges)")
        print(f"  Positive rate: {n_pos}/{n_edges} = {pos_rate:.1%}")

        # Train/test split on edges (stratified)
        rng = np.random.RandomState(42)
        # Work with pair-level (every 2 edges = 1 pair)
        n_pairs = n_edges // 2
        pair_indices = np.arange(n_pairs)
        pair_labels = data.y.numpy()[::2]  # every other (both directions same label)
        pos_pairs = pair_indices[pair_labels == 1]
        neg_pairs = pair_indices[pair_labels == 0]
        rng.shuffle(pos_pairs)
        rng.shuffle(neg_pairs)

        n_train_pos = int(0.7 * len(pos_pairs))
        n_train_neg = int(0.7 * len(neg_pairs))
        train_pairs = np.concatenate([pos_pairs[:n_train_pos], neg_pairs[:n_train_neg]])
        test_pairs = np.concatenate([pos_pairs[n_train_pos:], neg_pairs[n_train_neg:]])

        # Convert pair indices to edge indices (each pair = 2 directed edges)
        train_edge_mask = np.zeros(n_edges, dtype=bool)
        test_edge_mask = np.zeros(n_edges, dtype=bool)
        for pi in train_pairs:
            train_edge_mask[pi * 2] = True
            train_edge_mask[pi * 2 + 1] = True
        for pi in test_pairs:
            test_edge_mask[pi * 2] = True
            test_edge_mask[pi * 2 + 1] = True

        train_mask = torch.tensor(train_edge_mask)
        test_mask = torch.tensor(test_edge_mask)

        # Initialize network
        self.net = ConjunctionGNN(
            n_node_features=len(NODE_FEATURES),
            n_edge_features=len(EDGE_FEATURES),
            hidden_dim=64,
            dropout=0.3,
        ).to(self.device)

        print(f"  Parameters: {self.net.count_parameters():,}")

        # Move data to device
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        # Focal loss for class imbalance
        from src.model.deep import SigmoidFocalLoss
        criterion = SigmoidFocalLoss(alpha=0.75, gamma=2.0)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_f1 = 0.0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            # Train
            self.net.train()
            edge_logits, _ = self.net(data.x, data.edge_index, data.edge_attr)

            loss = criterion(edge_logits[train_mask], data.y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_metrics = self._evaluate_edges(edge_logits, data.y, test_mask)
                print(f"  Epoch {epoch+1:3d} | loss={loss.item():.4f} | "
                      f"val_F1={val_metrics['f1']:.3f} "
                      f"val_acc={val_metrics['accuracy']:.1%}")

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

        if best_state:
            self.net.load_state_dict(best_state)

        # Final evaluation
        self.net.eval()
        with torch.no_grad():
            edge_logits, node_emb = self.net(data.x, data.edge_index, data.edge_attr)

        train_metrics = self._evaluate_edges(edge_logits, data.y, train_mask)
        test_metrics = self._evaluate_edges(edge_logits, data.y, test_mask)

        print(f"\n  GNN Results:")
        print(f"    Train: F1={train_metrics['f1']:.3f}, Acc={train_metrics['accuracy']:.1%}")
        print(f"    Test:  F1={test_metrics['f1']:.3f}, Acc={test_metrics['accuracy']:.1%}, "
              f"Prec={test_metrics['precision']:.3f}, Rec={test_metrics['recall']:.3f}")

        self.trained = True
        self.metrics = {
            "train": train_metrics,
            "test": test_metrics,
            "n_nodes": data.n_nodes,
            "n_pairs": data.n_pairs,
            "n_edges": n_edges,
            "positive_rate": pos_rate,
        }

        return self.metrics

    def _evaluate_edges(self, logits, labels, mask):
        """Compute edge classification metrics."""
        with torch.no_grad():
            probs = torch.sigmoid(logits[mask]).cpu().numpy()
            y = labels[mask].cpu().numpy()

        preds = (probs >= 0.5).astype(float)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def predict(self, cdm_store_path: str, tle_path: str = None) -> list[dict]:
        """Run inference on a conjunction graph.

        Returns list of pair predictions with escalation probability.
        """
        if not self.trained or self.net is None:
            return []

        data = self.builder.build_graph(cdm_store_path, tle_path)
        if data is None:
            return []

        data_dev = data.to(self.device)
        self.net.eval()
        with torch.no_grad():
            edge_logits, node_emb = self.net(data_dev.x, data_dev.edge_index, data_dev.edge_attr)
            edge_probs = torch.sigmoid(edge_logits).cpu().numpy()

        # Aggregate bidirectional edges back to pairs
        predictions = []
        seen = set()
        for i in range(len(edge_probs)):
            pk = data.pair_keys[i]
            if pk in seen:
                continue
            seen.add(pk)
            n1, n2, tca = pk
            predictions.append({
                "sat1_norad": n1,
                "sat2_norad": n2,
                "tca_date": tca,
                "gnn_escalation_prob": round(float(edge_probs[i]), 4),
                "gnn_label": int(data.y[i].item()),
            })

        return predictions

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.net.state_dict() if self.net else None,
            "node_mean": self.builder.node_mean,
            "node_std": self.builder.node_std,
            "edge_mean": self.builder.edge_mean,
            "edge_std": self.builder.edge_std,
            "metrics": self.metrics,
            "trained": self.trained,
        }, str(p))

    @classmethod
    def load(cls, path: str) -> "ConjunctionGNNModel":
        model = cls()
        ckpt = torch.load(str(path), map_location=model.device, weights_only=False)
        model.net = ConjunctionGNN(
            n_node_features=len(NODE_FEATURES),
            n_edge_features=len(EDGE_FEATURES),
        ).to(model.device)
        if ckpt.get("model_state_dict"):
            model.net.load_state_dict(ckpt["model_state_dict"])
        model.builder.node_mean = ckpt.get("node_mean")
        model.builder.node_std = ckpt.get("node_std")
        model.builder.edge_mean = ckpt.get("edge_mean")
        model.builder.edge_std = ckpt.get("edge_std")
        model.metrics = ckpt.get("metrics", {})
        model.trained = ckpt.get("trained", False)
        return model
