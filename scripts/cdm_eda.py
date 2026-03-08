#!/usr/bin/env python3
"""Generate publication-quality EDA figures for CDM forecast model."""

import json
import math
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.cdm_forecast import (
    CDMForecastModel, load_cdm_pairs, extract_sequence_features,
    _build_feature_vector, PC_ACTION_THRESHOLD,
)

# Paths
CDM_STORE = ROOT / "data" / "prediction_logs" / "cdm_store.jsonl"
FORECAST_JSON = ROOT / "webapp-react" / "public" / "cdm_forecast.json"
DAILY_SUMMARIES = ROOT / "data" / "prediction_logs" / "daily_summaries.jsonl"
FIG_DIR = ROOT / "writeup" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
BG = "#0a0a1a"
TEXT = "#e0e0e0"
DANGER = "#ff4f5a"
SAFE = "#4fff8a"
NEUTRAL = "#4f8aff"
PURPLE = "#8b5cf6"
WARNING = "#ffb84f"
GRID_ALPHA = 0.15

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=GRID_ALPHA, color="#444", linewidth=0.5)

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Load data ──────────────────────────────────────────────────────────
print("Loading CDM data...")
pair_map = load_cdm_pairs(CDM_STORE)
all_cdms = []
for cdms in pair_map.values():
    all_cdms.extend(cdms)
print(f"  {len(all_cdms)} CDMs, {len(pair_map)} pairs")

with open(FORECAST_JSON) as f:
    forecast = json.load(f)
metrics = forecast.get("model_metrics", {})


# ── Figure 1: Pc distribution ─────────────────────────────────────────
print("\n[1/6] Pc distribution...")
pcs = [c["pc"] for c in all_cdms if c.get("pc") and c["pc"] > 0]
log_pcs = [math.log10(p) for p in pcs]

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
style_ax(ax, "Distribution of Collision Probability (Pc) in CDM Data",
         "log₁₀(Pc)", "Count")

ax.hist(log_pcs, bins=50, color=NEUTRAL, alpha=0.8, edgecolor="#222")

# Threshold lines
ax.axvline(math.log10(1e-4), color=WARNING, linewidth=2, linestyle="--", label="1e-4 screening threshold")
ax.axvline(math.log10(5e-4), color=DANGER, linewidth=2, linestyle="-", label="5e-4 maneuver threshold")

# Annotate regions
ymax = ax.get_ylim()[1]
ax.annotate("All public CDMs\nexceed 1e-4\n(pre-filtered by 18th SDS)",
            xy=(math.log10(1e-4), ymax * 0.85), fontsize=8, color=WARNING,
            ha="center", fontstyle="italic")
ax.annotate("Maneuver planning\nregion (Pc ≥ 5e-4)\n~19% of pairs",
            xy=(math.log10(5e-4) + 0.3, ymax * 0.7), fontsize=8, color=DANGER,
            ha="left", fontweight="bold")

n_above = sum(1 for p in pcs if p >= 5e-4)
ax.text(0.98, 0.95, f"N = {len(pcs):,} CDMs\n{n_above} above 5e-4 ({100*n_above/len(pcs):.1f}%)",
        transform=ax.transAxes, ha="right", va="top", color=TEXT, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#333"))

ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
          labelcolor=TEXT)
save_fig(fig, "cdm_pc_distribution.png")


# ── Figure 2: Pair evolution ──────────────────────────────────────────
print("[2/6] Pair evolution...")

# Find pairs with multiple updates, split by threshold
exceeded_pairs = []
safe_pairs = []
for key, cdms in pair_map.items():
    sorted_cdms = sorted(cdms, key=lambda c: c.get("cdm_id", 0))
    max_pc = max(c.get("pc", 0) or 0 for c in sorted_cdms)
    n = len(sorted_cdms)
    if n < 3:
        continue
    entry = (key, sorted_cdms, max_pc, n)
    if max_pc >= PC_ACTION_THRESHOLD:
        exceeded_pairs.append(entry)
    else:
        safe_pairs.append(entry)

# Pick 2 most interesting from each (most updates)
exceeded_pairs.sort(key=lambda x: x[3], reverse=True)
safe_pairs.sort(key=lambda x: x[3], reverse=True)
selected = exceeded_pairs[:2] + safe_pairs[:2]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, "Pc Evolution for Selected Conjunction Pairs",
         "Hours to TCA", "log₁₀(Pc)")

colors_sel = [DANGER, WARNING, SAFE, NEUTRAL]
for i, (key, cdms, max_pc, n) in enumerate(selected):
    feats = extract_sequence_features(cdms)
    seq = feats["sequence"]
    times = seq[:, 2]  # hours to TCA
    log_pc = seq[:, 0]
    label_exceeded = "exceeded" if max_pc >= PC_ACTION_THRESHOLD else "safe"
    name = f"{cdms[0].get('sat1_name','?')} vs {cdms[0].get('sat2_name','?')}"
    if len(name) > 35:
        name = name[:32] + "..."
    ax.plot(times, log_pc, "o-", color=colors_sel[i], markersize=4,
            linewidth=1.5, label=f"{name} [{label_exceeded}]", alpha=0.9)

ax.axhline(math.log10(PC_ACTION_THRESHOLD), color=DANGER, linewidth=1.5,
           linestyle="--", alpha=0.7, label="5e-4 threshold")
ax.invert_xaxis()
ax.legend(loc="best", fontsize=7, facecolor="#1a1a2e", edgecolor="#333",
          labelcolor=TEXT)
save_fig(fig, "cdm_pair_evolution.png")


# ── Figure 3: Feature importance ──────────────────────────────────────
print("[3/6] Feature importance...")

model = CDMForecastModel()
train_metrics = model.train(CDM_STORE)
print(f"  Model trained: {train_metrics.get('mode', 'unknown')}, "
      f"test acc={train_metrics.get('test', {}).get('accuracy', 'N/A')}")

FEATURE_NAMES = [
    "Latest log₁₀(Pc)", "Latest log₁₀(miss km)", "Time to TCA (hrs)",
    "Pc trend (slope)", "Miss distance trend", "Pc acceleration",
    "Max log₁₀(Pc)", "Min log₁₀(Pc)", "N updates (log)",
    "Sat1 RCS class", "Sat2 RCS class", "Has debris"
]

if model.weights is not None and model._feature_std is not None:
    importance = np.abs(model.weights) * model._feature_std
    order = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    style_ax(ax, "Feature Importance (|weight| × feature std)",
             "Importance", "")

    names_sorted = [FEATURE_NAMES[i] for i in order]
    vals_sorted = importance[order]
    colors_bar = [PURPLE if v < np.median(vals_sorted) else NEUTRAL for v in vals_sorted]
    ax.barh(range(len(order)), vals_sorted, color=colors_bar, edgecolor="#222", height=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_sorted, fontsize=9, color=TEXT)
    save_fig(fig, "cdm_feature_importance.png")
else:
    print("  SKIP: No trained weights available")


# ── Figure 4: Confusion matrix ───────────────────────────────────────
print("[4/6] Confusion matrix...")

test_m = train_metrics.get("test", {})
tp, fp, fn, tn = test_m.get("tp", 0), test_m.get("fp", 0), test_m.get("fn", 0), test_m.get("tn", 0)
total = tp + fp + fn + tn

cm = np.array([[tn, fp], [fn, tp]])
cm_pct = cm / max(total, 1) * 100

fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
style_ax(ax, f"Confusion Matrix (n={total})", "Predicted", "Actual")

# Manual heatmap
cmap_colors = ["#0a2a1a", "#1a3a2a", "#3a1a1a", "#5a1a1a"]
cell_colors = [[SAFE, WARNING], [WARNING, DANGER]]
for i in range(2):
    for j in range(2):
        color = SAFE if i == j else DANGER
        alpha = 0.3 if cm[i, j] < total * 0.1 else 0.7
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor=color, alpha=alpha, edgecolor="#444"))
        ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Safe (<5e-4)", "Exceeds 5e-4"], color=TEXT, fontsize=10)
ax.set_yticklabels(["Safe (<5e-4)", "Exceeds 5e-4"], color=TEXT, fontsize=10)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1.5, -0.5)

# Metrics annotation
acc = test_m.get("accuracy", 0)
prec = test_m.get("precision", 0)
rec = test_m.get("recall", 0)
f1 = test_m.get("f1", 0)
ax.text(1.02, 0.5, f"Accuracy: {acc:.1%}\nPrecision: {prec:.1%}\nRecall: {rec:.1%}\nF1: {f1:.3f}",
        transform=ax.transAxes, va="center", fontsize=10, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333"))

save_fig(fig, "cdm_confusion_matrix.png")


# ── Figure 5: Error analysis ─────────────────────────────────────────
print("[5/6] Error analysis...")

# Rebuild test set to get per-sample predictions
rng = np.random.RandomState(42)
labeled_pairs = []
now = datetime.utcnow()
for pair_key, pair_cdms in pair_map.items():
    sorted_cdms = sorted(pair_cdms, key=lambda c: c.get("cdm_id", 0))
    max_pc = max(c.get("pc", 0) or 0 for c in sorted_cdms)
    label = 1 if max_pc >= PC_ACTION_THRESHOLD else 0
    labeled_pairs.append({"cdms": sorted_cdms, "label": label})

rng.shuffle(labeled_pairs)

X_all, y_all, meta_all = [], [], []
for lp in labeled_pairs:
    feats = extract_sequence_features(lp["cdms"])
    seq = feats["sequence"]
    if seq.shape[0] == 0:
        continue
    n = seq.shape[0]
    obs_len = max(1, (n + 1) // 2)
    obs_seq = seq[:obs_len]
    fv = _build_feature_vector(obs_seq, feats["static"])
    X_all.append(fv)
    y_all.append(lp["label"])
    meta_all.append({"log_pc": float(fv[0]), "n_updates": n})

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.float32)

n_total = len(X)
n_train = max(3, int(n_total * 0.7))
X_test_raw = X[n_train:]
y_test = y[n_train:]
meta_test = meta_all[n_train:]

# Predict on test set
probs = model._predict_proba(X_test_raw)
preds = (probs >= 0.5).astype(float)

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
style_ax(ax, "Error Analysis: Misclassified Test Examples",
         "Latest log₁₀(Pc) (at prediction time)", "Number of CDM Updates")

for i in range(len(y_test)):
    lpc = meta_test[i]["log_pc"]
    nu = meta_test[i]["n_updates"]
    if preds[i] == y_test[i]:
        ax.scatter(lpc, nu, color="#555", alpha=0.3, s=20, zorder=1)
    elif preds[i] == 1 and y_test[i] == 0:  # FP
        ax.scatter(lpc, nu, color=DANGER, alpha=0.9, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    else:  # FN
        ax.scatter(lpc, nu, color=NEUTRAL, alpha=0.9, s=60, zorder=3, edgecolors="white", linewidths=0.5)

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=6, label="Correct", linestyle="None"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=DANGER, markersize=8, label="False Positive", linestyle="None"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NEUTRAL, markersize=8, label="False Negative", linestyle="None"),
]
ax.legend(handles=legend_els, loc="upper left", fontsize=9, facecolor="#1a1a2e",
          edgecolor="#333", labelcolor=TEXT)

ax.axvline(math.log10(PC_ACTION_THRESHOLD), color=DANGER, linewidth=1, linestyle="--", alpha=0.5)

n_fp = int(((preds == 1) & (y_test == 0)).sum())
n_fn = int(((preds == 0) & (y_test == 1)).sum())
ax.text(0.98, 0.95, f"FP: {n_fp}  |  FN: {n_fn}\nTest N: {len(y_test)}",
        transform=ax.transAxes, ha="right", va="top", color=TEXT, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#333"))

save_fig(fig, "cdm_error_analysis.png")


# ── Figure 6: CDM accumulation over time ──────────────────────────────
print("[6/6] CDM accumulation over time...")

# Parse creation dates
dates = []
for c in all_cdms:
    cd = c.get("creation_date", "")
    try:
        dt = datetime.fromisoformat(cd.replace("Z", "+00:00").replace("+00:00", ""))
        dates.append(dt)
    except (ValueError, AttributeError):
        pass

dates.sort()
# Cumulative count by date
from collections import Counter
date_counts = Counter(d.date() for d in dates)
sorted_dates = sorted(date_counts.keys())
cumulative = []
running = 0
xs, ys = [], []
for d in sorted_dates:
    running += date_counts[d]
    xs.append(d)
    ys.append(running)

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
style_ax(ax, "CDM Data Accumulation", "Date", "Cumulative CDMs")

ax.fill_between(xs, ys, color=NEUTRAL, alpha=0.3)
ax.plot(xs, ys, color=NEUTRAL, linewidth=2)

# Daily count on secondary axis
ax2 = ax.twinx()
daily_vals = [date_counts[d] for d in sorted_dates]
ax2.bar(sorted_dates, daily_vals, color=PURPLE, alpha=0.4, width=0.8)
ax2.set_ylabel("Daily new CDMs", color=PURPLE, fontsize=10)
ax2.tick_params(colors=PURPLE, labelsize=9)
ax2.spines["right"].set_color(PURPLE)

# Annotate model accuracy
test_acc = metrics.get("test_accuracy", train_metrics.get("test", {}).get("accuracy", 0))
ax.text(0.02, 0.95,
        f"Total: {len(all_cdms):,} CDMs | {len(pair_map)} pairs\n"
        f"Model accuracy: {test_acc:.1%}",
        transform=ax.transAxes, va="top", fontsize=10, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333"))

fig.autofmt_xdate(rotation=30)
save_fig(fig, "cdm_model_growth.png")


print(f"\nDone! All figures saved to {FIG_DIR}")
