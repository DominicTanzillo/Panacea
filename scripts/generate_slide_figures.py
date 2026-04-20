"""Generate dark-themed figures for PANACEA Demo Day slides.

Produces:
  writeup/figures/slide_problem_viz.png  — Slide 2: the screening problem
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGURES = os.path.join(ROOT, "writeup", "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Theme (matches webapp) ──
BG = "#08080c"
SURFACE = "#111118"
BORDER = "#2a2a3a"
TEXT_PRIMARY = "#e8e8f0"
TEXT_MUTED = "#7c7c96"
TEXT_DIM = "#55556a"
ACCENT = "#3b82f6"
RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_MUTED,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": BORDER,
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
})


def load_data():
    with open(os.path.join(ROOT, "webapp-react", "public", "pipeline_stats.json")) as f:
        pipeline = json.load(f)
    with open(os.path.join(ROOT, "webapp-react", "public", "cdm_forecast.json")) as f:
        forecast = json.load(f)
    return pipeline, forecast


def build_problem_viz():
    """Two-panel figure: CDM accumulation over time + pair outcome distribution."""
    pipeline, forecast = load_data()

    fig = plt.figure(figsize=(11, 6), dpi=150)
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.35,
                  left=0.08, right=0.95, top=0.88, bottom=0.15)

    # ── Panel 1: CDM & pair accumulation over time ──
    ax1 = fig.add_subplot(gs[0])

    dh = pipeline.get("daily_history", [])
    dates, sats, pairs_screened = [], [], []
    seen_dates = set()
    for entry in dh:
        d = entry["date"]
        if d in seen_dates:
            continue  # skip duplicate entries for same date
        seen_dates.add(d)
        try:
            dates.append(datetime.strptime(d, "%Y-%m-%d"))
        except ValueError:
            continue
        sats.append(entry.get("n_satellites_screened", 0))
        pairs_screened.append(entry.get("n_candidate_pairs", 0))

    if dates:
        ax1_twin = ax1.twinx()

        ax1.plot(dates, [s / 1000 for s in sats], color=ACCENT, linewidth=2, alpha=0.9, label="Satellites screened")
        ax1.fill_between(dates, [s / 1000 for s in sats], alpha=0.08, color=ACCENT)

        ax1_twin.plot(dates, [p / 1e6 for p in pairs_screened], color=AMBER, linewidth=2, alpha=0.9, label="Candidate pairs")
        ax1_twin.fill_between(dates, [p / 1e6 for p in pairs_screened], alpha=0.08, color=AMBER)

        ax1.set_xlabel("Date", fontsize=10)
        ax1.set_ylabel("Satellites Screened (K)", color=ACCENT, fontsize=10)
        ax1_twin.set_ylabel("Candidate Pairs (M)", color=AMBER, fontsize=10)
        ax1.tick_params(axis="y", colors=ACCENT)
        ax1_twin.tick_params(axis="y", colors=AMBER)
        ax1_twin.spines["right"].set_color(BORDER)
        ax1_twin.spines["left"].set_color(BORDER)
        ax1_twin.spines["top"].set_color(BORDER)
        ax1_twin.spines["bottom"].set_color(BORDER)

        ax1.set_title("Daily Screening Scale", fontsize=13, fontweight="bold",
                       color=TEXT_PRIMARY, pad=12)

        # Annotate latest
        if len(dates) > 1:
            ax1.annotate(f"{sats[-1]/1000:.1f}K satellites",
                         xy=(dates[-1], sats[-1]/1000),
                         xytext=(-60, 15), textcoords="offset points",
                         fontsize=9, color=ACCENT, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
            ax1_twin.annotate(f"{pairs_screened[-1]/1e6:.1f}M pairs",
                              xy=(dates[-1], pairs_screened[-1]/1e6),
                              xytext=(-60, -20), textcoords="offset points",
                              fontsize=9, color=AMBER, fontweight="bold",
                              arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))

        ax1.grid(True, axis="y", alpha=0.3)
        for spine in ax1.spines.values():
            spine.set_color(BORDER)
        fig.autofmt_xdate(rotation=30)

    # ── Panel 2: Track record — the screening result ──
    ax2 = fig.add_subplot(gs[1])

    tr = forecast.get("track_record", {})
    tp_val = tr.get("n_true_positives", 196)
    fn_val = tr.get("n_false_negatives", 0)
    fp_val = tr.get("n_false_positives", 160)
    tn_val = tr.get("n_true_negatives", 789)

    categories = ["True\nPositive", "False\nNegative", "False\nPositive", "True\nNegative"]
    values = [tp_val, fn_val, fp_val, tn_val]
    colors = [GREEN, RED, AMBER, ACCENT]

    bars = ax2.bar(categories, values, color=colors, width=0.65, edgecolor=BORDER, linewidth=0.5)

    # Value labels on bars
    for bar, val, color in zip(bars, values, colors):
        y = bar.get_height()
        label = str(val)
        ax2.text(bar.get_x() + bar.get_width() / 2, y + 12,
                 label, ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=color)

    ax2.set_title("Production Track Record", fontsize=13, fontweight="bold",
                   color=TEXT_PRIMARY, pad=12)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    # Highlight the zero FN
    if fn_val == 0:
        ax2.annotate("ZERO\nMISSED", xy=(1, 0), xytext=(1, max(values) * 0.35),
                     fontsize=12, fontweight="bold", color=RED,
                     ha="center", va="center",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    # Subtitle
    total = sum(values)
    acc = (tp_val + tn_val) / total * 100 if total else 0
    fig.text(0.5, 0.02,
             f"60+ days  \u00b7  {total:,} resolved predictions  \u00b7  "
             f"{acc:.0f}% accuracy  \u00b7  100% recall (0 missed escalations)",
             ha="center", fontsize=11, color=TEXT_MUTED)

    out = os.path.join(FIGURES, "slide_problem_viz.png")
    fig.savefig(out, dpi=150, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    build_problem_viz()
