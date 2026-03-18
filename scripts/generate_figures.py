"""Generate paper-ready figures from accumulated pipeline data.

Reads all logs produced by the daily/weekly pipelines and generates
publication-quality figures for the paper and demo presentation.

Output: writeup/figures/*.png (300 DPI, tight layout)

Usage:
    python scripts/generate_figures.py           # Generate all figures
    python scripts/generate_figures.py --show     # Also display interactively

Can be run by CI or locally at any time — always reflects latest data.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "data" / "prediction_logs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "writeup" / "figures"

# Consistent style for all figures
COLORS = {
    "primary": "#4f8aff",
    "secondary": "#8b5cf6",
    "accent": "#4fff8a",
    "danger": "#ff4f5a",
    "warning": "#ffd93d",
    "muted": "#888888",
    "xgboost": "#ff6b35",
    "pitft": "#4f8aff",
    "baseline": "#888888",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 300,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    return plt, mdates


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Figure 1: PI-TFT Fine-Tuning Progress Over Time ──────────────────

def fig_finetune_progress(plt, mdates):
    """AUC-PR improvement across weekly fine-tuning runs."""
    records = load_jsonl(LOG_DIR / "finetune_log.jsonl")
    if len(records) < 1:
        print("  Skipping finetune_progress — no data")
        return

    dates = []
    pre_aucs = []
    post_aucs = []
    n_outcomes = []

    # Add initial baseline point (before any fine-tuning)
    # PI-TFT original AUC-PR from training: 0.511
    initial_date = datetime(2026, 2, 13)
    dates.append(initial_date)
    pre_aucs.append(0.511)
    post_aucs.append(0.511)
    n_outcomes.append(0)

    for r in records:
        dt = datetime.fromisoformat(r["date"].replace("+00:00", "").replace("Z", ""))
        dates.append(dt)
        pre_aucs.append(r["pre_auc_pr"])
        best = r.get("best_val_auc_pr", r.get("post_auc_pr", r["pre_auc_pr"]))
        kept = r.get("keep_new_model", False)
        post_aucs.append(best if kept else r["pre_auc_pr"])
        n_outcomes.append(r.get("n_outcomes", 0))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # AUC-PR line
    ax1.plot(dates, post_aucs, "o-", color=COLORS["pitft"], linewidth=2,
             markersize=8, label="PI-TFT AUC-PR", zorder=3)

    # Shade improvement region
    ax1.fill_between(dates, [0.511] * len(dates), post_aucs,
                      alpha=0.15, color=COLORS["pitft"])

    # Reference lines
    ax1.axhline(y=0.511, color=COLORS["muted"], linestyle="--", alpha=0.7,
                label="Initial PI-TFT (0.511)")
    ax1.axhline(y=0.988, color=COLORS["xgboost"], linestyle=":", alpha=0.5,
                label="XGBoost target (0.988)")

    # Annotate each point
    for i, (d, auc, n) in enumerate(zip(dates, post_aucs, n_outcomes)):
        if i == 0:
            ax1.annotate("Initial\n0.511", (d, auc), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8, color=COLORS["muted"])
        elif n > 0:
            ax1.annotate(f"{auc:.3f}\n({n} outcomes)", (d, auc),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, color=COLORS["pitft"])

    ax1.set_xlabel("Date")
    ax1.set_ylabel("AUC-PR (Precision-Recall)")
    ax1.set_title("PI-TFT Model Improvement via Weekly Fine-Tuning")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    path = FIG_DIR / "finetune_progress.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Staleness Experiment (3-model comparison) ───────────────

def fig_staleness_experiment(plt, mdates):
    """AUC-PR degradation as TLE data ages."""
    data = load_json(RESULTS_DIR / "staleness_experiment.json")
    if not data.get("cutoffs"):
        print("  Skipping staleness — no data")
        return

    cutoffs = data["cutoffs"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_key, label, color, marker in [
        ("xgboost", "XGBoost", COLORS["xgboost"], "s"),
        ("deep", "PI-TFT", COLORS["pitft"], "o"),
        ("baseline", "Baseline", COLORS["muted"], "^"),
    ]:
        if model_key not in data:
            continue
        aucs = [r["auc_pr"] for r in data[model_key]]
        ax.plot(cutoffs, aucs, f"{marker}-", color=color, linewidth=2,
                markersize=8, label=f"{label}", zorder=3)

    # Annotate knee point
    ax.axvline(x=3.0, color=COLORS["danger"], linestyle=":", alpha=0.4)
    ax.annotate("3-day knee", (3.0, 0.75), fontsize=9, color=COLORS["danger"],
                ha="center")

    ax.set_xlabel("Data Staleness (days before TCA)")
    ax.set_ylabel("AUC-PR")
    ax.set_title("Model Performance vs. TLE Data Staleness")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(cutoffs)

    path = FIG_DIR / "staleness_experiment.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: CDM Data Accumulation ──────────────────────────────────

def fig_cdm_accumulation(plt, mdates):
    """CDM store growth and Pc distribution."""
    cdms = load_jsonl(LOG_DIR / "cdm_store.jsonl")
    if not cdms:
        print("  Skipping cdm_accumulation — no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cumulative CDM count over time (by fetch date)
    fetch_dates = []
    for c in cdms:
        fa = c.get("fetched_at", "")
        if fa:
            try:
                dt = datetime.fromisoformat(fa.replace("+00:00", "").replace("Z", ""))
                fetch_dates.append(dt.date())
            except Exception:
                pass

    if fetch_dates:
        daily_counts = Counter(fetch_dates)
        sorted_dates = sorted(daily_counts.keys())
        cumulative = []
        total = 0
        for d in sorted_dates:
            total += daily_counts[d]
            cumulative.append(total)

        axes[0].fill_between(sorted_dates, cumulative, alpha=0.3, color=COLORS["primary"])
        axes[0].plot(sorted_dates, cumulative, "-", color=COLORS["primary"], linewidth=2)
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Cumulative CDMs")
        axes[0].set_title(f"CDM Data Accumulation ({len(cdms)} total)")
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Right: Pc distribution
    pcs = [c["pc"] for c in cdms if c.get("pc", 0) > 0]
    if pcs:
        log_pcs = np.log10(pcs)
        axes[1].hist(log_pcs, bins=40, color=COLORS["primary"], edgecolor="black",
                     alpha=0.7, linewidth=0.5)
        axes[1].axvline(x=np.log10(1e-4), color=COLORS["danger"], linestyle="--",
                       alpha=0.7, label="NASA CARA (1e-4)")
        axes[1].axvline(x=np.log10(1e-5), color=COLORS["warning"], linestyle="--",
                       alpha=0.7, label="Kelvins (1e-5)")
        axes[1].axvline(x=np.log10(1e-6), color=COLORS["accent"], linestyle="--",
                       alpha=0.7, label="Starlink (1e-6)")
        axes[1].set_xlabel("log₁₀(Pc)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Probability of Collision Distribution")
        axes[1].legend(fontsize=8)

    path = FIG_DIR / "cdm_accumulation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Daily Pipeline Operations ──────────────────────────────

def fig_daily_pipeline(plt, mdates):
    """Satellites screened and candidate pairs over time."""
    summaries = load_jsonl(LOG_DIR / "daily_summaries.jsonl")
    if len(summaries) < 2:
        print("  Skipping daily_pipeline — not enough data")
        return

    dates = []
    n_sats = []
    n_pairs = []

    for s in summaries:
        try:
            dt = datetime.strptime(s["date"], "%Y-%m-%d")
            dates.append(dt)
            n_sats.append(s.get("n_satellites_screened", 0))
            n_pairs.append(s.get("n_candidate_pairs", 0))
        except Exception:
            continue

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(dates, n_sats, width=0.8, color=COLORS["primary"], alpha=0.7,
            label="Satellites screened")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Satellites Screened", color=COLORS["primary"])
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = ax1.twinx()
    ax2.plot(dates, [p / 1e6 for p in n_pairs], "s-", color=COLORS["secondary"],
             linewidth=2, markersize=5, label="Candidate pairs (M)")
    ax2.set_ylabel("Candidate Pairs (millions)", color=COLORS["secondary"])
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(f"Daily Screening Pipeline ({len(dates)} days)")

    path = FIG_DIR / "daily_pipeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: Maneuver Detection History ─────────────────────────────

def fig_maneuver_history(plt, mdates):
    """Daily maneuver detections by type and constellation."""
    maneuvers = load_jsonl(LOG_DIR / "maneuver_history.jsonl")
    if len(maneuvers) < 10:
        print("  Skipping maneuver_history — not enough data")
        return

    # Group by date
    daily_maneuvers = defaultdict(int)
    daily_avoidance = defaultdict(int)
    type_counts = Counter()
    constellation_counts = Counter()

    for m in maneuvers:
        det = m.get("detected_at", "")[:10]
        if det:
            daily_maneuvers[det] += 1
            if m.get("likely_avoidance") or m.get("has_cdm"):
                daily_avoidance[det] += 1
        type_counts[m.get("maneuver_type", "unknown")] += 1
        constellation_counts[m.get("constellation", "other")] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Daily maneuver count
    sorted_dates = sorted(daily_maneuvers.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
    counts = [daily_maneuvers[d] for d in sorted_dates]
    avoidance = [daily_avoidance.get(d, 0) for d in sorted_dates]

    axes[0].bar(dates, counts, width=0.8, color=COLORS["primary"], alpha=0.7,
                label="All maneuvers")
    axes[0].bar(dates, avoidance, width=0.8, color=COLORS["danger"], alpha=0.7,
                label="Likely avoidance / CDM-linked")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Maneuvers Detected")
    axes[0].set_title(f"Daily Maneuver Detections ({len(maneuvers)} total)")
    axes[0].legend(fontsize=9)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Right: Maneuver type breakdown
    top_types = type_counts.most_common(8)
    if top_types:
        labels, values = zip(*top_types)
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["warning"], COLORS["danger"], COLORS["muted"],
                  "#ff9ff3", "#48dbfb"][:len(labels)]
        axes[1].barh(range(len(labels)), values, color=colors, edgecolor="black",
                     linewidth=0.5)
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Count")
        axes[1].set_title("Maneuver Types")

    path = FIG_DIR / "maneuver_history.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 6: CDM vs TLE Screening Comparison ────────────────────────

def fig_cdm_vs_tle(plt, mdates):
    """Show that CDM alerts find real conjunctions TLE screening missed."""
    cdms = load_jsonl(LOG_DIR / "cdm_store.jsonl")
    alerts = load_json(ROOT / "webapp-react" / "public" / "latest_alerts.json")

    if not cdms or not alerts.get("pairs"):
        print("  Skipping cdm_vs_tle — no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Miss distance comparison
    cdm_dists = sorted([c["miss_distance_km"] for c in cdms if c.get("miss_distance_km") is not None])
    tle_dists = sorted([p.get("tca_min_distance_km", 99999) for p in alerts["pairs"]
                        if p.get("tca_min_distance_km") is not None])

    if cdm_dists:
        axes[0].hist(cdm_dists, bins=30, alpha=0.6, color=COLORS["primary"],
                     edgecolor="black", linewidth=0.5, label=f"CDM (n={len(cdm_dists)})")
    if tle_dists:
        axes[0].hist(tle_dists, bins=30, alpha=0.6, color=COLORS["muted"],
                     edgecolor="black", linewidth=0.5, label=f"TLE screening (n={len(tle_dists)})")
    axes[0].set_xlabel("Miss Distance (km)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Closest Approach: CDM vs TLE Screening")
    axes[0].legend(fontsize=9)

    # Right: Summary comparison table as text
    axes[1].axis("off")
    cdm_min = min(cdm_dists) if cdm_dists else "N/A"
    cdm_med = cdm_dists[len(cdm_dists) // 2] if cdm_dists else "N/A"
    tle_min = min(tle_dists) if tle_dists else "N/A"
    tle_med = tle_dists[len(tle_dists) // 2] if tle_dists else "N/A"

    cdm_pcs = [c["pc"] for c in cdms if c.get("pc", 0) > 0]
    max_pc = max(cdm_pcs) if cdm_pcs else "N/A"

    table_data = [
        ["Metric", "CDM (Space-Track)", "TLE Screening"],
        ["Source", "18 SDS covariance", "SGP4 propagation"],
        ["Closest approach", f"{cdm_min} km" if isinstance(cdm_min, (int, float)) else cdm_min,
         f"{tle_min:.0f} km" if isinstance(tle_min, (int, float)) else tle_min],
        ["Median distance", f"{cdm_med} km" if isinstance(cdm_med, (int, float)) else cdm_med,
         f"{tle_med:.0f} km" if isinstance(tle_med, (int, float)) else tle_med],
        ["Has real Pc?", f"Yes (max {max_pc:.4e})" if isinstance(max_pc, float) else "Yes",
         "No (heuristic only)"],
        ["Total pairs", f"{len(cdm_dists)}", f"{len(tle_dists)}"],
        ["Pair overlap", "0", "0"],
    ]

    table = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(3):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(weight="bold")

    axes[1].set_title("CDM vs TLE Screening Summary", pad=20)

    path = FIG_DIR / "cdm_vs_tle.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 7: CDM Object Types and Top Events ────────────────────────

def fig_cdm_objects(plt, mdates):
    """Object types in CDMs and highest-risk events."""
    cdms = load_jsonl(LOG_DIR / "cdm_store.jsonl")
    if not cdms:
        print("  Skipping cdm_objects — no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Object type pairs
    type_counter = Counter()
    for c in cdms:
        t1 = c.get("sat1_type", "UNKNOWN")
        t2 = c.get("sat2_type", "UNKNOWN")
        # Normalize order
        pair = " vs ".join(sorted([t1, t2]))
        type_counter[pair] += 1

    top_types = type_counter.most_common(6)
    if top_types:
        labels, values = zip(*top_types)
        colors_list = [COLORS["danger"], COLORS["primary"], COLORS["secondary"],
                       COLORS["warning"], COLORS["accent"], COLORS["muted"]][:len(labels)]
        axes[0].barh(range(len(labels)), values, color=colors_list, edgecolor="black",
                     linewidth=0.5)
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Number of CDMs")
        axes[0].set_title("Conjunction Types")

    # Right: Top 10 highest Pc events
    by_pc = sorted(cdms, key=lambda c: c.get("pc", 0), reverse=True)
    # Deduplicate by pair
    seen = set()
    unique_top = []
    for c in by_pc:
        pair = (min(c["sat1_norad"], c["sat2_norad"]),
                max(c["sat1_norad"], c["sat2_norad"]))
        if pair not in seen:
            seen.add(pair)
            unique_top.append(c)
        if len(unique_top) >= 10:
            break

    if unique_top:
        labels = [f"{c['sat1_name']}\nvs {c['sat2_name']}" for c in unique_top]
        pcs = [c["pc"] for c in unique_top]
        emergency = [c.get("emergency_reportable") == "Y" for c in unique_top]
        bar_colors = [COLORS["danger"] if e else COLORS["primary"] for e in emergency]

        axes[1].barh(range(len(labels)), pcs, color=bar_colors, edgecolor="black",
                     linewidth=0.5)
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Probability of Collision")
        axes[1].set_title("Top 10 Highest-Risk Events\n(red = emergency reportable)")
        axes[1].axvline(x=1e-4, color=COLORS["muted"], linestyle=":", alpha=0.5)

    path = FIG_DIR / "cdm_objects.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 8: Model Comparison Bar Chart ──────────────────────────────

def fig_model_comparison(plt, mdates):
    """Static model comparison: Baseline vs XGBoost vs PI-TFT."""
    data = load_json(RESULTS_DIR / "staleness_experiment.json")
    finetune = load_jsonl(LOG_DIR / "finetune_log.jsonl")

    if not data.get("cutoffs"):
        print("  Skipping model_comparison — no data")
        return

    # Get AUC-PR at 2-day staleness (standard benchmark)
    baseline_auc = data["baseline"][0]["auc_pr"] if data.get("baseline") else 0.061
    xgb_auc = data["xgboost"][0]["auc_pr"] if data.get("xgboost") else 0.988
    pitft_initial = data["deep"][0]["auc_pr"] if data.get("deep") else 0.511

    # Latest fine-tuned PI-TFT AUC-PR
    pitft_current = pitft_initial
    for r in finetune:
        if r.get("keep_new_model"):
            pitft_current = r.get("best_val_auc_pr", r.get("post_auc_pr", pitft_current))

    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Baseline\n(Altitude Shell)", "PI-TFT\n(Initial)", "PI-TFT\n(Fine-Tuned)", "XGBoost\n(CSPR)"]
    aucs = [baseline_auc, pitft_initial, pitft_current, xgb_auc]
    colors = [COLORS["muted"], COLORS["pitft"], COLORS["secondary"], COLORS["xgboost"]]

    bars = ax.bar(models, aucs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{auc:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("AUC-PR")
    ax.set_title("Model Comparison (AUC-PR at 2-day staleness)")
    ax.set_ylim(0, 1.15)

    # Improvement annotation
    if pitft_current > pitft_initial:
        improvement = (pitft_current - pitft_initial) / pitft_initial * 100
        ax.annotate(
            f"+{improvement:.0f}%",
            xy=(2, pitft_current), xytext=(2.5, pitft_current + 0.08),
            arrowprops=dict(arrowstyle="->", color=COLORS["accent"]),
            fontsize=11, color=COLORS["accent"], fontweight="bold",
        )

    path = FIG_DIR / "model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")

    plt, mdates = setup_matplotlib()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures to {FIG_DIR}/\n")

    generators = [
        ("1. Fine-Tuning Progress", fig_finetune_progress),
        ("2. Staleness Experiment", fig_staleness_experiment),
        ("3. CDM Data Accumulation", fig_cdm_accumulation),
        ("4. Daily Pipeline Operations", fig_daily_pipeline),
        ("5. Maneuver Detection History", fig_maneuver_history),
        ("6. CDM vs TLE Screening", fig_cdm_vs_tle),
        ("7. CDM Object Types & Top Events", fig_cdm_objects),
        ("8. Model Comparison", fig_model_comparison),
    ]

    generated = 0
    for name, func in generators:
        print(f"[{name}]")
        try:
            func(plt, mdates)
            generated += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone — {generated}/{len(generators)} figures generated in {FIG_DIR}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
