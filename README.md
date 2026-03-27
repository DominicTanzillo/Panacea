# PANACEA

**ML-based satellite conjunction assessment from public CDM data**

[Live Dashboard](https://dominictanzillo.github.io/Panacea/) · [Methodology Paper](writeup/methodology.md) · [Models on HuggingFace](https://huggingface.co/DTanzillo/panacea-models)

AIPI 540 Final Project — Duke University

---

## What It Does

Every day, the US military issues thousands of Conjunction Data Messages (CDMs) warning satellite operators about potential collisions. Most resolve harmlessly — but the few that escalate look identical early on.

PANACEA predicts **where those warnings are headed**. Given a sequence of CDM updates, our ensemble predicts whether collision probability (Pc) will exceed the 5e-4 maneuver-planning threshold before TCA.

**98.7% recall** — catches 74 out of 75 escalating events — with conformal prediction providing calibrated uncertainty bounds.

```bash
pip install panacea-ssa
panacea predict --cdm-store my_cdms.jsonl -o predictions.json
```

## Key Results

5-fold cross-validation on 670 conjunction pairs from Space-Track public CDMs (826 total in corpus, 670 used for CV):

| Model | F1 | Recall | Precision | Role |
|-------|-----|--------|-----------|------|
| Logistic Regression | 0.854 | 0.899 | 0.813 | Interpretable baseline (40% weight) |
| BiLSTM + Transfer | 0.597 | 0.518 | 0.849 | Temporal patterns (30% weight) |
| **Ensemble** | **0.847** | **0.987** | **0.743** | **Production model** |

Supplementary models:

| Model | Key Metric | Value |
|-------|-----------|-------|
| Graph Neural Network | F1 / Recall | 0.962 / 1.000 |
| Autoregressive Forecaster | MAE(log₁₀ Pc) | 0.093 |
| Conformal Prediction (α=0.05) | Coverage | 97.0% |

## How It Works

```
Space-Track CDMs ─── Feature Engineering (23 features) ─── Ensemble Prediction
       │                     │                                      │
       │              Pc trends, volatility,               40% LogReg (interpretable)
       │              RCS, debris flags,                   30% BiLSTM (temporal)
       │              space weather (F10.7, Kp, Ap)        30% Regression signal
       │                                                          │
       v                                                          v
  Daily Pipeline ──────── Retrain on resolved pairs ──── Conformal UQ ──── Webapp
  (GitHub Actions)         (TCA in past = known label)   (coverage bounds)  (GitHub Pages)
  00:00 UTC daily          826+ pairs, 4,590+ CDMs
```

The system retrains itself nightly. When a conjunction's TCA passes, the outcome becomes a labeled training example.

## Installation

### As a Package

```bash
pip install panacea-ssa                    # minimal: numpy, requests
pip install panacea-ssa[full]              # adds torch, sklearn, sgp4, etc.
```

```bash
# Predict on your CDMs
panacea predict --cdm-store data.jsonl -o predictions.json

# Train a model
panacea train --cdm-store data.jsonl -o model.json
```

### From Source

```bash
git clone https://github.com/DominicTanzillo/Panacea.git
cd Panacea
pip install -e ".[full]"
```

### Webapp (3D Globe)

```bash
cd webapp-react
npm install
npm run dev          # http://localhost:5173
```

## Daily Pipeline

Runs automatically at 00:00 UTC via GitHub Actions:

1. **Fetch CDMs** from Space-Track (18th Space Defense Squadron data)
2. **Fetch space weather** indices (F10.7, Kp, Ap) from NOAA SWPC
3. **Extract 23 features** per conjunction pair (trends, derivatives, metadata, space weather)
4. **Retrain** ensemble on all resolved pairs (TCA in past = known label)
5. **Predict** escalation probability for all active pairs
6. **Export** forecasts + uncertainty to webapp, deploy to GitHub Pages

Current scale: **16,000+ satellites** screened, **6M candidate pairs** filtered, **500 predictions** logged nightly.

## Repository Structure

```
Panacea/
├── src/
│   ├── cli.py                        # panacea predict / panacea train
│   ├── data/
│   │   ├── space_weather.py          # NOAA SWPC F10.7/Kp/Ap fetch + cache
│   │   ├── spacetrack_crossref.py    # Space-Track CDM API client
│   │   └── ...
│   ├── model/
│   │   ├── cdm_forecast.py           # Production logistic regression (23 features)
│   │   ├── cdm_sequence_model.py     # BiLSTM with transfer learning + focal loss
│   │   ├── conjunction_gnn.py        # Graph neural network on conjunction network
│   │   ├── cdm_autoregressive.py     # Autoregressive next-CDM forecaster
│   │   ├── conformal.py              # Split-conformal prediction intervals
│   │   └── ...
│   └── evaluation/
├── scripts/
│   ├── daily_predictions.py          # Daily automated pipeline (2000+ lines)
│   ├── tune_and_analyze.py           # Hyperparameter grid search + feature importance
│   ├── train_models.py               # Multi-model training + 5-fold CV
│   └── train_supplementary.py        # GNN, autoregressive, conformal
├── webapp-react/                     # Interactive 3D dashboard
│   └── src/components/
│       ├── Globe.tsx                  # Three.js Earth + 25K satellite dots
│       ├── CDMForecast.tsx            # Ensemble forecast visualization
│       ├── RiskDashboard.tsx          # Pipeline metrics + space weather
│       └── ConjunctionAlerts.tsx      # Real-time risk alerts
├── writeup/
│   ├── methodology.md                # Full methodology paper
│   └── paper.tex                     # LaTeX (NeurIPS format)
├── models/                           # Trained checkpoints (JSON + PyTorch)
├── pyproject.toml                    # pip install panacea-ssa
└── .github/workflows/
    ├── daily-predictions.yml         # 00:00 UTC daily
    └── weekly-finetune.yml           # Sundays 02:00 UTC
```

## Strategic Positioning

PANACEA does **not** compete with commercial SSA providers:

- **LeoLabs** generates CDMs and screens conjunctions (60% LEO market)
- **Slingshot Aerospace** trains on 6.4M proprietary CDMs
- **NASA CARA** concluded ML "has not shown promise" for risk assessment (AMOS 2025)

PANACEA predicts where those CDMs are headed — an **open-source ML layer on top of their data**. We reframe from Pc regression (which NASA CARA found intractable) to binary escalation prediction (which works). The conformal prediction provides the calibrated uncertainty bounds that NASA CARA has called out as missing from every ML approach they've evaluated.

## Novel Contributions

1. **Conformal prediction for conjunction assessment** — first application; no prior work exists
2. **Cross-dataset transfer learning** — Kelvins (103 features) → Space-Track (20 features) via feature dropout
3. **Conjunction network GNN** — graph over relational structure of conjunction events
4. **Space weather experiment** — controlled test of F10.7/Kp/Ap features, reported honestly
5. **Continuously-retrained open-source pipeline** — first for public CDM data

## Tech Stack

**ML:** PyTorch, scikit-learn, NumPy · **Data:** Space-Track API, NOAA SWPC, CelesTrak · **Frontend:** React 19, Three.js, Recharts, Vite · **Infra:** GitHub Actions, GitHub Pages

## Credits

- **Data:** [Space-Track.org](https://www.space-track.org/) (18th SDS), [ESA Kelvins Challenge](https://kelvins.esa.int/collision-avoidance-challenge/), [NOAA SWPC](https://www.swpc.noaa.gov/)
- **TLEs:** [CelesTrak](https://celestrak.org/) (Dr. T.S. Kelso)
- **Course:** AIPI 540 — Deep Learning Applications, Duke University

## License

MIT
