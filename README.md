# Panacea

**AI-powered satellite collision avoidance and orbital debris visualization**

[Live Demo](https://tanzillo.me/Panacea/) | [Models on HuggingFace](https://huggingface.co/DTanzillo/panacea-models)

AIPI 540 Final Project - Duke University

---

## What It Does

Panacea predicts orbital conjunction events (potential satellite collisions) using machine learning on ESA's Conjunction Data Messages, and visualizes the full tracked-object catalog on an interactive 3D globe with real-time TLE data from CelesTrak.

**Key capabilities:**
- 3 ML models for collision risk scoring (Baseline, XGBoost, Physics-Informed Transformer)
- Live daily screening of 14,000+ active satellites via GitHub Actions
- Automated weekly model fine-tuning using Starlink maneuver detections as proxy labels
- Interactive 3D Earth with all tracked orbital objects, conjunction alerts, and model comparison dashboard

## Model Performance

Trained and evaluated on the [ESA Kelvins Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/) dataset (13K events, 162K CDMs).

| Model | AUC-PR | F1 | Description |
|-------|--------|-----|-------------|
| Orbital Shell Baseline | 0.061 | 0.132 | Altitude-binned historical collision rates |
| **XGBoost** | **0.988** | **0.947** | Gradient-boosted trees on 112 engineered CDM features |
| PI-TFT | 0.511 | 0.519 | Physics-Informed Temporal Fusion Transformer on CDM sequences |

The PI-TFT improves automatically over time via weekly fine-tuning on real-world maneuver outcomes.

### Staleness Experiment

How quickly does prediction accuracy degrade as TLE data ages?

| Model | 2 days | 3 days | 6 days |
|-------|--------|--------|--------|
| XGBoost | 0.988 | 0.711 | 0.322 |
| PI-TFT | 0.511 | 0.313 | 0.184 |
| Baseline | 0.061 | 0.061 | 0.061 |

XGBoost has a sharp performance cliff at 3 days, demonstrating that timely data is critical for operational conjunction assessment.

## Architecture

```
CelesTrak TLEs ──> Daily Screening ──> Risk Scoring ──> Firebase + JSONL
                    (GitHub Actions)     (Baseline)       (Outcomes)
                                                              │
                                                              v
ESA Kelvins CDMs ──> Feature Eng ──> XGBoost ──────────> Model Comparison
                         │                                    │
                         v                                    v
                   CDM Sequences ──> PI-TFT ──> Weekly ──> HuggingFace
                                    (Transformer)  Fine-Tune   Hub
                                         │
GitHub Pages ◄── React + Three.js ◄──── FastAPI ◄── All 3 Models
 (3D Globe)       (Visualization)      (Inference)
```

## Repository Structure

```
Panacea/
├── src/
│   ├── data/
│   │   ├── cdm_loader.py          # ESA Kelvins dataset loading + feature engineering
│   │   ├── sequence_builder.py    # CDM sequence dataset for PI-TFT (temporal + delta features)
│   │   ├── density_features.py    # Space debris density computation
│   │   ├── firebase_client.py     # Firestore prediction/outcome logging
│   │   └── maneuver_detector.py   # Kelecy SMA-change maneuver detection
│   ├── model/
│   │   ├── baseline.py            # Orbital shell baseline (altitude-binned rates)
│   │   ├── classical.py           # XGBoost conjunction model
│   │   ├── deep.py                # Physics-Informed TFT + focal loss + physics loss
│   │   ├── pretrain.py            # Self-supervised TLE encoder pre-training
│   │   └── triage.py              # Urgency tier classifier (LOW/MODERATE/HIGH)
│   └── evaluation/
│       ├── metrics.py             # AUC-PR, F1, calibration, miss-distance metrics
│       ├── staleness.py           # TLE staleness sensitivity experiment
│       └── conformal.py           # Conformal prediction intervals
├── app/
│   └── main.py                    # FastAPI inference server (5 endpoints)
├── scripts/
│   ├── train.py                   # Train baseline + XGBoost
│   ├── train_deep.py              # Train PI-TFT
│   ├── daily_predictions.py       # Daily CelesTrak screening pipeline
│   ├── weekly_finetune.py         # Weekly PI-TFT fine-tuning from outcomes
│   └── run_experiment.py          # Staleness experiment runner
├── webapp-react/                  # 3D visualization frontend
│   └── src/
│       ├── App.tsx                # Main app with globe, panels, dashboard
│       └── components/
│           ├── Globe.tsx          # Three.js Earth with satellite orbits
│           ├── ConjunctionAlerts.tsx  # Real-time risk alerts panel
│           ├── RiskDashboard.tsx  # Model comparison + experiment charts
│           ├── SearchFilter.tsx   # Satellite search by name/NORAD ID
│           └── AboutPage.tsx      # Project info modal
├── models/                        # Trained model artifacts
├── results/                       # Experiment results (JSON)
└── .github/workflows/
    ├── daily-predictions.yml      # Cron: 00:00 UTC daily
    ├── weekly-finetune.yml        # Cron: 02:00 UTC Sundays
    ├── upload-models.yml          # Manual: push models to HuggingFace
    └── deploy-webapp.yml          # Deploy React app to GitHub Pages
```

## Operational Feedback Loop

Panacea runs a continuous improvement cycle without human intervention:

1. **Daily (00:00 UTC)** - Fetch TLEs, screen 14K+ satellites, score pairs, log predictions to Firebase
2. **Daily +24h** - Compare yesterday's predictions against detected maneuvers (Kelecy SMA-change)
3. **Weekly (Sunday 02:00 UTC)** - Pull accumulated outcomes, fine-tune PI-TFT, upload improved model to HuggingFace
4. **Ongoing** - PI-TFT improves as more maneuver labels accumulate (current AUC-PR: 0.511, improving)

## Quick Start

### Frontend (3D Globe)

```bash
cd webapp-react
npm install
npm run dev          # http://localhost:5173
```

### Backend (API Server)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload    # http://localhost:8000
```

### Training (requires ESA Kelvins data in `data/cdm/`)

```bash
python scripts/train.py          # Baseline + XGBoost
python scripts/train_deep.py     # PI-TFT
python scripts/run_experiment.py # Staleness experiment
```

## Tech Stack

**ML:** PyTorch, XGBoost, scikit-learn, NumPy, SciPy

**Backend:** FastAPI, Firebase Firestore, HuggingFace Hub

**Frontend:** React 19, Three.js, Recharts, TailwindCSS, Vite

**Infrastructure:** GitHub Actions (daily/weekly cron), GitHub Pages, CelesTrak API

## Credits

- **Dataset:** [ESA Kelvins Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/)
- **TLE Data:** [CelesTrak](https://celestrak.org/) (Dr. T.S. Kelso)
- **Maneuver Detection:** Kelecy semi-major axis change method
- **Course:** AIPI 540 — Deep Learning Applications, Duke University

## License

MIT
