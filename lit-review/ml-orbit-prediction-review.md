# ML for Satellite Orbit Prediction & Collision Avoidance — Literature Review

## 1. Standard Approach: SGP4/SDP4

- Analytical orbit propagator consuming TLEs; US Space Force standard
- **Accuracy**: ~1 km at epoch, growing 1-3 km/day; in-track errors >25 km after days
- **Error sources**: TLE fitting, atmospheric drag uncertainty (dominant for LEO), simplified perturbations, no maneuver modeling, no covariance output
- **Key paper**: Acciarini, Baydin, Izzo (2024) "Closing the Gap Between SGP4 and High-Precision Propagation via Differentiable Programming" — dSGP4 reimplements SGP4 in PyTorch

## 2. Where ML Adds Value

### 2.1 SGP4 Error Correction (Most Mature)
- Neural networks learn residual errors between SGP4 and high-precision ephemerides
- 75%+ improvement in along-track, 90%+ in cross-track/radial over 14-day windows
- **dSGP4** (ESA, open-source): differentiable SGP4 enables end-to-end gradient learning

### 2.2 Thermospheric Density Prediction
- Dominant LEO error source; ML reduces MAPE from 40-60% to ~20% (Acciarini et al. 2024)
- **Karman** package: shared benchmarking framework
- Temporal Fusion Transformers for ionospheric parameter forecasting (Perez et al.)

### 2.3 CDM-Based Risk Prediction (Panacea's Focus)
- ESA Kelvins Challenge (2019): 96 teams, 862 submissions
- **Winner**: LightGBM with temporal feature engineering (not neural networks)
- All top-tier solutions used gradient boosted trees
- Feature engineering quality > model architecture complexity
- LSTM/GRU approaches: Sanchez et al. (2021) report 97% go/no-go accuracy
- **Kessler library** (ESA): Bayesian LSTM for CDM prediction with uncertainty

### 2.4 Uncertainty Quantification
- ML uncertainty estimates achieve >1000x speedup over Monte Carlo
- Bayesian LSTM (Kessler lib), MC-dropout, deep ensembles — not yet operational

### 2.5 Mega-Constellation Management
- Starlink: ~9,400 satellites, 300,000+ avoidance maneuvers in 2025
- Attention networks for maneuver forecasting; GNNs for spatial interactions

## 3. NASA CARA Findings (Mashiku & Newman)

> "AI/ML solutions undertaken to date have not shown promise for applicability in risk assessment"

- **Root cause**: No truth data (no actual collisions), sparse CDM sequences (<10 per event), non-deterministic conjunctions, need for explainability
- Tried: Fuzzy Inference, unsupervised clustering, DNNs, LSTMs
- None outperformed operational Pc calculations

## 4. ESA Kelvins Challenge — Detailed Results

| Approach | Placement | Notes |
|----------|-----------|-------|
| LightGBM + feature engineering | 1st (sesc) | Ensemble, 25% random feature subsets, shallow trees |
| XGBoost + temporal features | Top tier | 200-500 engineered features from 103 columns |
| Neural networks (LSTM, etc.) | Not in top tier | Data-limited; feature engineering dominated |

## 5. Identified Gaps

| Gap | Description | Opportunity |
|-----|-------------|-------------|
| **Dataset staleness** | Kelvins (2019) is only public CDM dataset; catalog has grown 5x since | Updated datasets needed |
| **Self-supervised pre-training** | No work on SSL for CDM transformers | **Panacea's SSL branch is novel** |
| **Physics-informed CDM models** | PINNs for orbit determination only, not CDMs | **Panacea's physics loss is novel** |
| **Uncertainty quantification** | Not integrated into operational pipelines | Conformal prediction, deep ensembles |
| **Multi-source fusion** | No CDM + radar + space weather fusion | Neural networks' natural advantage |
| **Cross-operator transfer** | ESA CDMs ≠ NASA CDMs ≠ SpaceX decisions | Domain adaptation needed |
| **Streaming CDM processing** | All approaches batch, not streaming | Real-time incremental updates |
| **Explainability** | Research optimizes accuracy, ops needs trust | Temporal attention maps |

## 6. Operational Systems

### NASA CARA (Goddard Space Flight Center)
- Screens ~70 NASA missions
- Pipeline: 18th SDS screening → CDMs via SFTP → CAS automated triage → Human OSA analysis → Maneuver planning
- Pc threshold typically 1e-4 for LEO

### ESA ESOC Space Debris Office
- Sentinel satellites: avoidance maneuver every ~3 months
- Developing autonomous CA system
- Kessler library for ML research

### SpaceX Autonomous CA
- Multi-tier neural network predicting global traffic 12 hours ahead
- Graph-based optimizer for dynamic beam/crosslink reassignment
- Autonomous maneuver planning without human-in-the-loop

## 7. Relevance to Panacea

1. **XGBoost dominance (AUC-PR 0.978)** — validated by Kelvins results
2. **PI-TFT gap** — expected; documented across multiple studies
3. **Self-supervised pre-training** — genuinely novel; no published work on SSL for CDM transformers
4. **Physics-informed loss** — novel application to CDM-level assessment
5. **Staleness experiment** — directly addresses NASA CARA's "early CDM" question
6. **3-model comparison** — follows recommended evaluation methodology

## Key References

- Acciarini et al. (2024) "dSGP4" — Acta Astronautica, [arXiv:2402.04830](https://arxiv.org/abs/2402.04830)
- Uriot, Izzo et al. (2022) "Spacecraft Collision Avoidance Challenge" — Astrodynamics
- Mashiku & Newman (2025) "NASA CARA AI/ML Compendium" — [NTRS](https://ntrs.nasa.gov/citations/20250002065)
- Pinto et al. (2021) "Kessler Library" — [GitHub](https://github.com/kesslerlib/kessler)
- Sanchez et al. (2021) "Predicting Risk of Satellite Collisions Using ML"
- Caldas & Soares (2024) "ML in Orbit Estimation: A Survey" — [arXiv:2207.08993](https://arxiv.org/abs/2207.08993)
- Catulo et al. (2023) "Bayesian ML for Collision Probability"
- Varey et al. (2024) "PINNs for Satellite State Estimation"
- Matney (2017) "Algorithms for the Computation of Debris Risk" — NASA/JSC
