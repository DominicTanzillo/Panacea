# NASA CARA AI/ML Compendium for Satellite Collision Avoidance

**Authors**: Dr. Alinda K. Mashiku (NASA GSFC), Lauri K. Newman (NASA HQ)
**Source**: NASA Conjunction Assessment Risk Analysis (CARA) program

## Key Questions Investigated
1. Can risk assessment be performed with increased certainty using early CDM data?
2. Can rapid/early decision-making (collision avoidance) be performed using early CDMs?

## Critical Findings — AI/ML NOT Promising for Risk Assessment

> "The AI/ML solutions undertaken to date have not shown promise for applicability in risk assessment."

### Why ML Struggles Here
1. **No truth data**: No dataset of actual collisions exists for training. Only mitigated close approaches.
2. **Non-deterministic factors**: Each conjunction is unique; miss distance is the only common collision factor.
3. **Need to simulate/redefine risk**: Must augment data since real collisions don't exist in training sets.
4. **Explainability required**: Operational implementation needs explainable outcomes, not black-box models.
5. **Sparse CDM sequences**: Most events have <10 CDMs; with 5km radial filter, most have <4.
6. **Late identification**: Events can be identified only 7 days before TCA (propagation accuracy limit),
   or as late as 8 hours before TCA due to unexpected atmospheric drag from solar storms.

## Approaches Investigated
1. **Statistical & Information Theory Parameters**: Augmented statistical parameters for Pc computation
2. **Fuzzy Inference System**: Weighted parameter assignments using multiple FIS models
3. **Unsupervised ML Clustering**: Aid FIS weight inference via hierarchical clustering (Steinbach et al.)
4. **Deep Neural Networks**: Characterize stochastic nature of evolving CDMs
5. **LSTM Networks**: Time-series prediction using CDM sequences of varying lengths

## Operational Context
- Standard approach: Monte Carlo simulation of state space (most accurate but computationally infeasible for routine catalog scans)
- Current operational method: Probability of collision (Pc) using 2D projected uncertainty volumes
- Known limitation: 2D Pc assumptions break down for low relative velocity encounters
- Operators analyze: Pc threshold, miss distance, tracking history, orbit determination quality

## Data Characteristics (2015-2018)
- 200,000+ CDMs
- Most events have <10 CDMs
- CDMs are time-varying as risk evolves toward TCA
- Variable-length sequences per event

## Key References
- Robertson & Mashiku (2021): Early information parameter-set analysis for close approaches using ML
- Mashiku et al. (2019): Predicting close approaches using statistical parameters with AI
- Newman et al. (2009): Justification for close approach prediction and risk assessment
- Burton et al. (2018): Assessing measures to reliably predict collisions under uncertainty

## Implications for Panacea Project
1. Our ESA Kelvins dataset has similar structure: variable-length CDM sequences, ~13K events
2. Binary risk classification (risk > -5 threshold) is a proxy since no real collisions in the dataset
3. LSTM/Transformer approaches for CDM sequences align with our PI-TFT architecture
4. The key gap: early CDM prediction and uncertainty quantification
5. Staleness experiment directly addresses the "early CDM" question
6. Our 3-model comparison (Baseline → XGBoost → PI-TFT) shows the progression
   from simple heuristics to learned temporal patterns
