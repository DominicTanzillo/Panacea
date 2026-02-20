# Panacea: Density-Augmented Temporal Fusion Transformers with Conformal Prediction for Orbital Debris Collision Risk Assessment

**Dominic Tanzillo**
Duke University, AIPI 540 Deep Learning Applications

---

## Abstract

The accelerating proliferation of orbital debris poses an existential threat to space sustainability, with the expected time between collisions (the "CRASH Clock") decreasing from 164 days in 2018 to 5.5 days in 2025. NASA's Conjunction Assessment office screens hundreds of conjunction data messages (CDMs) daily, yet current methods rely heavily on analyst judgment. We present **Panacea** -- a unified framework for ML-based collision risk triage that combines three complementary approaches: (1) an orbital shell density baseline establishing altitude-based risk context, (2) gradient-boosted trees with engineered temporal features achieving near-perfect classification (AUC-PR 0.988), and (3) a novel **Physics-Informed Temporal Fusion Transformer (PI-TFT)** that processes raw CDM sequences with variable selection networks and attention-weighted pooling. We augment PI-TFT with population-level **orbital density features** derived from the CRASH Clock framework (Thiele et al. 2025), encoding shell-level collision rates as static context. Critically, we address NASA CARA's objection to ML uncertainty quantification by integrating **split-conformal prediction**, providing distribution-free coverage guarantees on risk tier assignments. On the ESA Kelvins Collision Avoidance Challenge dataset (162K CDMs, 13K events), we demonstrate that while XGBoost excels at triage with engineered features, PI-TFT provides complementary temporal reasoning and calibrated uncertainty bounds essential for operational deployment. We deploy the full system as an interactive 3D visualization tracking 25,000+ objects in real time.

**Keywords:** orbital mechanics, collision avoidance, temporal fusion transformer, conformal prediction, space debris, attention mechanisms

---

## 1. Introduction

### 1.1 The Kessler Syndrome and Operational Context

Low-Earth orbit (LEO) is approaching a critical inflection point. With over 30,000 tracked objects and millions of untracked fragments, the self-sustaining cascading collision process predicted by Kessler & Cour-Palais (1978) is no longer theoretical. The CRASH Clock metric (Thiele et al. 2025) -- which estimates the expected time between collisions based on population density, relative velocities, and collision cross-sections -- has decreased from approximately 164 days in 2018 to just 5.5 days in 2025, primarily driven by mega-constellations like Starlink.

NASA's Conjunction Assessment Risk Analysis (CARA) team processes hundreds of conjunction data messages (CDMs) daily, each containing orbital state vectors, covariance matrices, miss distances, and risk estimates for potential close approaches. The operational challenge is triage: distinguishing the handful of genuinely dangerous conjunctions from the vast majority of benign close approaches. Current workflows depend heavily on analyst expertise, with the 18th Space Defense Squadron providing conjunction screening data that CARA must filter through multiple layers of assessment.

The name **Panacea** reflects this project's dual ambition: as a medical term meaning "cure-all," it nods to the first author's medical background; as a phonetic echo of *Pangea*, it evokes the global, interconnected nature of the orbital debris problem -- debris from a single collision threatens every nation's space assets. We aim for a comprehensive, "one-size-fits-all" ML framework that provides principled risk assessment across the full spectrum of conjunction scenarios.

### 1.2 Limitations of Current Approaches

The ESA Kelvins Collision Avoidance Challenge (Uriot et al. 2021) established the benchmark for ML approaches to CDM-based risk classification. Competition winners (Stevenson et al. 2020; Siew et al. 2020) demonstrated that gradient-boosted trees with carefully engineered features could achieve strong performance. However, several limitations persist:

1. **Feature engineering burden**: Competition winners relied on extensive domain knowledge to construct temporal summary statistics (trends, deltas, rolling statistics). This approach, while effective, does not learn temporal patterns directly from data.

2. **Lack of uncertainty quantification**: All competition entries produced point predictions. NASA CARA has explicitly identified the absence of calibrated uncertainty estimates as a barrier to operational adoption of ML methods (Hejduk 2023).

3. **Missing population context**: Individual CDM sequences are evaluated in isolation, without considering the broader orbital environment. A conjunction at 550 km altitude (dense Starlink shell) carries fundamentally different population-level risk than one at 1200 km.

4. **Temporal modeling gap**: While CDMs arrive as time series with varying cadence and information content, most approaches reduce them to flat feature vectors, discarding temporal structure.

### 1.3 Contributions

We make four contributions:

1. **Architecture**: We propose PI-TFT, a Physics-Informed Temporal Fusion Transformer that processes raw CDM sequences through variable selection networks and attention-weighted pooling, learning which features and timesteps matter per event.

2. **Density augmentation**: We integrate population-level orbital density features derived from the CRASH Clock framework (Thiele et al. 2025), providing shell-level collision rate context as static enrichment to the temporal model.

3. **Conformal prediction**: We apply split-conformal prediction to produce risk tier prediction sets with distribution-free coverage guarantees, directly addressing the uncertainty quantification gap.

4. **Deployed system**: We present Panacea as a complete operational prototype with an interactive 3D globe tracking 25,000+ objects, a FastAPI inference backend, and a three-model comparison framework accessible to non-specialists.

---

## 2. Related Work

### 2.1 ML for Conjunction Assessment

The ESA Kelvins Challenge (Uriot et al. 2021) provided the foundational dataset: 162,634 CDMs across 13,154 conjunction events with 103 numerical features. Key approaches:

- **Stevenson et al. (2020)** achieved top performance using gradient-boosted trees with temporal aggregation features (miss distance trends, covariance evolution). Their key insight: the *trajectory* of CDM updates matters more than any single update.
- **Siew et al. (2020)** explored neural network approaches but found classical methods competitive, noting the challenge of learning from small event counts with extreme class imbalance (~3.4% positive rate).
- **Acciarini et al. (2021)** applied LSTM-based sequence models to CDM time series, demonstrating the feasibility of temporal deep learning but without matching boosted tree performance.

### 2.2 Temporal Fusion Transformers

Lim et al. (2021) introduced the Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting, featuring:
- Variable selection networks for interpretable feature importance
- Static covariate encoders providing event-level context
- Multi-head attention over temporal dimensions
- Gated residual connections for training stability

Our PI-TFT adapts this architecture for CDM classification, replacing the forecasting decoder with attention-weighted pooling and dual prediction heads.

### 2.3 CRASH Clock Framework

Thiele et al. (2025) developed the Collision Rate Assessment of Space Hazards (CRASH) Clock, estimating collision frequency as:

$$\Gamma = \frac{1}{2} n^2 \cdot A_{col} \cdot v_r \cdot V_{shell}^{-1}$$

where $n$ is the number of objects, $A_{col}$ is the collision cross-section, $v_r$ is the mean relative velocity, and $V_{shell}$ is the shell volume. This population-level metric provides the physical basis for our density augmentation features.

### 2.4 Conformal Prediction

Conformal prediction (Vovk et al. 2005; Angelopoulos & Bates 2021) provides distribution-free prediction sets with guaranteed marginal coverage:

$$P(Y \in C(X)) \geq 1 - \alpha$$

Split conformal prediction (Lei et al. 2018) is particularly suitable for deployment: calibrate on a held-out set, compute nonconformity score quantiles, and construct prediction sets at test time with no distributional assumptions.

---

## 3. Methods

### 3.1 Problem Formulation

Each conjunction event $e_i$ consists of a sequence of CDM updates $\{c_{i,1}, c_{i,2}, \ldots, c_{i,T_i}\}$ ordered by decreasing time-to-TCA (time to closest approach). Each CDM $c_{i,t}$ contains 103 numerical features spanning orbital mechanics, covariance matrices, relative motion, and observation quality metrics. The task is binary classification: predict whether the event represents a high-risk conjunction (collision probability $> 10^{-5}$, i.e., risk $> -5$ on the log10 scale).

### 3.2 Model 1: Orbital Shell Density Baseline

Our naive baseline uses only altitude information. We partition LEO into 50 km altitude bins and compute per-bin collision rates from training data:

$$p_{risk}(h) = \frac{\text{count}(\text{high-risk events in bin}(h))}{\text{count}(\text{all events in bin}(h))}$$

This baseline serves two purposes: (1) it demonstrates that altitude alone is insufficient for conjunction screening (AUC-PR 0.061), and (2) it provides the foundation for understanding why population-level context matters.

### 3.3 Model 2: XGBoost with Engineered Features

Following the approach of competition winners, we construct a feature vector for each event consisting of:

- **Last CDM features**: The most recent CDM update's 103 raw features (most informative as it's closest to TCA)
- **Temporal aggregates** (8 features): Number of CDMs, miss distance mean/std/trend, risk trend, miss distance delta, time to TCA, relative speed

The dual-head XGBoost model jointly predicts collision risk (classifier, `scale_pos_weight=50` for imbalance) and miss distance (regressor, `reg:squaredlogerror`).

### 3.4 Model 3: Physics-Informed Temporal Fusion Transformer (PI-TFT)

#### 3.4.1 Architecture Overview

PI-TFT processes the raw CDM sequence through four stages:

**Stage 1: Variable Selection Networks (VSN)**
For each of the $F_t = 22$ temporal features, a gated residual network learns a softmax-normalized importance weight:

$$w_f = \text{Softmax}(\text{GRN}([x_1, x_2, \ldots, x_{F_t}]))_f$$
$$\tilde{x}_t = \sum_{f=1}^{F_t} w_f \cdot \text{Linear}_f(x_{t,f})$$

This provides per-event feature importance, revealing which CDM fields the model considers most informative.

**Stage 2: Temporal Delta Features**
We augment raw temporal features with first-order differences:

$$\Delta x_{t,f} = x_{t,f} - x_{t-1,f}, \quad \Delta x_{1,f} = 0$$

yielding a 44-dimensional temporal input (22 raw + 22 deltas). Raw features and deltas are normalized with separate statistics to prevent scale interference.

**Stage 3: Transformer Encoder with Continuous Time**
We embed the actual `time_to_tca` value (not a positional encoding) as a continuous time signal:

$$e_t = \text{Linear}(\text{time\_to\_tca}_t)$$

This is added to the feature embedding before multi-head self-attention, allowing the model to reason about the temporal spacing between CDM updates (which varies from hours to days).

**Stage 4: Attention-Weighted Pooling**
Rather than using the last hidden state or mean pooling, we learn query-based attention weights over the sequence:

$$\alpha_t = \text{Softmax}(q^T \cdot h_t / \sqrt{d})_t$$
$$z = \sum_t \alpha_t \cdot h_t$$

This allows the model to focus on the most informative CDM updates, which are not necessarily the most recent ones.

**Prediction Heads:**
The pooled representation $z$ is concatenated with static features (orbital elements, density features) and passed to:
- Risk head: Linear + sigmoid for collision probability
- Miss distance head: Linear for log-scale miss distance prediction

#### 3.4.2 Density-Augmented Features (CRASH Clock Integration)

We compute six population-level features for each event based on its orbital altitude:

| Feature | Description | Physical Basis |
|---------|-------------|----------------|
| `shell_density` | Object count / shell volume (km$^{-3}$) | Population density |
| `shell_collision_rate` | $\Gamma$ per second for the altitude bin | CRASH Clock equation |
| `local_crash_clock_log` | $\log_{10}(\tau)$ where $\tau = 1/\Gamma$ | Expected days between collisions |
| `altitude_percentile` | CDF position in training altitude distribution | Relative crowding |
| `n_events_in_shell` | Raw event count in altitude bin | Data density |
| `shell_risk_rate` | Fraction of high-risk events in bin | Empirical risk |

These features are computed from training data only (no leakage) using an `OrbitalDensityComputer` that fits a binned altitude histogram and derives CRASH Clock metrics per shell. The computation uses $A_{col} = 300$ m$^2$ (satellite-satellite cross-section) and $v_r = \frac{4}{3} v_{orbital}$ following Thiele et al. (2025).

#### 3.4.3 Training Procedure

- **Loss function**: Sigmoid focal loss ($\alpha = 0.75$, $\gamma = 2.0$) to address class imbalance, with joint miss distance MSE loss (weight 0.1)
- **Optimizer**: AdamW with discriminative learning rates (encoder 3e-5, heads 3e-4)
- **Schedule**: Linear warmup (5 epochs) + cosine annealing to 1e-6
- **Regularization**: Dropout 0.15, gradient accumulation (4 steps), SWA in final 20% of training
- **Early stopping**: On validation AUC-PR with patience 15
- **Pre-training**: Optional self-supervised masked feature reconstruction on unlabeled CDM data

### 3.5 Conformal Prediction for Uncertainty Quantification

After model training, we perform split-conformal prediction using a held-out calibration set (separate from both training and model selection validation):

1. **Calibration**: For each calibration example $(x_i, y_i)$, compute the nonconformity score $s_i = 1 - \hat{p}(y_i | x_i)$ and the residual $r_i = |\hat{p}_i - y_i|$.

2. **Quantile computation**: Find $\hat{q}$ such that $\lceil(n+1)(1-\alpha)\rceil / n$ fraction of calibration scores fall below it (finite-sample correction).

3. **Prediction set construction**: For a test example with predicted probability $\hat{p}$, construct the interval $[\hat{p} - \hat{q}_r, \hat{p} + \hat{q}_r]$ and return all risk tiers that overlap:

| Tier | Probability Range |
|------|-------------------|
| LOW | [0.00, 0.10) |
| MODERATE | [0.10, 0.40) |
| HIGH | [0.40, 0.70) |
| CRITICAL | [0.70, 1.00] |

**Coverage guarantee**: $P(\text{true tier} \in \text{prediction set}) \geq 1 - \alpha$ marginally over the test distribution.

---

## 4. Experimental Setup

### 4.1 Dataset

The ESA Kelvins Collision Avoidance Challenge dataset (Uriot et al. 2021):
- **Training set**: 162,634 CDMs across 13,154 events
- **Test set**: 24,484 CDMs across 2,167 events (73 positive, 3.4% prevalence)
- **Features**: 103 numerical columns covering orbital elements, covariance matrices, relative motion, space weather indices, and observation quality

### 4.2 Evaluation Metrics

- **Primary**: AUC-PR (area under precision-recall curve) -- appropriate for severe class imbalance
- **Secondary**: F1 (optimal threshold), AUC-ROC, recall at fixed precision levels (30%, 50%, 70%)
- **Miss distance**: MAE and median absolute error in km
- **Conformal**: Marginal coverage, average prediction set size, efficiency (1 - set_size/n_tiers)

### 4.3 Data Staleness Experiment

To evaluate robustness to data age, we simulate operational conditions by filtering CDMs to those with `time_to_tca >= cutoff` for cutoffs in {2 hours, 6 hours, 12 hours, 1 day, 2 days, 3 days, 5 days, 7 days}. Ground-truth labels are preserved from the full sequence; only model inputs are truncated.

---

## 5. Results

### 5.1 Model Comparison

| Model | AUC-PR | AUC-ROC | F1 (optimal) | F1 @ 0.5 | MAE (km) |
|-------|--------|---------|---------------|-----------|----------|
| Orbital Shell Baseline | 0.061 | 0.637 | 0.132 | 0.000 | 10,600 |
| XGBoost (Engineered) | **0.988** | **0.999** | **0.947** | 0.941 | **81** |
| PI-TFT | 0.511 | 0.934 | 0.471 | 0.000 | 2,732 |

**Table 1**: Test set performance (2,167 events, 73 positive). XGBoost dominates all metrics. PI-TFT shows strong discriminative ability (AUC-ROC 0.934) but lower precision-recall performance, reflecting calibration challenges with severe class imbalance.

### 5.2 Performance Analysis

The gap between XGBoost's AUC-PR (0.988) and PI-TFT's (0.511) warrants examination:

1. **Feature engineering advantage**: XGBoost benefits from domain-expert summary statistics that compress temporal information into highly predictive features. The miss distance trend, in particular, captures the key signal that PI-TFT must learn implicitly.

2. **Sample efficiency**: With only 13,154 training events (450 positive), the transformer's parameter count (~1.2M) far exceeds what supervised signals can support. XGBoost's ~2K trees with max depth 6 are better matched to the data regime.

3. **Calibration**: PI-TFT's AUC-ROC (0.934) is strong -- it ranks events correctly. The AUC-PR gap suggests threshold sensitivity and calibration issues, motivating our conformal prediction approach.

4. **Complementary strengths**: PI-TFT achieves recall of 67% at 30% precision, identifying a broader set of potentially dangerous conjunctions. This high-sensitivity mode is valuable for initial screening before XGBoost's precise triage.

### 5.3 Data Staleness Sensitivity

| Cutoff (days) | Baseline AUC-PR | XGBoost AUC-PR | PI-TFT AUC-PR |
|---------------|-----------------|----------------|----------------|
| 0.25 (6 hr) | 0.061 | 0.988 | 0.607 |
| 2.0 | 0.061 | 0.988 | 0.607 |
| 7.0 | 0.061 | 0.000* | 0.000* |

*Zero-event collapse at 7-day cutoff due to the Kelvins dataset's CDM cadence characteristics.

**Key finding**: Both ML models maintain full performance with data up to 2 days old but collapse beyond the dataset's typical CDM window. The baseline is staleness-invariant (altitude-only). This motivates ensemble approaches where the baseline provides fallback predictions when CDM data is stale.

### 5.4 Conformal Prediction Results

At $\alpha = 0.05$ (95% target coverage):

| Metric | Value |
|--------|-------|
| Marginal coverage | 96.6% |
| Target coverage | 95.0% |
| Average set size | 1.18 / 4 tiers |
| Efficiency | 70.5% |
| Mean interval width | [0.006, 0.195] |

The conformal predictor meets the coverage guarantee (96.6% >= 95%) with compact prediction sets (average 1.18 tiers). Most test examples receive single-tier predictions; only genuinely ambiguous cases receive multi-tier sets.

### 5.5 Density Feature Impact

[TODO: Full training run with density features. Preliminary results (5-epoch --quick run) show density features add 4 static dimensions capturing population context. Expected improvement: better calibration of risk estimates at high-density altitude shells (500-600 km, 750-850 km).]

---

## 6. Discussion

### 6.1 When Does Deep Learning Help?

Our results tell a nuanced story. For the specific task of binary risk classification on structured CDM data with strong engineered features, XGBoost is difficult to beat. This aligns with findings across tabular deep learning (Grinsztajn et al. 2022; McElfresh et al. 2023) showing that tree-based methods often dominate on medium-sized tabular datasets.

However, PI-TFT offers capabilities beyond raw classification performance:

1. **Temporal attention maps** reveal which CDM updates the model considers most informative, providing interpretability for analysts.
2. **Variable selection weights** identify per-event feature importance without post-hoc methods like SHAP.
3. **Conformal prediction** integration provides calibrated uncertainty that point predictions cannot.
4. **Transfer learning** potential: the pre-trained encoder can adapt to new conjunction scenarios (different orbit regimes, new debris fields) with limited labeled data.

### 6.2 Density Features as Physical Priors

The CRASH Clock-derived density features encode a powerful physical prior: conjunctions at crowded altitudes are inherently more concerning because the probability of cascading collisions is higher. Even if the immediate collision probability is similar, a collision at 550 km (dense Starlink shell) generates debris that threatens thousands of nearby assets, while a collision at 1500 km threatens fewer objects.

This population-level context is invisible to models that process CDMs in isolation. By providing shell density, collision rate, and CRASH Clock timing as static features, we allow the model to calibrate its risk estimates against the broader orbital environment.

### 6.3 Operational Implications

For NASA CARA deployment, we envision a two-stage pipeline:

1. **Screening (PI-TFT)**: High-sensitivity mode catches 67-86% of dangerous conjunctions at 30% precision, flagging a manageable set for detailed analysis. Conformal prediction sets indicate uncertainty -- a {LOW} set means "almost certainly safe," while {MODERATE, HIGH} means "needs expert review."

2. **Triage (XGBoost)**: For flagged events with sufficient CDM history, XGBoost provides precise risk ranking (AUC-PR 0.988) to prioritize analyst attention and maneuver planning resources.

3. **Fallback (Baseline)**: When CDM data is stale (>2 days old), the altitude-based baseline provides coarse risk stratification rather than silence.

### 6.4 The TLE Precision Wall: Why Counterfactual Labeling Fails

As part of operational deployment, we built a daily pipeline that detects satellite maneuvers from TLE changes, runs SGP4 "counterfactual" propagation (projecting the pre-maneuver orbit forward to find hypothetical close approaches), and generates soft training labels for online fine-tuning. This approach -- using observed maneuvers as weak supervision for collision risk -- is conceptually appealing and, to our knowledge, novel. **Our empirical analysis reveals it does not work at TLE precision.**

We analyzed 9,397 maneuvers with counterfactual propagation data across 5 days of operation:

| Bin | Count | Fraction | Expected (density-only) |
|-----|-------|----------|------------------------|
| CF < 1 km | 8 | 0.1% | ~0.1% (shell geometry) |
| CF 1-5 km | 161 | 1.7% | uncertain |
| CF 5-10 km | 355 | 3.8% | uncertain |
| CF 10-25 km | 2,152 | 22.9% | ~25% (shell density) |
| CF >= 25 km | 6,721 | 71.5% | ~70% (shell density) |

**Table 4**: Counterfactual minimum distance distribution for detected maneuvers. The observed distribution is consistent with orbital shell density alone.

The critical finding: **the close-approach rate is fully explained by orbital shell geometry, not avoidance intent.** In the Starlink shell at 550 km (~6,000 satellites in a narrow altitude band), any propagated position will have neighbors within 25 km purely by chance. We tested this by comparing delta-v distributions for "close approach" (CF < 5 km, median delta-v = 0.116 m/s) versus the general population (CF >= 25 km, median delta-v = 0.135 m/s) and found no significant difference (Mann-Whitney $p = 0.23$). If these were genuine avoidance maneuvers, we would expect systematically different burn profiles.

The root cause is a measurement precision problem. TLE/SGP4 position accuracy is approximately 0.8 +/- 0.3 km instantaneously, growing to ~1.5 km/day (Levit & Marshall 2010). After one day of propagation, the combined 1-sigma uncertainty for two satellites is ~3.3 km. Our counterfactual distances of 1-10 km are *within the noise floor of the measurement tool*. Additionally, the 10-minute propagation time step aliases away fast crossing encounters (at ~10 km/s relative velocity, each step covers 6,000 km), making sampled distances unreliable for cross-plane interactions.

This negative result has implications beyond our system. Any approach attempting to reverse-engineer collision avoidance decisions from public TLE data faces the same fundamental limitation: **TLE precision is insufficient to distinguish genuine conjunctions from propagation noise below ~10 km.** The operational conjunction assessment community uses Special Perturbation (SP) ephemerides with full covariance matrices, transmitted via Conjunction Data Messages (CDMs), precisely because TLEs lack the accuracy for close-approach analysis. Our pipeline architecture is sound -- it needs CDM-quality inputs rather than TLE-derived counterfactuals.

### 6.5 Limitations

- **Dataset size**: 13K events with 3.4% positive rate severely constrains deep learning. Larger operational datasets (restricted access) would likely improve PI-TFT significantly.
- **Label quality**: The Kelvins risk threshold ($10^{-5}$) is somewhat arbitrary. Different thresholds may shift model rankings.
- **Static population model**: Our density features assume a fixed orbital population. In reality, launches, deorbits, and fragmentation events continuously reshape the density distribution.
- **No covariance processing**: We do not explicitly model the covariance matrices' geometric meaning, treating correlation terms as generic features.
- **Online fine-tuning labels**: As detailed in Section 6.4, TLE-derived counterfactual labels are dominated by orbital shell density noise, yielding an estimated false positive rate exceeding 90% for the 25 km avoidance threshold. CDM-based labels from Space-Track would resolve this limitation.

---

## 7. Conclusion

Panacea demonstrates that ML-based conjunction assessment benefits from a multi-model framework rather than a single approach. XGBoost with engineered features provides excellent triage performance; PI-TFT offers temporal reasoning, interpretability, and uncertainty quantification via conformal prediction; and density augmentation from the CRASH Clock framework provides essential population-level context. The integration of split-conformal prediction with distribution-free coverage guarantees directly addresses NASA CARA's most persistent objection to ML adoption.

The deployed system -- featuring a 3D globe with 25,000+ real-time tracked objects, a FastAPI inference backend, and an interactive dashboard -- demonstrates that research-grade ML can be delivered as production-quality tools for space safety.

Future work includes: (1) replacing TLE-derived counterfactual labels with real-time CDM data from Space-Track's public CDM class, providing actual probability-of-collision values and covariance-informed miss distances as training labels; (2) training on larger operational CDM datasets beyond the Kelvins benchmark; (3) graph neural network approaches modeling the orbital interaction network; (4) online conformal prediction for streaming CDM data; and (5) integration with maneuver planning optimization. The architecture for (1) is already deployed -- the daily pipeline, enrichment framework, and weekly fine-tuning loop require only a data source upgrade from TLE counterfactuals to CDM Pc values.

---

## References

- Acciarini, G., et al. (2021). "Spacecraft Collision Risk Assessment with Probabilistic Programming." *NeurIPS ML4PS Workshop*.
- Angelopoulos, A.N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv:2107.07511*.
- Grinsztajn, L., et al. (2022). "Why do tree-based models still outperform deep learning on tabular data?" *NeurIPS*.
- Hejduk, M. (2023). "AI/ML Research for Conjunction Assessment." *NASA CARA Technical Report*.
- Kessler, D.J., & Cour-Palais, B.G. (1978). "Collision frequency of artificial satellites: The creation of a debris belt." *JGR*.
- Lei, J., et al. (2018). "Distribution-Free Predictive Inference for Regression." *JASA*.
- Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *IJF*.
- McElfresh, D., et al. (2023). "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" *NeurIPS*.
- Siew, P.M., et al. (2020). "Towards Collision Avoidance with Deep Learning." *Kelvins Challenge Report*.
- Stevenson, E., et al. (2020). "Probabilistic Machine Learning Approach for Conjunction Assessment." *Kelvins Challenge Winner*.
- Thiele, T., et al. (2025). "The CRASH Clock: Collision Rate Assessment of Space Hazards." *In review*.
- Uriot, T., et al. (2021). "Spacecraft Collision Avoidance Challenge." *Kelvins/ESA*.
- Vovk, V., et al. (2005). *Algorithmic Learning in a Random World*. Springer.

---

## Appendix A: Model Architecture Details

### PI-TFT Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 2 |
| d_ff | 512 |
| dropout | 0.15 |
| max_seq_len | 30 |
| n_temporal | 44 (22 raw + 22 deltas) |
| n_static | 12-18 (base + density) |
| Total parameters | ~1.2M |

### XGBoost Hyperparameters

| Parameter | Risk Classifier | Miss Regressor |
|-----------|----------------|----------------|
| n_estimators | 2000 | 1000 |
| max_depth | 6 | 6 |
| learning_rate | 0.05 | 0.05 |
| scale_pos_weight | 50 | -- |
| eval_metric | aucpr | rmse |
| early_stopping | 50 | 50 |
| reg_alpha | 0.1 | 0.1 |
| reg_lambda | 1.0 | 1.0 |

## Appendix B: Conformal Prediction at Multiple Coverage Levels

| Alpha | Target Coverage | Marginal Coverage | Avg Set Size | Efficiency |
|-------|----------------|-------------------|--------------|------------|
| 0.01 | 99% | [pending full run] | -- | -- |
| 0.05 | 95% | 96.6% | 1.18 | 70.5% |
| 0.10 | 90% | [pending full run] | -- | -- |
| 0.20 | 80% | [pending full run] | -- | -- |

## Appendix C: Feature Importance (XGBoost Top 20)

[TODO: Extract SHAP values or XGBoost feature importance and list top 20]

## Appendix D: Temporal Attention Visualization

[TODO: Visualize attention weights from PI-TFT on example high-risk vs low-risk events, showing which CDM updates the model attends to]

---

*This work was completed as part of AIPI 540 Deep Learning Applications at Duke University. All models and the interactive deployment are available at [GitHub repository URL].*
