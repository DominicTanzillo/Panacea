import { PanaceaLogo } from './Header';

interface AboutPageProps {
  visible: boolean;
  onClose: () => void;
}

export function AboutPage({ visible, onClose }: AboutPageProps) {
  if (!visible) return null;

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-2xl max-h-[85vh] mx-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)]/98 backdrop-blur-md shadow-2xl overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <PanaceaLogo />
            <h2 className="text-lg font-bold tracking-tight">About PANACEA</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-2)] transition-colors text-lg"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 text-sm leading-relaxed tracking-normal">
          {/* Overview */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-2">What is PANACEA?</h3>
            <p className="text-[var(--color-text-muted)]">
              <strong className="text-[var(--color-text)] font-semibold">PANACEA</strong> &mdash;{' '}
              <strong className="text-[var(--color-accent)] font-semibold">P</strong>redictive{' '}
              <strong className="text-[var(--color-accent)] font-semibold">A</strong>ssessment{' '}
              <strong className="text-[var(--color-accent)] font-semibold">N</strong>etwork for{' '}
              <strong className="text-[var(--color-accent)] font-semibold">A</strong>utomated{' '}
              <strong className="text-[var(--color-accent)] font-semibold">C</strong>onjunction{' '}
              <strong className="text-[var(--color-accent)] font-semibold">E</strong>valuation &amp;{' '}
              <strong className="text-[var(--color-accent)] font-semibold">A</strong>voidance &mdash;{' '}
              is an orbital debris collision prediction system that combines real-time
              satellite tracking with machine learning to assess collision risk in
              low-Earth orbit. The CRASH Clock metric estimates that collisions now
              occur roughly every 5.5 days in orbit.
            </p>
          </section>

          {/* The Problem */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-2">The Kessler Syndrome Problem</h3>
            <p className="text-[var(--color-text-muted)]">
              Over 30,000 tracked objects orbit Earth, with millions of smaller untracked
              fragments. As mega-constellations like Starlink add thousands more satellites,
              NASA's Conjunction Assessment office (CARA) must screen hundreds of conjunction
              data messages (CDMs) daily to decide which warrant expensive avoidance maneuvers.
              Current methods rely heavily on analyst judgment. Machine learning can help
              triage these events, flagging the highest-risk conjunctions for human review.
            </p>
          </section>

          {/* Three Models */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-3">Three Modeling Approaches</h3>
            <div className="space-y-4">
              {/* Baseline */}
              <div className="rounded-lg border border-[var(--color-border)] p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-[#888]" />
                  <h4 className="font-semibold text-xs uppercase tracking-widest text-[var(--color-text-muted)]">
                    Naive Baseline
                  </h4>
                </div>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  <strong className="text-[var(--color-text)]">Orbital Shell Logistic Regression</strong> &mdash;
                  Predicts collision probability using only altitude-based features (apogee,
                  perigee, orbital shell). Demonstrates why simple geometric features alone are
                  insufficient for conjunction screening. Provides the lower bound for model comparison.
                </p>
              </div>

              {/* XGBoost */}
              <div className="rounded-lg border border-[var(--color-accent)]/30 p-4 bg-[var(--color-accent)]/5">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-[var(--color-accent)]" />
                  <h4 className="font-semibold text-xs uppercase tracking-widest text-[var(--color-accent)]">
                    Classical ML
                  </h4>
                </div>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  <strong className="text-[var(--color-text)]">Gradient Boosted Trees (XGBoost)</strong> &mdash;
                  Uses hand-engineered summary statistics from CDM sequences: miss distance
                  trends, risk evolution, covariance determinants, and observation quality metrics.
                  Achieves AUC-PR of 0.988 on the ESA Kelvins benchmark (offline evaluation).
                  The production system uses logistic regression + LSTM on Space-Track CDMs (different feature set).
                </p>
              </div>

              {/* PI-TFT */}
              <div className="rounded-lg border border-[#8b5cf6]/30 p-4 bg-[#8b5cf6]/5">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-[#8b5cf6]" />
                  <h4 className="font-semibold text-xs uppercase tracking-widest text-[#8b5cf6]">
                    Deep Learning
                  </h4>
                </div>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  <strong className="text-[var(--color-text)]">Physics-Informed Temporal Fusion Transformer (PI-TFT)</strong> &mdash;
                  A novel architecture that processes raw CDM time series with attention
                  mechanisms. Key innovations:
                </p>
                <ul className="text-xs text-[var(--color-text-muted)] mt-2 space-y-1.5 ml-4 list-disc leading-relaxed">
                  <li>
                    <strong className="text-[var(--color-text)]">Variable Selection Networks</strong> &mdash;
                    Learned feature gating identifies which CDM fields matter most per event
                  </li>
                  <li>
                    <strong className="text-[var(--color-text)]">Temporal Delta Features</strong> &mdash;
                    First-order differences capture how quantities evolve between CDM updates
                  </li>
                  <li>
                    <strong className="text-[var(--color-text)]">Density-Augmented Features</strong> &mdash;
                    CRASH Clock-inspired orbital shell density and collision rate features
                    encode population-level risk context
                  </li>
                  <li>
                    <strong className="text-[var(--color-text)]">Attention-Weighted Pooling</strong> &mdash;
                    Self-attention over the CDM sequence lets the model focus on the most
                    informative updates
                  </li>
                  <li>
                    <strong className="text-[var(--color-text)]">Dual Prediction Heads</strong> &mdash;
                    Jointly predicts collision probability and miss distance for multi-task learning
                  </li>
                </ul>
              </div>
            </div>
          </section>

          {/* Conformal Prediction */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-2">Uncertainty Quantification</h3>
            <p className="text-[var(--color-text-muted)]">
              Raw model probabilities are often miscalibrated. PANACEA implements
              <strong className="text-[var(--color-text)]"> split-conformal prediction</strong> for
              the ESA Kelvins benchmark, producing prediction sets with distribution-free
              coverage guarantees. This technique outputs a set of risk tiers (e.g.,
              LOW, MODERATE) that provably contains the true risk level
              at a specified confidence level (e.g., 90%). This directly addresses
              NASA CARA's criticism about uncertainty quantification in ML-based
              collision risk assessment.
              <em> Note: implemented for Kelvins benchmark; not yet deployed in production pipeline.</em>
            </p>
          </section>

          {/* Live Globe */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-2">Live 3D Visualization</h3>
            <p className="text-[var(--color-text-muted)]">
              The interactive globe displays real-time positions of 25,000+ tracked objects
              using TLE data from CelesTrak. Each satellite's position is computed via
              SGP4 orbital propagation (satellite.js) and smoothly animated at 60fps using
              velocity-based dead-reckoning between propagation updates. Color coding
              distinguishes active satellites, debris fields (Cosmos 2251, Iridium 33,
              Fengyun 1C), space stations, and mega-constellations.
            </p>
          </section>

          {/* Dataset */}
          <section>
            <h3 className="text-base font-semibold tracking-tight mb-2">Training Data</h3>
            <p className="text-[var(--color-text-muted)]">
              Models are trained on the ESA Kelvins Collision Avoidance Challenge dataset:
              162,634 conjunction data messages across 13,154 unique conjunction events,
              with 103 numerical features covering orbital mechanics, covariance matrices,
              space weather indices, and observation quality metrics. This is real operational
              data from ESA's Space Debris Office.
            </p>
          </section>

          {/* Credits */}
          <section className="border-t border-[var(--color-border)] pt-4">
            <p className="text-xs text-[var(--color-text-muted)] text-center leading-relaxed">
              PANACEA &mdash; AIPI 540 Deep Learning Applications, Duke University
              <br />
              Built with React, Three.js, PyTorch, and FastAPI
              <br />
              Satellite data from CelesTrak | CDM dataset from ESA Kelvins Challenge
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
