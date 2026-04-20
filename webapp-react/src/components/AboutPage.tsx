import { useEffect, useState } from 'react';
import type { OverlayView } from '../App';
import { FullPagePanel } from './Overlay';

interface LiveStats {
  nPairs: number;
  nCdms: number;
  lrF1: number;
  nSpaceTrackPairs: number;
  positiveRate: number;
  ensembleRecall: number;
  ensembleFnCount: number;
  ensembleTpCount: number;
  conformalCoverage: number;
  arCorrelation: number;
}

const FALLBACK: LiveStats = {
  nPairs: 866, nCdms: 5100, lrF1: 0.854, nSpaceTrackPairs: 670,
  positiveRate: 0.251,
  ensembleRecall: 98.7, ensembleFnCount: 2, ensembleTpCount: 152,
  conformalCoverage: 90.8, arCorrelation: 0.931,
};

function useLiveStats(): LiveStats {
  const [stats, setStats] = useState<LiveStats>(FALLBACK);
  useEffect(() => {
    Promise.all([
      fetch('/cdm_forecast.json').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('/pipeline_stats.json').then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([forecast, pipeline]) => {
      if (!forecast && !pipeline) return;
      const s: LiveStats = { ...FALLBACK };
      if (pipeline?.cdm_stats) {
        s.nCdms = pipeline.cdm_stats.total_cdms ?? s.nCdms;
      }
      if (pipeline?.forecast_model) {
        const fm = pipeline.forecast_model;
        s.nPairs = fm.n_pairs_total ?? s.nPairs;
        s.nSpaceTrackPairs = fm.n_pairs_total ?? s.nSpaceTrackPairs;
        if (fm.positive_rate) s.positiveRate = fm.positive_rate;
        if (fm.test?.f1) s.lrF1 = fm.test.f1;
      }
      if (forecast?.track_record) {
        const tr = forecast.track_record;
        const tp = tr.n_true_positives ?? 0;
        const fn = tr.n_false_negatives ?? 0;
        s.ensembleTpCount = tp;
        s.ensembleFnCount = fn;
        s.ensembleRecall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : s.ensembleRecall;
      }
      if (forecast?.conformal) {
        const cov = forecast.conformal.regression_coverage;
        if (cov != null) s.conformalCoverage = cov * 100;
      }
      if (forecast?.autoregressive_metrics?.correlation_pc != null) {
        s.arCorrelation = forecast.autoregressive_metrics.correlation_pc;
      }
      setStats(s);
    });
  }, []);
  return stats;
}

interface AboutPageProps {
  visible: boolean;
  onClose: () => void;
  onNavigate?: (view: OverlayView) => void;
}

export function AboutPage({ visible, onClose, onNavigate }: AboutPageProps) {
  const live = useLiveStats();

  const navLink = (label: string, view: OverlayView) => (
    <button
      onClick={() => { onNavigate?.(view); }}
      style={{
        background: 'rgba(59,130,246,0.1)',
        border: '1px solid rgba(59,130,246,0.3)',
        borderRadius: 6,
        color: '#3b82f6',
        cursor: 'pointer',
        fontFamily: 'inherit',
        fontSize: 13,
        fontWeight: 500,
        padding: '4px 12px',
        marginLeft: 6,
      }}
    >
      {label}
    </button>
  );

  return (
    <FullPagePanel visible={visible} onClose={onClose} title="About PANACEA" subtitle="Orbital collision prediction for the age of mega-constellations" maxWidth={860}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 32, paddingTop: 24 }}>

        {/* The problem */}
        <Section title="The Problem">
          <p>
            Over 30,000 tracked objects orbit Earth. Every 5 days, two of them collide,
            creating thousands of new fragments, each a future collision. This is the <strong style={{ color: '#e8e8f0' }}>Kessler Syndrome</strong>:
            a cascading chain reaction that could make entire orbital shells unusable.
          </p>
          <p>
            NASA's Conjunction Assessment office (CARA) screens hundreds of <strong style={{ color: '#e8e8f0' }}>Conjunction Data Messages</strong> (CDMs)
            daily to decide which events warrant expensive avoidance maneuvers. Current methods rely heavily
            on analyst judgment. Machine learning can triage these events, flagging the highest-risk
            conjunctions for human review.
          </p>
        </Section>

        {/* What PANACEA does */}
        <Section title="What PANACEA Does">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginTop: 8 }}>
            <FeatureCard
              title="Predicts Escalation"
              desc="Given a sequence of CDM updates, predicts whether collision probability will exceed the maneuver planning threshold before time of closest approach."
            />
            <FeatureCard
              title="Quantifies Uncertainty"
              desc="Conformal prediction provides distribution-free coverage guarantees. Operators know how much to trust each forecast."
            />
            <FeatureCard
              title="Runs Daily"
              desc="Automated pipeline fetches real CDMs from Space-Track, runs inference, and deploys updated forecasts to this dashboard."
            />
          </div>
          {onNavigate && (
            <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 13, color: '#55556a', lineHeight: '28px' }}>Explore:</span>
              {navLink('View Forecasts', 'forecast')}
              {navLink('See Alerts', 'alerts')}
              {navLink('Pipeline Stats', 'dashboard')}
              {navLink('Model Details', 'models')}
            </div>
          )}
        </Section>

        {/* Technical approach */}
        <Section title="Technical Approach">
          <p>
            PANACEA is a <strong style={{ color: '#e8e8f0' }}>screening mechanism</strong>: catch every escalating event, then run detailed analysis on the flagged subset.
            The naive approach — assume nothing escalates — gets {((1 - live.positiveRate) * 100).toFixed(0)}% accuracy but catches zero events.
            Sensitivity is what matters.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <ApproachRow label="00" title="Naive Baseline" desc={`Predict no escalation. Accuracy = ${((1 - live.positiveRate) * 100).toFixed(0)}%, but recall = 0%. Misses every dangerous event.`} />
            <ApproachRow label="01" title="Logistic Regression" desc={`23 engineered features from CDM sequences. Retrained daily on resolved pairs. F1 = ${live.lrF1.toFixed(3)}, the strongest single model.`} />
            <ApproachRow label="02" title="BiLSTM with Transfer Learning" desc={`Pre-trained on 13K ESA Kelvins events, fine-tuned on ${live.nSpaceTrackPairs.toLocaleString()} Space-Track pairs. Focal loss for class imbalance. Attention-weighted pooling.`} />
            <ApproachRow label="03" title="Ensemble" desc={`40% LR + 30% BiLSTM + 30% regression signal. Achieves ${live.ensembleRecall.toFixed(1)}% recall${live.ensembleFnCount === 0 ? `, missing zero out of ${live.ensembleTpCount} positive events` : `, misses only ${live.ensembleFnCount} out of ${live.ensembleTpCount + live.ensembleFnCount} positive events`}.`} />
            <ApproachRow label="04" title="Graph Neural Network" desc="GraphSAGE on the conjunction network. Captures orbital neighborhood effects that per-pair models cannot." />
            <ApproachRow label="05" title="Autoregressive Forecaster" desc={`Predicts next CDM update. Multi-step rollouts with Monte Carlo dropout uncertainty. Correlation r = ${live.arCorrelation.toFixed(3)}.`} />
            <ApproachRow label="06" title="Conformal Prediction" desc={`Distribution-free coverage guarantees. ${live.conformalCoverage.toFixed(1)}% empirical coverage at 90% target with no distributional assumptions.`} />
          </div>
        </Section>

        {/* Data */}
        <Section title="Data Sources">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <DataCard title="ESA Kelvins Challenge" stats="162,634 CDMs · 13,154 events · 103 features" desc="Real operational data from ESA's Space Debris Office. Used for pre-training and benchmark evaluation." />
            <DataCard title="Space-Track CDMs" stats={`${live.nPairs.toLocaleString()}+ pairs · ${live.nCdms.toLocaleString()}+ CDMs · 23 features`} desc="Public conjunction data from the 18th Space Defense Squadron. Updated twice daily via automated pipeline." />
          </div>
        </Section>

        {/* Visualization */}
        <Section title="Real-time Visualization">
          <p>
            The interactive globe displays real-time positions of 25,000+ tracked objects using TLE data
            from CelesTrak. Each satellite's position is computed via SGP4 orbital propagation and smoothly
            animated at 60fps using velocity-based dead-reckoning. Click any conjunction pair to watch the
            approach unfold on the globe with trajectory trails and separation distance charts.
          </p>
        </Section>

        {/* Credits */}
        <div style={{ borderTop: '1px solid #1e1e2c', paddingTop: 20, textAlign: 'center' }}>
          <p style={{ fontSize: 14, color: '#55556a', lineHeight: 1.8 }}>
            <strong style={{ color: '#7c7c96' }}>PANACEA</strong> &middot; AIPI 540 Deep Learning Applications, Duke University
            <br />
            Built with React, Three.js, PyTorch, and FastAPI
            <br />
            Satellite data: CelesTrak &middot; CDM data: Space-Track 18th SDS &middot; Benchmark: ESA Kelvins
          </p>
        </div>
      </div>
    </FullPagePanel>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ borderBottom: '1px solid #1e1e2c', paddingBottom: 24 }}>
      <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 12 }}>
        {title}
      </div>
      <div style={{ fontSize: 14, color: '#7c7c96', lineHeight: 1.7, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {children}
      </div>
    </div>
  );
}

function FeatureCard({ title, desc }: { title: string; desc: string }) {
  return (
    <div style={{ padding: '16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 13, color: '#55556a', lineHeight: 1.5 }}>{desc}</div>
    </div>
  );
}

function ApproachRow({ label, title, desc }: { label: string; title: string; desc: string }) {
  return (
    <div style={{ display: 'flex', gap: 16, padding: '12px 16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#3b82f6', flexShrink: 0, width: 24 }}>
        {label}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0', marginBottom: 2 }}>{title}</div>
        <div style={{ fontSize: 13, color: '#55556a', lineHeight: 1.5 }}>{desc}</div>
      </div>
    </div>
  );
}

function DataCard({ title, stats, desc }: { title: string; stats: string; desc: string }) {
  return (
    <div style={{ padding: '16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: '#3b82f6', marginBottom: 6 }}>{stats}</div>
      <div style={{ fontSize: 13, color: '#55556a', lineHeight: 1.5 }}>{desc}</div>
    </div>
  );
}
