import { FullPagePanel } from './Overlay';

interface AboutPageProps {
  visible: boolean;
  onClose: () => void;
}

export function AboutPage({ visible, onClose }: AboutPageProps) {
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
        </Section>

        {/* Technical approach */}
        <Section title="Technical Approach">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <ApproachRow label="01" title="Logistic Regression" desc="23 engineered features from CDM sequences. Retrained daily on resolved pairs. F1 = 0.854, the strongest single model." />
            <ApproachRow label="02" title="BiLSTM with Transfer Learning" desc="Pre-trained on 13K ESA Kelvins events, fine-tuned on 670 Space-Track pairs. Focal loss for class imbalance. Attention-weighted pooling." />
            <ApproachRow label="03" title="Ensemble" desc="40% LR + 30% BiLSTM + 30% regression signal. Achieves 98.7% recall, misses only 2 out of 152 positive events." />
            <ApproachRow label="04" title="Graph Neural Network" desc="GraphSAGE on the conjunction network. Captures orbital neighborhood effects that per-pair models cannot." />
            <ApproachRow label="05" title="Autoregressive Forecaster" desc="Predicts next CDM update. Multi-step rollouts with Monte Carlo dropout uncertainty. Correlation r = 0.931." />
            <ApproachRow label="06" title="Conformal Prediction" desc="Distribution-free coverage guarantees. 90.8% empirical coverage at 90% target with no distributional assumptions." />
          </div>
        </Section>

        {/* Data */}
        <Section title="Data Sources">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <DataCard title="ESA Kelvins Challenge" stats="162,634 CDMs \u00b7 13,154 events \u00b7 103 features" desc="Real operational data from ESA's Space Debris Office. Used for pre-training and benchmark evaluation." />
            <DataCard title="Space-Track CDMs" stats="670 pairs \u00b7 3,937 CDMs \u00b7 5 features" desc="Public conjunction data from the 18th Space Defense Squadron. Updated daily. Used for production predictions." />
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
