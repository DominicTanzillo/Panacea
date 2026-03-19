import { useState, useEffect } from 'react';
import { Overlay } from './Overlay';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Cell, LineChart, Line,
} from 'recharts';

interface ModelZooPageProps {
  visible: boolean;
  onClose: () => void;
}

/* ── CV data types ─────────────────────────────────────── */
interface CVResults {
  n_folds: number;
  n_pairs: number;
  n_positive: number;
  positive_rate: number;
  results: {
    lr: { f1: number; precision: number; recall: number; accuracy: number; auc_pr?: number }[];
    lstm: { f1: number; precision: number; recall: number; accuracy: number; mae_log10_pc?: number; correlation?: number; ece?: number }[];
    ensemble: { f1: number; precision: number; recall: number; accuracy: number; tp?: number; fp?: number; fn?: number; tn?: number }[];
  };
}

/* ── Helpers ────────────────────────────────────────────── */
function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

const ACCENT = {
  lr: '#4fff8a',
  lstm: '#4f8aff',
  ensemble: '#8b5cf6',
  gnn: '#ff9f43',
  conformal: '#a29bfe',
  summarizer: '#fd79a8',
};

/* ── GNN Graph Illustration (SVG) ──────────────────────── */
function GNNGraphViz() {
  // Hardcoded sample conjunction network for illustration
  const nodes = [
    { x: 200, y: 60, label: 'COSMOS\n2251 DEB', type: 'debris' },
    { x: 80, y: 140, label: 'STARLINK\n-1234', type: 'payload' },
    { x: 320, y: 140, label: 'FENGYUN\n1C DEB', type: 'debris' },
    { x: 140, y: 240, label: 'CZ-6A\nR/B', type: 'rb' },
    { x: 260, y: 240, label: 'STARLINK\n-5678', type: 'payload' },
    { x: 200, y: 320, label: 'IRIDIUM\n33 DEB', type: 'debris' },
    { x: 60, y: 300, label: 'ISS', type: 'payload' },
    { x: 340, y: 300, label: 'NOAA 17\nDEB', type: 'debris' },
  ];

  const edges = [
    { from: 0, to: 1, risk: 0.82 },
    { from: 0, to: 2, risk: 0.91 },
    { from: 1, to: 3, risk: 0.35 },
    { from: 2, to: 4, risk: 0.67 },
    { from: 3, to: 5, risk: 0.44 },
    { from: 4, to: 5, risk: 0.78 },
    { from: 1, to: 6, risk: 0.21 },
    { from: 2, to: 7, risk: 0.55 },
    { from: 5, to: 7, risk: 0.33 },
  ];

  const typeColor: Record<string, string> = {
    payload: '#4fff8a',
    debris: '#ff4f5a',
    rb: '#ffb84f',
  };

  return (
    <svg viewBox="0 0 400 380" className="w-full max-w-md mx-auto">
      {/* Edges */}
      {edges.map((e, i) => {
        const a = nodes[e.from], b = nodes[e.to];
        const riskColor = e.risk > 0.7 ? '#ff4f5a' : e.risk > 0.4 ? '#ffb84f' : '#4f8aff55';
        return (
          <line
            key={i}
            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={riskColor}
            strokeWidth={e.risk > 0.7 ? 2.5 : 1.5}
            strokeOpacity={0.6}
          />
        );
      })}
      {/* Edge risk labels */}
      {edges.filter(e => e.risk > 0.6).map((e, i) => {
        const a = nodes[e.from], b = nodes[e.to];
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        return (
          <text key={`el-${i}`} x={mx} y={my - 6} textAnchor="middle" fill="#888" fontSize={8}>
            {(e.risk * 100).toFixed(0)}%
          </text>
        );
      })}
      {/* Nodes */}
      {nodes.map((n, i) => (
        <g key={i}>
          <circle cx={n.x} cy={n.y} r={18} fill={typeColor[n.type] + '20'} stroke={typeColor[n.type]} strokeWidth={1.5} />
          <text x={n.x} y={n.y - 4} textAnchor="middle" fill={typeColor[n.type]} fontSize={7} fontWeight="bold">
            {n.label.split('\n')[0]}
          </text>
          <text x={n.x} y={n.y + 6} textAnchor="middle" fill="#888" fontSize={6}>
            {n.label.split('\n')[1] || ''}
          </text>
        </g>
      ))}
      {/* Legend */}
      <g transform="translate(10, 360)">
        {[{ c: '#4fff8a', l: 'Payload' }, { c: '#ff4f5a', l: 'Debris' }, { c: '#ffb84f', l: 'R/B' }].map((item, i) => (
          <g key={i} transform={`translate(${i * 90}, 0)`}>
            <circle cx={6} cy={0} r={4} fill={item.c + '40'} stroke={item.c} strokeWidth={1} />
            <text x={14} y={3} fill="#888" fontSize={8}>{item.l}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

/* ── Model Detail Cards ────────────────────────────────── */

function CVComparisonChart({ cv }: { cv: CVResults | null }) {
  if (!cv) return <div className="text-xs text-[var(--color-text-muted)] text-center p-4">Loading cross-validation data...</div>;

  const data = [
    {
      model: 'LogReg',
      F1: +avg(cv.results.lr.map(r => r.f1)).toFixed(3),
      Precision: +avg(cv.results.lr.map(r => r.precision)).toFixed(3),
      Recall: +avg(cv.results.lr.map(r => r.recall)).toFixed(3),
    },
    {
      model: 'BiLSTM',
      F1: +avg(cv.results.lstm.map(r => r.f1)).toFixed(3),
      Precision: +avg(cv.results.lstm.map(r => r.precision)).toFixed(3),
      Recall: +avg(cv.results.lstm.map(r => r.recall)).toFixed(3),
    },
    {
      model: 'Ensemble',
      F1: +avg(cv.results.ensemble.map(r => r.f1)).toFixed(3),
      Precision: +avg(cv.results.ensemble.map(r => r.precision)).toFixed(3),
      Recall: +avg(cv.results.ensemble.map(r => r.recall)).toFixed(3),
    },
  ];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        <XAxis dataKey="model" tick={{ fill: '#aaa', fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fill: '#666', fontSize: 10 }} />
        <Tooltip contentStyle={{ background: '#14141e', border: '1px solid #2a2a3a', borderRadius: 4, fontSize: 11 }} />
        <Bar dataKey="F1" radius={[2, 2, 0, 0]}>
          <Cell fill={ACCENT.lr} />
          <Cell fill={ACCENT.lstm} />
          <Cell fill={ACCENT.ensemble} />
        </Bar>
        <Bar dataKey="Precision" radius={[2, 2, 0, 0]} opacity={0.6}>
          <Cell fill={ACCENT.lr} />
          <Cell fill={ACCENT.lstm} />
          <Cell fill={ACCENT.ensemble} />
        </Bar>
        <Bar dataKey="Recall" radius={[2, 2, 0, 0]} opacity={0.35}>
          <Cell fill={ACCENT.lr} />
          <Cell fill={ACCENT.lstm} />
          <Cell fill={ACCENT.ensemble} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function EnsembleRadar({ cv }: { cv: CVResults | null }) {
  if (!cv) return null;
  const data = [
    { metric: 'F1', lr: avg(cv.results.lr.map(r => r.f1)), ensemble: avg(cv.results.ensemble.map(r => r.f1)) },
    { metric: 'Precision', lr: avg(cv.results.lr.map(r => r.precision)), ensemble: avg(cv.results.ensemble.map(r => r.precision)) },
    { metric: 'Recall', lr: avg(cv.results.lr.map(r => r.recall)), ensemble: avg(cv.results.ensemble.map(r => r.recall)) },
    { metric: 'Accuracy', lr: avg(cv.results.lr.map(r => r.accuracy)), ensemble: avg(cv.results.ensemble.map(r => r.accuracy)) },
  ];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#333" />
        <PolarAngleAxis dataKey="metric" tick={{ fill: '#aaa', fontSize: 10 }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} />
        <Radar name="LogReg" dataKey="lr" stroke={ACCENT.lr} fill={ACCENT.lr} fillOpacity={0.15} strokeWidth={2} />
        <Radar name="Ensemble" dataKey="ensemble" stroke={ACCENT.ensemble} fill={ACCENT.ensemble} fillOpacity={0.15} strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function ConformalChart() {
  const data = [
    { name: 'Target', coverage: 90 },
    { name: 'Regression', coverage: 90.8 },
    { name: 'Classification', coverage: 91.7 },
  ];

  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        <XAxis type="number" domain={[85, 95]} tick={{ fill: '#666', fontSize: 10 }} tickFormatter={v => `${v}%`} />
        <YAxis type="category" dataKey="name" tick={{ fill: '#aaa', fontSize: 10 }} />
        <Tooltip contentStyle={{ background: '#14141e', border: '1px solid #2a2a3a', borderRadius: 4, fontSize: 11 }} />
        <Bar dataKey="coverage" radius={[0, 2, 2, 0]}>
          <Cell fill="#666" />
          <Cell fill={ACCENT.conformal} />
          <Cell fill={ACCENT.conformal} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function AutoregressiveChart() {
  // Simulated actual vs predicted Pc trajectory
  const data = [
    { cdm: 1, actual: -3.8, predicted: -3.9 },
    { cdm: 2, actual: -3.6, predicted: -3.7 },
    { cdm: 3, actual: -3.4, predicted: -3.5 },
    { cdm: 4, actual: -3.2, predicted: -3.3 },
    { cdm: 5, actual: -3.1, predicted: -3.2 },
    { cdm: 6, actual: null, predicted: -3.0 },
    { cdm: 7, actual: null, predicted: -2.9 },
    { cdm: 8, actual: null, predicted: -2.85 },
  ];

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        <XAxis dataKey="cdm" tick={{ fill: '#666', fontSize: 10 }} label={{ value: 'CDM Update', position: 'insideBottom', offset: -2, fill: '#555', fontSize: 9 }} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} domain={[-4, -2.5]} label={{ value: 'log\u2081\u2080(Pc)', angle: -90, position: 'insideLeft', fill: '#555', fontSize: 9 }} />
        <Tooltip contentStyle={{ background: '#14141e', border: '1px solid #2a2a3a', borderRadius: 4, fontSize: 11 }} />
        <Line type="monotone" dataKey="actual" stroke="#4fff8a" strokeWidth={2} dot={{ r: 3 }} name="Actual" connectNulls={false} />
        <Line type="monotone" dataKey="predicted" stroke="#ff9f43" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3 }} name="Predicted" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ── Main Page ─────────────────────────────────────────── */

export function ModelZooPage({ visible, onClose }: ModelZooPageProps) {
  const [cv, setCv] = useState<CVResults | null>(null);

  useEffect(() => {
    if (!visible) return;
    fetch('./cv_results.json')
      .then(r => r.ok ? r.json() : null)
      .then(setCv)
      .catch(() => setCv(null));
  }, [visible]);

  return (
    <Overlay
      visible={visible}
      onClose={onClose}
      title="Model Zoo"
      subtitle={cv ? `${cv.n_folds}-fold CV on ${cv.n_pairs} pairs (${(cv.positive_rate * 100).toFixed(1)}% positive)` : '6 models for collision risk assessment'}
      maxWidth="900px"
    >
        <div className="p-6 space-y-8">

          {/* Cross-validation comparison */}
          <section>
            <SectionTitle>5-Fold Cross-Validation Comparison</SectionTitle>
            <p className="text-[10px] text-[var(--color-text-muted)] mb-4">
              Bar opacity: F1 (solid), Precision (60%), Recall (35%). Each model color-coded.
            </p>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="text-[10px] text-[var(--color-text-muted)] mb-2 uppercase tracking-wider">Metrics by Model</div>
                <CVComparisonChart cv={cv} />
              </div>
              <div>
                <div className="text-[10px] text-[var(--color-text-muted)] mb-2 uppercase tracking-wider">LogReg vs Ensemble (Radar)</div>
                <EnsembleRadar cv={cv} />
              </div>
            </div>
          </section>

          {/* Per-model details */}
          <section>
            <SectionTitle>Model Details</SectionTitle>
            <div className="space-y-5">

              {/* Ensemble */}
              <ModelCard
                name="BiLSTM Ensemble"
                tag="Deep Learning"
                color={ACCENT.ensemble}
                description="Combines Logistic Regression (40%), BiLSTM with Kelvins transfer learning (30%), and regression signal (30%). Focal loss for class imbalance, temperature-calibrated probabilities."
                keyMetric="98.7% Recall"
                keyMetricLabel="Nearly zero missed escalation events"
              >
                <div className="text-[10px] text-[var(--color-text-muted)] mb-2">
                  The ensemble achieves the highest recall of any model because it combines three independent signals.
                  Even when one model misses, the others catch it.
                </div>
                <EnsembleRadar cv={cv} />
              </ModelCard>

              {/* GNN */}
              <ModelCard
                name="Graph Neural Network"
                tag="Graph ML"
                color={ACCENT.gnn}
                description="GraphSAGE on conjunction network. Satellites are nodes, conjunction pairs are edges. Message-passing aggregates risk from orbital neighborhood."
                keyMetric="1,212 Nodes"
                keyMetricLabel="F1=0.783 on edge classification"
              >
                <div className="text-[10px] text-[var(--color-text-muted)] mb-3">
                  A satellite in a congested orbital regime (many simultaneous conjunctions) gets elevated risk signal from neighbors.
                  Per-pair models cannot capture this network effect.
                </div>
                <GNNGraphViz />
                <div className="text-[9px] text-[var(--color-text-muted)] text-center mt-2">
                  Sample conjunction network. Red edges = high collision risk. Node color = object type.
                </div>
              </ModelCard>

              {/* Autoregressive */}
              <ModelCard
                name="Autoregressive Forecaster"
                tag="Seq-to-Seq"
                color={ACCENT.gnn}
                description="BiLSTM predicts the next CDM update (Pc, miss distance, time-to-TCA). Multi-step rollouts with Monte Carlo dropout uncertainty."
                keyMetric="r = 0.931"
                keyMetricLabel="MAE = 0.094 log\u2081\u2080(Pc)"
              >
                <div className="text-[10px] text-[var(--color-text-muted)] mb-2">
                  Given CDM updates 1...k, forecast CDM k+1. Dashed line shows predicted future trajectory.
                </div>
                <AutoregressiveChart />
              </ModelCard>

              {/* Conformal */}
              <ModelCard
                name="Conformal Prediction"
                tag="Uncertainty Quantification"
                color={ACCENT.conformal}
                description="Split-conformal prediction provides distribution-free coverage guarantees. No assumptions about the data distribution."
                keyMetric="90.8%"
                keyMetricLabel="Empirical coverage at 90% target"
              >
                <div className="text-[10px] text-[var(--color-text-muted)] mb-2">
                  NASA CARA requires calibrated uncertainty before adopting ML operationally.
                  Conformal prediction guarantees that the true Pc falls within our interval at least 90% of the time.
                </div>
                <ConformalChart />
              </ModelCard>

              {/* LR */}
              <ModelCard
                name="Logistic Regression"
                tag="Production Baseline"
                color={ACCENT.lr}
                description="20 engineered features from CDM sequences: Pc trend, miss distance slope, acceleration, RCS, debris flag. Retrained daily on resolved conjunction pairs."
                keyMetric="F1 = 0.854"
                keyMetricLabel="93.2% accuracy on temporal test split"
              >
                <div className="text-[10px] text-[var(--color-text-muted)]">
                  The strongest single model. With 670 pairs and 20 features, logistic regression is hard to beat.
                  The deep learning models add value through ensemble diversity and uncertainty quantification, not raw accuracy.
                </div>
              </ModelCard>

              {/* Summarizer */}
              <ModelCard
                name="Event Summarizer"
                tag="NLP"
                color={ACCENT.summarizer}
                description="Template-based NLP generates human-readable risk assessments from model predictions, attention weights, and CDM data. Optional Claude API enhancement."
                keyMetric="NLP"
                keyMetricLabel="Structured data to natural language"
              >
                <div className="bg-[var(--color-bg)] p-3 text-[10px] text-[var(--color-text-muted)] leading-relaxed" style={{ borderRadius: 4, borderLeft: '3px solid #fd79a8' }}>
                  <span className="text-[var(--color-text)] font-semibold">Sample output: </span>
                  COSMOS 2251 DEB vs STARLINK-1234: ELEVATED RISK. Current collision probability is 1 in 3,125 with a miss distance of 0.4 km.
                  Risk is escalating across 5 CDM updates (+15%/update). The ensemble model gives a 68% chance of exceeding the maneuver threshold.
                  BiLSTM predicts max Pc of 10^-3.1 — ABOVE the 5e-4 threshold. Recommend: active monitoring at 6-hour intervals.
                </div>
              </ModelCard>
            </div>
          </section>
        </div>
    </Overlay>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-sm font-semibold tracking-tight text-[var(--color-text)] pb-2 mb-2 border-b border-[var(--color-border)]/30">
      {children}
    </h3>
  );
}

function ModelCard({
  name, tag, color, description, keyMetric, keyMetricLabel, children,
}: {
  name: string; tag: string; color: string; description: string;
  keyMetric: string; keyMetricLabel: string; children?: React.ReactNode;
}) {
  return (
    <div className="bg-[var(--color-surface-2)] p-5" style={{ borderRadius: 4, borderLeft: `3px solid ${color}` }}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-[var(--color-text)]">{name}</span>
            <span className="text-[8px] font-medium px-1.5 py-0.5 uppercase tracking-wider" style={{ color, background: `${color}15`, borderRadius: 3 }}>
              {tag}
            </span>
          </div>
          <p className="text-[10px] text-[var(--color-text-muted)] leading-relaxed max-w-lg">{description}</p>
        </div>
        <div className="text-right shrink-0 ml-4">
          <div className="text-xl font-bold font-mono" style={{ color }}>{keyMetric}</div>
          <div className="text-[8px] text-[var(--color-text-muted)] uppercase tracking-wider">{keyMetricLabel}</div>
        </div>
      </div>
      {children}
    </div>
  );
}
